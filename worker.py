"""Resident MLX worker for Qwen-Image-2.1 (edit + txt2img).

Runs inside the mflux interpreter (NOT inside ComfyUI's Python).  It loads
``QwenImage21Edit`` once, then serves newline-delimited JSON requests on stdin
and answers on stdout.  Everything else -- mflux logs, tqdm bars, warnings --
is redirected to stderr, which the node pack writes to ``logs/worker.log``.

Why a resident worker instead of one CLI call per generation:

* the 4-bit package is ~9.6 GB; reloading it for every prompt costs 40-90 s,
* a single loaded model serves both text-to-image (no reference) and
  image editing (1..10 references), so the graph's two branches share it,
* the transformer stays warm in unified memory.

Protocol (one JSON object per line, UTF-8):

    -> {"id": "...", "op": "generate", "params": {...}}
    <- {"type": "ready", ...}
    <- {"type": "status", "phase": "encoding" | "denoising" | "decoding"}
    <- {"type": "progress", "step": 3, "total": 40}
    <- {"type": "result", "path": "...", "width": 1024, "height": 1024, ...}
    <- {"type": "error", "message": "...", "traceback": "..."}

The worker is deliberately single-threaded: ComfyUI serialises runs per model
on the node side, and MLX generation is not safe to overlap anyway.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

# ---------------------------------------------------------------- protocol ---
# Keep a private handle on the real stdout for protocol frames, then point fd 1
# at stderr so any ``print`` from mflux/transformers/tokenizers cannot corrupt
# the stream the node parses.
_PROTO = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
os.dup2(2, 1)
sys.stdout = sys.stderr


def emit(**frame) -> None:
    _PROTO.write(json.dumps(frame, ensure_ascii=False) + "\n")
    _PROTO.flush()


def log(message: str) -> None:
    print(f"[qwen21-mlx] {message}", file=sys.stderr, flush=True)


class ProgressCallback:
    """mflux callback: turn its denoise loop into ComfyUI progress frames."""

    def __init__(self) -> None:
        self.total = 0
        self.last = -1

    def call_before_loop(self, seed, prompt, latents, config, **_) -> None:
        self.total = int(getattr(config, "num_inference_steps", 0) or 0)
        self.last = -1
        emit(type="status", phase="denoising", total=self.total)

    def call_in_loop(self, t, seed, prompt, latents, config, time_steps=None, **_) -> None:
        total = self.total or int(getattr(config, "num_inference_steps", 0) or 0)
        # ``t`` is the index into the (possibly shortened) time_steps slice;
        # tqdm's ``n`` tracks the absolute step when strength < 1.
        step = int(getattr(time_steps, "n", t + 1) or (t + 1))
        if step == self.last:
            return
        self.last = step
        emit(type="progress", step=step, total=total)

    def call_after_loop(self, seed, prompt, latents, config, **_) -> None:
        emit(type="status", phase="decoding")


class EncoderReleaser:
    """Drop the 4.8 GB Qwen3-VL encoder once the prompt is encoded.

    mflux's CLI does this through its MemorySaver and it is worth ~1.8x on the
    denoise loop: with the encoder resident the 1024² run peaked at 15.0 GB and
    stepped at 14.8 s; released, the same run peaked at 10.3 GB and stepped at
    8.3 s.  The encoder is reloaded from disk (a few seconds) on the next
    request that needs it.
    """

    def __init__(self, model) -> None:
        self.model = model

    def call_before_loop(self, seed, prompt, latents, config, **_) -> None:
        if getattr(self.model, "text_encoder", None) is None:
            return
        self.model.text_encoder = None
        gc.collect()
        import mlx.core as mx

        mx.clear_cache()
        emit(type="status", phase="text-encoder-released")


def _reload_text_encoder(model) -> None:
    """Rebuild just the Qwen3-VL encoder (not the transformer/VAE)."""
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten

    from mflux.models.common.weights.loading.weight_applier import WeightApplier
    from mflux.models.common.weights.loading.weight_loader import WeightLoader
    from mflux.models.qwen21.model.qwen21_text_encoder.text_encoder import QwenImage21TextEncoder
    from mflux.models.qwen21.qwen21_initializer import Qwen21Initializer
    from mflux.models.qwen21.weights.qwen_image21_weight_definition import QwenImage21WeightDefinition

    started = time.time()
    emit(type="status", phase="text-encoder-loading")
    definition = QwenImage21WeightDefinition
    component = next(c for c in definition.get_components() if c.name == "text_encoder")
    module = QwenImage21TextEncoder(model._component_configs["text_encoder"])
    weights = WeightLoader.load_single_local(component, Path(model._checkpoint_path))
    supplied = dict(tree_flatten(weights.components[component.name]))
    supplied = Qwen21Initializer._normalize_text_encoder_weights(supplied, module)
    weights.components[component.name] = tree_unflatten(list(supplied.items()))
    WeightApplier.apply_and_quantize(
        weights=weights,
        models={component.name: module},
        quantize_arg=None,
        weight_definition=definition,
    )
    model.text_encoder = module
    mx.clear_cache()
    log(f"text encoder reloaded in {time.time() - started:.1f}s")


# ------------------------------------------------------------------- model ---


def load_model(args):
    import mlx.core as mx

    from mflux.models.common.resolution.config_resolution import ConfigResolution
    from mflux.models.qwen21.variants.edit.qwen_image_21_edit import QwenImage21Edit

    model_path = str(Path(os.path.expanduser(args.model)).resolve())
    model_config = ConfigResolution.resolve_restricted(
        None,
        "qwen-image-2.1",
        model_path=model_path,
        extra_keys=("qwen-image-2.1-turbo",),
        base_model=args.base_model or None,
    )

    quantize = None if args.quantize in (None, "", "auto") else int(args.quantize)

    lora_paths = None
    lora_scales = None
    if args.lora:
        lora_paths, lora_scales = [], []
        for raw_path, raw_scale in args.lora:
            lora_path = Path(os.path.expanduser(raw_path))
            if not lora_path.is_file():
                raise FileNotFoundError(f"LoRA 文件不存在：{lora_path}")
            lora_paths.append(str(lora_path))
            lora_scales.append(float(raw_scale))

    started = time.time()
    emit(type="status", phase="loading", model=model_path, quantize=args.quantize)
    model = QwenImage21Edit(
        quantize=quantize,
        model_path=model_path,
        model_config=model_config,
        lora_paths=lora_paths,
        lora_scales=lora_scales,
        bake_lora=not getattr(args, "no_bake_lora", False),
    )

    # Optional memory shaping.  The weights stay resident, but MLX's buffer cache
    # would otherwise grow by several GB across runs.
    if args.cache_limit_gb and args.cache_limit_gb > 0:
        mx.set_cache_limit(int(args.cache_limit_gb * 1e9))
    if args.vae_tiling or args.low_ram:
        from mflux.models.common.vae.tiling_config import TilingConfig

        model.tiling_config = TilingConfig(vae_decode_tile_size=args.vae_tile_size)
    mx.clear_cache()

    emit(
        type="ready",
        model=model_path,
        bits=getattr(model, "bits", None),
        load_seconds=round(time.time() - started, 1),
        peak_bytes=mx.get_peak_memory(),
    )
    return model


def generate(model, params: dict, callback: ProgressCallback, keep_text_encoder: bool = False) -> dict:
    import mlx.core as mx

    from mflux.callbacks.callback_registry import CallbackRegistry
    from mflux.models.qwen21.variants.edit import qwen_image_21_edit as edit_mod

    callback.total = 0
    callback.last = -1

    image_paths = [str(p) for p in (params.get("image_paths") or []) if p]
    mask_image = params.get("mask_image") or None
    auto_mask = (params.get("auto_mask") or "").strip() or None
    verify = bool(params.get("verify", False))

    width = params.get("width")
    height = params.get("height")
    width = int(width) if width and int(width) > 0 else None
    height = int(height) if height and int(height) > 0 else None

    if getattr(model, "text_encoder", None) is None:
        _reload_text_encoder(model)

    # One fresh registry per request: the encoder releaser must not pile up, and
    # a failed run must not leave half its callbacks behind.
    model.callbacks = CallbackRegistry()
    model.callbacks.register(callback)
    if not keep_text_encoder and not verify:
        model.callbacks.register(EncoderReleaser(model))

    started = time.time()
    mx.reset_peak_memory()
    emit(type="status", phase="encoding", references=len(image_paths))

    image = model.generate_image(
        seed=int(params.get("seed", 42)),
        prompt=str(params["prompt"]),
        negative_prompt=params.get("negative_prompt"),
        width=width,
        height=height,
        num_inference_steps=int(params.get("steps", 40)),
        guidance=float(params.get("guidance", 1.0)),
        scheduler=str(params.get("scheduler", "linear")),
        image_paths=image_paths,
        output_resolution=int(params.get("output_resolution", 1024)),
        use_kv_cache=bool(params.get("use_kv_cache", True)),
        mask_image=mask_image,
        auto_mask=auto_mask,
        strength=float(params.get("strength", 1.0)),
        use_step_cache=bool(params.get("use_step_cache", False)),
        step_cache_threshold=float(params.get("step_cache_threshold", 0.12)),
        enhance_prompt=bool(params.get("enhance_prompt", False)),
        verify=verify,
        verify_retries=int(params.get("verify_retries", 0)),
    )

    out_path = Path(params["out_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, export_json_metadata=False)

    elapsed = time.time() - started
    peak = mx.get_peak_memory()
    gc.collect()
    mx.clear_cache()
    return {
        "path": str(out_path),
        "width": image.image.width,
        "height": image.image.height,
        "elapsed": round(elapsed, 2),
        "peak_bytes": peak,
        "max_references": edit_mod.MAX_REFERENCES,
    }


# -------------------------------------------------------------------- main ---


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resident Qwen-Image-2.1 MLX worker.")
    parser.add_argument("--model", required=True, help="Local mflux/diffusers snapshot of Qwen-Image-2.1.")
    parser.add_argument("--base-model", default=None, help="Optional mflux base-model alias (qwen-image-2.1, qwen-image-2.1-turbo).")
    parser.add_argument("--quantize", default="auto", choices=["auto", "4", "8"], help="auto = use the checkpoint's own quantization metadata.")
    parser.add_argument("--low-ram", action="store_true", help="Enable VAE tiling and a 1 GB MLX cache limit.")
    parser.add_argument("--vae-tiling", action="store_true", help="Decode the VAE in tiles to cut peak memory.")
    parser.add_argument("--vae-tile-size", type=int, default=512)
    parser.add_argument("--cache-limit-gb", type=float, default=0.0)
    parser.add_argument(
        "--lora",
        action="append",
        nargs=2,
        metavar=("PATH", "SCALE"),
        default=None,
        help="Optional LoRA (local .safetensors) merged into the weights at load time. Repeatable.",
    )
    parser.add_argument(
        "--keep-text-encoder",
        action="store_true",
        help="Keep the 4.8 GB Qwen3-VL encoder resident (slower, higher peak; default is release after encoding).",
    )
    parser.add_argument("--no-bake-lora", action="store_true", help="Keep LoRA as runtime adapters instead of folding it into the weights.")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.low_ram and not args.vae_tiling:
        args.vae_tiling = True
        args.vae_tile_size = 512

    try:
        model = load_model(args)
    except Exception as exc:  # noqa: BLE001 - report anything to the node
        emit(
            type="fatal",
            message=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc()[-4000:],
        )
        return 1

    callback = ProgressCallback()
    log(f"model ready: {args.model}")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            emit(type="error", message=f"bad request frame: {exc}")
            continue

        op = request.get("op", "generate")
        request_id = request.get("id")
        if op == "shutdown":
            emit(type="bye", id=request_id)
            return 0
        if op != "generate":
            emit(type="error", id=request_id, message=f"unknown op {op!r}")
            continue

        try:
            result = generate(
                model,
                request.get("params") or {},
                callback,
                keep_text_encoder=bool(args.keep_text_encoder),
            )
            emit(type="result", id=request_id, **result)
        except KeyboardInterrupt:
            emit(type="error", id=request_id, message="interrupted")
        except Exception as exc:  # noqa: BLE001 - one bad run must not kill the worker
            emit(
                type="error",
                id=request_id,
                message=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc()[-6000:],
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
