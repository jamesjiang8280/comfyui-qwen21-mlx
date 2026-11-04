"""Qwen-Image-2.1 (edit + txt2img) on Apple Silicon via MLX -- ComfyUI nodes.

ComfyUI's native Qwen-Image nodes run the 20B-class pipeline on PyTorch/MPS.
On a Mac the same weights run through Apple's MLX runtime instead, which is what
mflux implements.  This pack keeps the ComfyUI graph shape and hands the heavy
lifting to a resident MLX worker:

    Qwen21 MLX Loader ─┬─▶ Qwen21 MLX Text to Image ─▶ IMAGE
                       └─▶ Qwen21 MLX Edit (1..10 refs, mask) ─▶ IMAGE + MASK

Why a resident worker: the 4-bit package is ~9.6 GB and loading it costs ~1 min.
Keeping one process alive means the second prompt starts sampling immediately,
and the same weights serve both the text-to-image and the edit branch.

What the edit path actually does (vs. a plain img2img checkpoint)
------------------------------------------------------------------
The package must contain the Qwen3-VL vision tower (``visual.*`` keys in the
text encoder).  With it, the model reads the reference image(s) as tokens and
follows instructions ("turn the jacket into black leather, keep the face"), can
take up to 10 references, honours a mask (white = repaint) and can auto-locate
an object from a phrase.  A text-only export without ``visual.*`` runs
text-to-image and plain img2img only -- the loader reports that capability and
the edit node refuses to pretend otherwise.

Defaults that matter
--------------------
* Qwen-Image-2.1 is guidance-distilled: ``guidance=1.0``.  Values >1 enable
  CFG and require a negative prompt (usually slower, rarely better here).
* 40 steps is the model's design point; ``strength < 1`` skips the high-noise
  steps and starts from the reference, which is the cheap way to do subtle
  retouches.
* 1024² with VAE tiling peaks around 15 GB of unified memory on this machine.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .runner import (
    Qwen21RunError,
    find_python,
    free_workers,
    run_generation,
)

try:  # optional: progress bar + interruption, only available inside ComfyUI
    import comfy.model_management as comfy_mm
    import comfy.utils as comfy_utils
except Exception:  # pragma: no cover
    comfy_mm = None
    comfy_utils = None

try:
    import folder_paths
except Exception:  # pragma: no cover
    folder_paths = None


VERSION = "1.2.0"
CATEGORY = "Qwen21 MLX"
QUANTS = ["auto", "4", "8"]
SCHEDULERS = ["linear", "viggle_turbo"]
MAX_REFERENCES = 10

_FIXES: list[str] = []


# ---------------------------------------------------------------- helpers ---


def _fix(name: str, msg: str) -> None:
    _FIXES.append(f"{name}: {msg}")


def _report_fixes() -> None:
    if _FIXES:
        print("[qwen21-mlx] 参数已自动规整：" + "; ".join(_FIXES), flush=True)
        _FIXES.clear()


def _as_int(value: Any, default: int, lo: int, hi: int, name: str) -> int:
    try:
        iv = int(float(value))
    except (TypeError, ValueError):
        _fix(name, f"invalid {value!r} -> {default}")
        return default
    if iv < lo or iv > hi:
        clamped = max(lo, min(hi, iv))
        _fix(name, f"{iv} 超出 [{lo},{hi}] -> {clamped}")
        iv = clamped
    return iv


def _as_str(value: Any, default: str, name: str, allow_numbers: bool = False) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return default
    if isinstance(value, bool):
        _fix(name, f"bool {value!r} -> default")
        return default
    if allow_numbers and isinstance(value, (int, float)):
        return str(value)
    _fix(name, f"str expected, got {value!r} -> default")
    return default


def _snap32(value: int, name: str) -> int:
    """Qwen's VAE is 16x and latents are packed 2x2 -> multiples of 32."""
    snapped = max(256, min(2048, int(round(value / 32.0)) * 32))
    if snapped != value:
        _fix(name, f"{value} 不是 32 的倍数 -> {snapped}")
    return snapped


def mlx_root() -> Path:
    env = os.environ.get("QWEN21_MLX_MODELS_DIR", "").strip()
    if env:
        return Path(os.path.expanduser(env))
    return Path.home() / "ComfyUI-Shared" / "models" / "MLX"


def _is_qwen21_snapshot(path: Path) -> bool:
    try:
        index = json.loads((path / "model_index.json").read_text(encoding="utf-8"))
        if "QwenImage21" in str(index.get("_class_name", "")):
            return True
    except Exception:
        pass
    try:
        cfg = json.loads((path / "transformer" / "config.json").read_text(encoding="utf-8"))
        return str(cfg.get("_class_name")) == "QwenImage21Transformer2DModel"
    except Exception:
        return False


def model_capability(path: Path) -> tuple[str, bool]:
    """``(capability, vision_chat)`` for a text encoder index.

    ``edit`` needs the Qwen3-VL visual tower (visual.* + language_model.*).
    ``vision_chat`` needs ``lm_head``, which powers auto_mask, prompt rewriting
    and verify.  Some 4-bit exports drop lm_head, so instruction editing still
    works but those three extras cannot run -- the edit node checks this before
    spending a full denoise.
    """
    index_path = path / "text_encoder" / "model.safetensors.index.json"
    try:
        weight_map = json.loads(index_path.read_text(encoding="utf-8")).get("weight_map", {})
    except Exception:
        return "txt2img", False
    keys = weight_map.keys()
    has_visual = any(k.startswith("visual.") for k in keys)
    has_language = any(k.startswith("language_model.") for k in keys)
    has_lm_head = any(k.startswith("lm_head.") for k in keys)
    return ("edit" if has_visual and has_language else "txt2img"), has_lm_head


def model_options() -> list[str]:
    """Every local Qwen-Image-2.1 MLX/mflux snapshot, newest first by name."""
    opts: list[str] = []
    root = mlx_root()
    if root.is_dir():
        for d in sorted(root.iterdir(), key=lambda x: x.name.lower()):
            if d.is_dir() and _is_qwen21_snapshot(d):
                opts.append(str(d))
    if not opts:
        fallback = root / "JoyFusionAI__Qwen-Image-2.1-Edit-MLX-4bit"
        opts.append(str(fallback))
    return opts


def default_model_value() -> str:
    opts = model_options()
    if not opts:
        return ""
    def score(path: str) -> int:
        name = Path(path).name.lower()
        return (
            (4 if "2.1" in name else 0)
            + (2 if "edit" in name else 0)
            + (2 if ("4bit" in name or "-q4" in name) else 0)
            + (1 if model_capability(Path(path))[0] == "edit" else 0)
        )
    return max(opts, key=score)


def _resolve_model(cfg: dict[str, Any]) -> Path:
    raw = (cfg.get("model_path_override") or "").strip() or cfg["model"]
    path = Path(os.path.expanduser(str(raw)))
    if not path.is_dir():
        raise Qwen21RunError(
            f"模型目录不存在：{path}\n"
            "把 mflux/diffusers 格式的 Qwen-Image-2.1 快照放到 "
            f"{mlx_root()} 下，或在上方 combo 里选择已有模型。"
        )
    if not _is_qwen21_snapshot(path):
        raise Qwen21RunError(
            f"{path} 不是 Qwen-Image-2.1 快照（缺 model_index.json / transformer/config.json）。"
        )
    return path


def _tensor_to_pil(tensor: torch.Tensor, index: int = 0) -> Image.Image:
    arr = tensor[index].detach().cpu().numpy()
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    arr = np.clip(arr, 0.0, 1.0)
    channels = arr.shape[-1]
    if channels == 4:
        return Image.fromarray((arr * 255.0).round().astype(np.uint8), "RGBA")
    if channels >= 3:
        return Image.fromarray((arr[..., :3] * 255.0).round().astype(np.uint8), "RGB")
    raise Qwen21RunError(f"无法识别的 IMAGE 通道数：{channels}")


def _save_reference(tensor: torch.Tensor, path: Path) -> None:
    image = _tensor_to_pil(tensor)
    if image.mode == "RGB":
        image = image.convert("RGBA")
    image.save(path)


def _save_mask(mask: torch.Tensor, path: Path) -> None:
    arr = mask.detach().cpu().numpy()
    if arr.ndim == 3:
        arr = arr[0]
    arr = np.clip(arr, 0.0, 1.0)
    Image.fromarray((arr * 255.0).round().astype(np.uint8), "L").save(path)


def _pil_to_image(image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
    """RGBA PNG -> (RGB over white, alpha as MASK).  Keeps transparent output usable."""
    rgba = image.convert("RGBA")
    alpha = np.asarray(rgba.getchannel("A")).astype(np.float32) / 255.0
    canvas = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    canvas.alpha_composite(rgba)
    rgb = np.asarray(canvas.convert("RGB")).astype(np.float32) / 255.0
    return (
        torch.from_numpy(rgb)[None, ...],
        torch.from_numpy(alpha)[None, ...],
    )


def _temp_dir() -> Path:
    if folder_paths is not None:
        try:
            path = Path(folder_paths.get_temp_directory())
            path.mkdir(parents=True, exist_ok=True)
            return path
        except Exception:
            pass
    return Path(tempfile.mkdtemp(prefix="qwen21_mlx_"))


def _cancel_check():
    if comfy_mm is None:
        return None

    def should_cancel() -> bool:
        try:
            return bool(comfy_mm.processing_interrupted())
        except Exception:
            return False

    return should_cancel


LORA_EXTS = (".safetensors", ".sft")


def lora_options() -> list[str]:
    """LoRA files ComfyUI knows about (including its extra_model_paths)."""
    names: list[str] = []
    if folder_paths is not None:
        try:
            names = [n for n in folder_paths.get_filename_list("loras") if n.lower().endswith(LORA_EXTS)]
        except Exception:
            names = []
    if not names:
        root = Path.home() / "ComfyUI-Shared" / "models" / "loras"
        if root.is_dir():
            names = [str(p.relative_to(root)) for p in root.rglob("*") if p.suffix.lower() in LORA_EXTS]
    return sorted(names, key=str.lower) or [""]


def resolve_lora(name: str) -> Path:
    name = (name or "").strip()
    if not name:
        raise Qwen21RunError("没有选择 LoRA 文件。")
    if folder_paths is not None:
        try:
            full = folder_paths.get_full_path("loras", name)
            if full and Path(full).is_file():
                return Path(full)
        except Exception:
            pass
    candidate = Path(os.path.expanduser(name))
    if not candidate.is_absolute():
        candidate = Path.home() / "ComfyUI-Shared" / "models" / "loras" / name
    if not candidate.is_file():
        raise Qwen21RunError(f"LoRA 文件不存在：{candidate}")
    return candidate


# ------------------------------------------------------------------ nodes ---


class Qwen21MlxLoader:
    """Resolve the MLX model + interpreter, emit a config object."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "model": (
                    model_options(),
                    {
                        "default": default_model_value(),
                        "tooltip": "扫描 ~/ComfyUI-Shared/models/MLX 下的 Qwen-Image-2.1 mflux 快照",
                    },
                ),
                "quantization": (
                    QUANTS,
                    {
                        "default": "auto",
                        "tooltip": "auto=用权重自带的量化元数据（Q4 包不要选 4，否则会再量化一次）",
                    },
                ),
                "keep_resident": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "生成后保留 9.6GB 模型常驻，下一张免加载；显存吃紧时关闭"},
                ),
                "low_ram": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "低内存模式：VAE 分块 + 小缓存（16GB 机器建议开）"},
                ),
                "vae_tiling": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "VAE 分块解码；1024² 及以上更稳，速度略降"},
                ),
                "cache_limit_gb": (
                    "INT",
                    {"default": 2, "min": 0, "max": 64, "step": 1, "tooltip": "0=不限；8~12 可压低长时间运行的内存峰值"},
                ),
            },
            "optional": {
                "model_path_override": (
                    "STRING",
                    {"default": "", "tooltip": "手动指定快照绝对路径（优先于上面的下拉）"},
                ),
                "verbose": ("BOOLEAN", {"default": False, "tooltip": "把 worker 的全部日志打到 ComfyUI 控制台"}),
            },
        }

    RETURN_TYPES = ("QWEN21_MLX_CONFIG",)
    RETURN_NAMES = ("qwen21_mlx_model",)
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(
        self,
        model: str,
        quantization: str,
        keep_resident: bool = True,
        low_ram: bool = False,
        vae_tiling: bool = False,
        cache_limit_gb: int = 2,
        model_path_override: str = "",
        verbose: bool = False,
    ) -> tuple[dict[str, Any]]:
        model = _as_str(model, "", "model").strip()
        override = _as_str(model_path_override, "", "model_path_override").strip()
        if not model and not override:
            raise Qwen21RunError("没有可选模型：请把 Qwen-Image-2.1 的 MLX 快照放进 " + str(mlx_root()))
        if quantization not in QUANTS:
            _fix("quantization", f"{quantization!r} -> auto")
            quantization = "auto"
        cache_limit_gb = _as_int(cache_limit_gb, 2, 0, 64, "cache_limit_gb")
        _report_fixes()

        cfg = {
            "model": model,
            "model_path_override": override,
            "quantization": quantization,
            "keep_resident": bool(keep_resident),
            "low_ram": bool(low_ram),
            "vae_tiling": bool(vae_tiling or low_ram),
            "cache_limit_gb": cache_limit_gb,
            "verbose": bool(verbose),
            "python": find_python(),
        }

        path = _resolve_model(cfg)
        capability, vision_chat = model_capability(path)
        cfg.update(model=str(path), capability=capability, vision_chat=vision_chat, label=path.name)

        print(
            f"[qwen21-mlx] v{VERSION} | model={path.name} | capability={capability} | "
            f"vision_chat={'yes' if vision_chat else 'no'} | "
            f"quant={quantization} | resident={bool(keep_resident)} | python={cfg['python']}",
            flush=True,
        )
        if not vision_chat:
            print(
                "[qwen21-mlx] 该权重缺 text_encoder 的 lm_head：指令编辑可用，"
                "但 auto_mask / enhance_prompt / verify 不可用（节点会提前拦截）。",
                flush=True,
            )
        if capability != "edit":
            print(
                "[qwen21-mlx] 该包缺少 Qwen3-VL 视觉塔（visual.*），只能文生图/普通重绘；"
                "图像编辑节点会拒绝运行。",
                flush=True,
            )
        return (cfg,)


class Qwen21MlxTextToImage:
    """Qwen-Image-2.1 text-to-image through mflux (MLX)."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "qwen21_mlx_model": ("QWEN21_MLX_CONFIG",),
                "prompt": (
                    "STRING",
                    {"default": "", "multiline": True, "tooltip": "Qwen 中英文都强；建议写清主体/构图/光线/风格"},
                ),
                "negative_prompt": (
                    "STRING",
                    {"default": "", "multiline": True, "tooltip": "仅 guidance>1 时生效；默认 1.0 不用填"},
                ),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 32}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 32}),
                "steps": (
                    "INT",
                    {"default": 40, "min": 1, "max": 100, "step": 1, "tooltip": "官方设计点 40 步；turbo 蒸馏包用 8 步"},
                ),
                "guidance": (
                    "FLOAT",
                    {"default": 1.0, "min": 1.0, "max": 20.0, "step": 0.1, "tooltip": "Qwen-Image-2.1 是 guidance 蒸馏模型，保持 1.0"},
                ),
                "seed": (
                    "INT",
                    {"default": 42, "min": 0, "max": 2**31 - 1, "step": 1, "control_after_generate": True},
                ),
                "scheduler": (SCHEDULERS, {"default": "linear", "tooltip": "linear=Euler；viggle_turbo 需配 6 步 + 蒸馏 LoRA"}),
                "use_kv_cache": ("BOOLEAN", {"default": True, "tooltip": "复用文本/参考图前缀注意力，默认开"}),
                "use_step_cache": ("BOOLEAN", {"default": False, "tooltip": "跳步缓存，更快但输出略有差异；10 步以下不生效"}),
                "step_cache_threshold": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    def generate(
        self,
        qwen21_mlx_model: dict[str, Any],
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        guidance: float,
        seed: int,
        scheduler: str,
        use_kv_cache: bool,
        use_step_cache: bool,
        step_cache_threshold: float,
    ):
        prompt = _as_str(prompt, "", "prompt", allow_numbers=True)
        negative_prompt = _as_str(negative_prompt, "", "negative_prompt")
        if not prompt.strip():
            raise Qwen21RunError("prompt 为空：请写下要生成的内容。")
        width = _snap32(_as_int(width, 1024, 256, 2048, "width"), "width")
        height = _snap32(_as_int(height, 1024, 256, 2048, "height"), "height")
        steps = _as_int(steps, 40, 1, 100, "steps")
        seed = _as_int(seed, 42, 0, 2**31 - 1, "seed")
        if scheduler not in SCHEDULERS:
            _fix("scheduler", f"{scheduler!r} -> linear")
            scheduler = "linear"
        if scheduler == "viggle_turbo" and steps != 6:
            _fix("steps", f"viggle_turbo 需要 6 步，{steps} -> 6")
            steps = 6
        if guidance > 1.0 and not negative_prompt.strip():
            # mflux rejects guidance>1 without an explicit negative prompt.
            negative_prompt = ""
            print("[qwen21-mlx] guidance>1：负向提示词为空字符串（等价于不做负向引导）", flush=True)
        _report_fixes()

        params = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "guidance": float(guidance),
            "seed": seed,
            "scheduler": scheduler,
            "use_kv_cache": bool(use_kv_cache),
            "use_step_cache": bool(use_step_cache),
            "step_cache_threshold": float(step_cache_threshold),
            "image_paths": [],
            "out_path": "",
        }
        return self._run(qwen21_mlx_model, params, steps)

    @staticmethod
    def _run(cfg: dict[str, Any], params: dict[str, Any], steps: int):
        bar = None
        if comfy_utils is not None:
            try:
                bar = comfy_utils.ProgressBar(steps)
            except Exception:
                bar = None

        tmp = _temp_dir()
        out_path = tmp / f"qwen21_t2i_{params['seed']}_{params['width']}x{params['height']}.png"
        params["out_path"] = str(out_path)

        def on_progress(step: int, total: int) -> None:
            if bar is not None:
                try:
                    bar.update_absolute(min(step, total), total)
                except Exception:
                    pass

        try:
            run_generation(
                cfg,
                params,
                on_progress=on_progress,
                should_cancel=_cancel_check(),
            )
            image = Image.open(out_path)
            image.load()
            return (_pil_to_image(image)[0],)
        finally:
            try:
                out_path.unlink(missing_ok=True)
            except OSError:
                pass


class Qwen21MlxEdit:
    """Qwen-Image-2.1 instruction editing with 1..10 references and optional mask."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        optional_inputs: dict[str, Any] = {
            f"image_{i}": ("IMAGE",) for i in range(2, MAX_REFERENCES + 1)
        }
        optional_inputs.update(
            {
                "mask": ("MASK", {"tooltip": "白色=重绘，黑色=保留；作用于第一张参考图"}),
                "auto_mask": (
                    "STRING",
                    {"default": "", "tooltip": "让 Qwen3-VL 自动定位要改的物体，如 'the red shirt'（与 mask 二选一）"},
                ),
                "use_kv_cache": ("BOOLEAN", {"default": True}),
                "use_step_cache": ("BOOLEAN", {"default": False}),
                "step_cache_threshold": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01}),
                "enhance_prompt": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "先用内置 Qwen3-VL 把指令扩写成详细提示词（多一次前向，较慢）"},
                ),
                "verify": ("BOOLEAN", {"default": False, "tooltip": "生成后让 Qwen3-VL 自检是否完成编辑"}),
                "verify_retries": ("INT", {"default": 1, "min": 0, "max": 5, "step": 1}),
                "output_resolution": (
                    "INT",
                    {"default": 1024, "min": 256, "max": 2048, "step": 32, "tooltip": "参考图缩放/自动输出尺寸的像素预算"},
                ),
            }
        )
        return {
            "required": {
                "qwen21_mlx_model": ("QWEN21_MLX_CONFIG",),
                "prompt": (
                    "STRING",
                    {"default": "把外套改成黑色皮衣，保持人物身份、姿势和背景不变", "multiline": True},
                ),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "image_1": ("IMAGE",),
                "width": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 32, "tooltip": "0=按参考图长宽比自动"}),
                "height": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 32, "tooltip": "0=按参考图长宽比自动"}),
                "steps": ("INT", {"default": 40, "min": 1, "max": 100, "step": 1}),
                "guidance": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.05, "max": 1.0, "step": 0.05, "tooltip": "1.0=完全按指令重画；0.6~0.8=保留更多原图，且更省步数"},
                ),
                "scheduler": (SCHEDULERS, {"default": "linear"}),
                "seed": (
                    "INT",
                    {"default": 42, "min": 0, "max": 2**31 - 1, "step": 1, "control_after_generate": True},
                ),
            },
            "optional": optional_inputs,
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "alpha")
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    def generate(
        self,
        qwen21_mlx_model: dict[str, Any],
        prompt: str,
        negative_prompt: str,
        image_1,
        width: int,
        height: int,
        steps: int,
        guidance: float,
        strength: float,
        scheduler: str,
        seed: int,
        image_2=None,
        image_3=None,
        image_4=None,
        image_5=None,
        image_6=None,
        image_7=None,
        image_8=None,
        image_9=None,
        image_10=None,
        mask=None,
        auto_mask: str = "",
        use_kv_cache: bool = True,
        use_step_cache: bool = False,
        step_cache_threshold: float = 0.12,
        enhance_prompt: bool = False,
        verify: bool = False,
        verify_retries: int = 1,
        output_resolution: int = 1024,
    ):
        cfg = qwen21_mlx_model or {}
        if cfg.get("capability") != "edit":
            raise Qwen21RunError(
                "当前模型包没有 Qwen3-VL 视觉塔，不支持指令式图像编辑。\n"
                "请改用完整 Edit 包（text_encoder 内含 visual.* 权重），例如 "
                "JoyFusionAI/Qwen-Image-2.1-Edit-MLX-4bit。"
            )

        prompt = _as_str(prompt, "", "prompt", allow_numbers=True)
        negative_prompt = _as_str(negative_prompt, "", "negative_prompt")
        auto_mask = _as_str(auto_mask, "", "auto_mask").strip()
        if not prompt.strip():
            raise Qwen21RunError("prompt 为空：请写下要做的编辑。")

        chat_used = []
        if auto_mask and mask is None:
            chat_used.append("auto_mask")
        if enhance_prompt:
            chat_used.append("enhance_prompt")
        if verify:
            chat_used.append("verify")
        if chat_used and not cfg.get("vision_chat"):
            raise Qwen21RunError(
                "当前权重缺少 text_encoder 的 lm_head，无法运行 " + " / ".join(chat_used) + "。\n"
                "这是 4bit 导出时裁掉了 Qwen3-VL 的文本生成头：指令编辑本身正常，"
                "但 auto_mask / enhance_prompt / verify 要靠它生成文本。\n"
                "处理：关闭这些选项；局部修改改用手动 mask（白=重绘，黑=保留）。"
            )

        strength = float(strength)
        if not 0.0 < strength <= 1.0:
            _fix("strength", f"{strength} -> 1.0")
            strength = 1.0
        steps = _as_int(steps, 40, 1, 100, "steps")
        seed = _as_int(seed, 42, 0, 2**31 - 1, "seed")
        output_resolution = _snap32(_as_int(output_resolution, 1024, 256, 2048, "output_resolution"), "output_resolution")
        if scheduler not in SCHEDULERS:
            _fix("scheduler", f"{scheduler!r} -> linear")
            scheduler = "linear"
        if scheduler == "viggle_turbo" and steps != 6:
            _fix("steps", f"viggle_turbo 需要 6 步，{steps} -> 6")
            steps = 6
        _report_fixes()

        refs = [t for t in (image_1, image_2, image_3, image_4, image_5, image_6, image_7, image_8, image_9, image_10) if t is not None]
        if not refs:
            raise Qwen21RunError("至少需要一张参考图（image_1）。")
        if len(refs) > MAX_REFERENCES:
            raise Qwen21RunError(f"最多支持 {MAX_REFERENCES} 张参考图，收到 {len(refs)} 张。")

        width_i = _as_int(width, 0, 0, 2048, "width")
        height_i = _as_int(height, 0, 0, 2048, "height")
        if width_i <= 0 or height_i <= 0:
            width_i = height_i = 0  # mflux derives both from the last reference's ratio
        else:
            width_i = _snap32(width_i, "width")
            height_i = _snap32(height_i, "height")

        tmp = _temp_dir()
        ref_paths: list[Path] = []
        mask_path: Path | None = None
        out_path = tmp / f"qwen21_edit_{seed}.png"
        try:
            for index, tensor in enumerate(refs):
                ref_path = tmp / f"qwen21_ref_{seed}_{index}.png"
                _save_reference(tensor, ref_path)
                ref_paths.append(ref_path)
            if mask is not None:
                mask_path = tmp / f"qwen21_mask_{seed}.png"
                _save_mask(mask, mask_path)

            if mask is not None and auto_mask:
                print("[qwen21-mlx] mask 与 auto_mask 同时存在：mflux 会忽略 auto_mask", flush=True)

            params = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": width_i,
                "height": height_i,
                "steps": steps,
                "guidance": float(guidance),
                "strength": strength,
                "scheduler": scheduler,
                "seed": seed,
                "image_paths": [str(p) for p in ref_paths],
                "mask_image": str(mask_path) if mask_path else None,
                "auto_mask": auto_mask or None,
                "use_kv_cache": bool(use_kv_cache),
                "use_step_cache": bool(use_step_cache),
                "step_cache_threshold": float(step_cache_threshold),
                "enhance_prompt": bool(enhance_prompt),
                "verify": bool(verify),
                "verify_retries": int(verify_retries),
                "output_resolution": output_resolution,
                "out_path": str(out_path),
            }

            bar = None
            if comfy_utils is not None:
                try:
                    bar = comfy_utils.ProgressBar(steps)
                except Exception:
                    bar = None

            def on_progress(step: int, total: int) -> None:
                if bar is not None:
                    try:
                        bar.update_absolute(min(step, total), total)
                    except Exception:
                        pass

            print(
                f"[qwen21-mlx] 编辑：{len(ref_paths)} 张参考图, {width_i or 'auto'}x{height_i or 'auto'}, "
                f"{steps} 步, strength={strength}, scheduler={scheduler}",
                flush=True,
            )
            run_generation(cfg, params, on_progress=on_progress, should_cancel=_cancel_check())

            image = Image.open(out_path)
            image.load()
            return _pil_to_image(image)
        finally:
            for path in ref_paths + ([mask_path] if mask_path else []) + [out_path]:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass


class Qwen21MlxLoraLoader:
    """Attach a Qwen-Image-2.1 LoRA to a config (baked or as runtime adapters)."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "qwen21_mlx_model": ("QWEN21_MLX_CONFIG",),
                "lora_name": (
                    lora_options(),
                    {
                        "default": next((n for n in lora_options() if "2.1" in n and ("turbo" in n or "light" in n)), ""),
                        "tooltip": "~/ComfyUI-Shared/models/loras 下的 .safetensors；8 步 turbo LoRA 能把 40 步降到 8 步",
                    },
                ),
                "strength": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05}),
                "enabled": ("BOOLEAN", {"default": True, "tooltip": "关闭=直通，便于 A/B 对比"}),
            },
            "optional": {
                "bake": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "烘焙进权重：省一点运行开销，但 mflux 会把被折叠加固的 4bit 层提到 8bit，显存明显变高"},
                ),
            },
        }

    RETURN_TYPES = ("QWEN21_MLX_CONFIG",)
    RETURN_NAMES = ("qwen21_mlx_model",)
    FUNCTION = "apply"
    CATEGORY = CATEGORY

    def apply(
        self,
        qwen21_mlx_model: dict[str, Any],
        lora_name: str,
        strength: float = 1.0,
        enabled: bool = True,
        bake: bool = False,
    ) -> tuple[dict[str, Any]]:
        cfg = dict(qwen21_mlx_model or {})
        if not enabled or not (lora_name or "").strip():
            for key in ("lora_paths", "lora_scales", "lora_bake"):
                cfg.pop(key, None)
            print("[qwen21-mlx] LoRA 已禁用，直通基础模型", flush=True)
            return (cfg,)

        path = resolve_lora(lora_name)
        if not path.is_file():
            raise Qwen21RunError(f"LoRA 文件不存在：{path}")
        scale = float(strength)
        cfg.update(lora_paths=[str(path)], lora_scales=[scale], lora_bake=bool(bake))
        print(
            f"[qwen21-mlx] LoRA: {path.name} (scale={scale}, "
            f"{'烘焙' if bake else '运行时适配器'}) -> worker 将重新加载模型",
            flush=True,
        )
        return (cfg,)


class Qwen21MlxLoraStack:
    """Stack up to four LoRAs in one node; chain nodes for more."""

    SLOTS = 4

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        choices = ["none"] + lora_options()
        required: dict[str, Any] = {"qwen21_mlx_model": ("QWEN21_MLX_CONFIG",)}
        for i in range(1, cls.SLOTS + 1):
            required[f"lora_{i}"] = (
                choices,
                {"default": "none", "tooltip": f"第 {i} 个 LoRA；none = 空槽"},
            )
            required[f"strength_{i}"] = (
                "FLOAT",
                {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05},
            )
        required["bake"] = (
            "BOOLEAN",
            {"default": False, "tooltip": "烘焙进权重：略快但 mflux 会把部分 4bit 层提到 8bit，显存更高"},
        )
        required["enabled"] = ("BOOLEAN", {"default": True, "tooltip": "关闭=整组直通，便于 A/B"})
        return {"required": required}

    RETURN_TYPES = ("QWEN21_MLX_CONFIG",)
    RETURN_NAMES = ("qwen21_mlx_model",)
    FUNCTION = "apply"
    CATEGORY = CATEGORY

    def apply(
        self,
        qwen21_mlx_model: dict[str, Any],
        lora_1: str = "none",
        strength_1: float = 1.0,
        lora_2: str = "none",
        strength_2: float = 1.0,
        lora_3: str = "none",
        strength_3: float = 1.0,
        lora_4: str = "none",
        strength_4: float = 1.0,
        bake: bool = False,
        enabled: bool = True,
    ) -> tuple[dict[str, Any]]:
        cfg = dict(qwen21_mlx_model or {})
        if not enabled:
            print("[qwen21-mlx] LoRA Stack 已禁用，直通上游配置", flush=True)
            return (cfg,)

        paths = [str(p) for p in (cfg.get("lora_paths") or [])]
        scales = [float(s) for s in (cfg.get("lora_scales") or [])]
        added: list[str] = []
        for name, strength in (
            (lora_1, strength_1),
            (lora_2, strength_2),
            (lora_3, strength_3),
            (lora_4, strength_4),
        ):
            name = (name or "").strip()
            if not name or name == "none":
                continue
            path = resolve_lora(name)
            if not path.is_file():
                raise Qwen21RunError(f"LoRA 文件不存在：{path}")
            paths.append(str(path))
            scales.append(float(strength))
            added.append(f"{path.name}@{float(strength):g}")

        cfg.update(lora_paths=paths, lora_scales=scales, lora_bake=bool(bake))
        summary = ", ".join(f"{Path(p).name}@{s:g}" for p, s in zip(paths, scales)) or "无"
        print(
            f"[qwen21-mlx] LoRA Stack: {len(paths)} 个 ({summary}), "
            f"{'烘焙' if bake else '运行时适配器'} -> worker 将重新加载模型",
            flush=True,
        )
        return (cfg,)


class Qwen21MlxFreeMemory:
    """Stop the resident MLX worker(s) and release unified memory."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "unload_all": ("BOOLEAN", {"default": True, "tooltip": "关闭=只卸载 match 里包含该字符串的模型"}),
                "match": ("STRING", {"default": "", "tooltip": "模型路径关键字，如 JoyFusion"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "free"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def free(self, unload_all: bool = True, match: str = "") -> tuple[str]:
        match = (match or "").strip()
        stopped = free_workers(None if unload_all or not match else match)
        status = f"已停止 {stopped} 个 MLX worker"
        print(f"[qwen21-mlx] {status}", flush=True)
        return (status,)


NODE_CLASS_MAPPINGS = {
    "Qwen21MlxLoader": Qwen21MlxLoader,
    "Qwen21MlxTextToImage": Qwen21MlxTextToImage,
    "Qwen21MlxEdit": Qwen21MlxEdit,
    "Qwen21MlxLoraLoader": Qwen21MlxLoraLoader,
    "Qwen21MlxLoraStack": Qwen21MlxLoraStack,
    "Qwen21MlxFreeMemory": Qwen21MlxFreeMemory,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen21MlxLoader": "Qwen-Image-2.1 MLX 模型加载器",
    "Qwen21MlxTextToImage": "Qwen-Image-2.1 MLX 文生图",
    "Qwen21MlxEdit": "Qwen-Image-2.1 MLX 图像编辑",
    "Qwen21MlxLoraLoader": "Qwen-Image-2.1 MLX LoRA 加载器",
    "Qwen21MlxLoraStack": "Qwen-Image-2.1 MLX LoRA Stack",
    "Qwen21MlxFreeMemory": "Qwen-Image-2.1 MLX 释放内存",
}

print(f"[qwen21-mlx] nodes v{VERSION} loaded", flush=True)
