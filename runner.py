"""ComfyUI-side bridge to the resident Qwen-Image-2.1 MLX worker.

The node pack never imports ``mflux``/``mlx`` into ComfyUI's process: mflux pins
its own ``mlx``/``transformers``/``huggingface_hub`` and ComfyUI pins different
ones.  Instead we start ``worker.py`` with an interpreter that has mflux, speak
newline-delimited JSON to it, and keep it alive between prompts so the ~9.6 GB
4-bit weights are loaded once.

Layout::

    ComfyUI node ──JSONL──▶ worker.py (mflux/MLX) ──PNG──▶ ComfyUI IMAGE
                        │
                        └─ logs/worker-*.log  (mflux stdout/stderr)

If ComfyUI interrupts a run, the worker is killed; the next run reloads it.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

NODE_DIR = Path(__file__).resolve().parent
LOG_DIR = NODE_DIR / "logs"

WORKER_MODULE_PROBE = "import mflux.models.qwen21.variants.edit.qwen_image_21_edit"

_PYTHON_CACHE: str | None = None
_WORKERS: dict[str, "Worker"] = {}
_LOCKS: dict[str, threading.Lock] = {}
_REGISTRY_LOCK = threading.Lock()


logger = logging.getLogger("qwen21-mlx")


def _say(message: str, **_) -> None:
    """Log through logging so the line reaches the terminal, comfyui.log and the UI console."""
    logger.info(message)


BAR_WIDTH = 24


def render_progress(step: int, total: int, rate: float | None = None) -> str:
    """One-line console progress bar; ``rate`` is seconds per step (None = warming up)."""
    total = max(1, total)
    step = max(0, min(step, total))
    frac = step / total
    filled = int(round(frac * BAR_WIDTH))
    bar = "█" * filled + "░" * (BAR_WIDTH - filled)
    if rate:
        eta = max(0.0, rate * (total - step))
        timing = f"{rate:4.1f}s/步  剩余 {eta:3.0f}s"
    else:
        timing = "预热中（首步含权重加载）"
    return f"{bar} {step:>3}/{total:<3} {frac * 100:5.1f}%  {timing}"


class Qwen21RunError(RuntimeError):
    """Raised when the MLX worker fails (missing weights, OOM, bad args...)."""


# ------------------------------------------------------------ interpreter ---


def _probe(python: str) -> bool:
    try:
        done = subprocess.run(
            [python, "-c", WORKER_MODULE_PROBE],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=180,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0


def find_python() -> str:
    """Locate an interpreter that can run mflux's Qwen-Image-2.1 model.

    Order: explicit override -> this pack's own ``.venv`` -> the user-level
    ``mflux-venv`` this machine already uses -> ComfyUI's interpreter.
    """
    global _PYTHON_CACHE
    if _PYTHON_CACHE:
        return _PYTHON_CACHE

    candidates: list[str] = []
    env = os.environ.get("QWEN21_MLX_PYTHON", "").strip()
    if env:
        candidates.append(os.path.expanduser(env))
    candidates += [
        str(NODE_DIR / ".venv" / "bin" / "python"),
        str(Path.home() / "mflux-venv" / "bin" / "python"),
        str(Path.home() / "ComfyUI-Shared" / "mflux-venv" / "bin" / "python"),
        sys.executable,
    ]

    seen: set[str] = set()
    for python in candidates:
        if python in seen:
            continue
        seen.add(python)
        if not Path(python).is_file():
            continue
        if _probe(python):
            _PYTHON_CACHE = python
            _say(f"[qwen21-mlx] mflux 解释器: {python}", flush=True)
            return python

    raise Qwen21RunError(
        "找不到能加载 Qwen-Image-2.1 的 mflux 环境。\n"
        "请安装 mflux（pip install mflux）或设置环境变量指向对应 python，例如：\n"
        "  export QWEN21_MLX_PYTHON=$HOME/mflux-venv/bin/python"
    )


# ---------------------------------------------------------------- worker ----


class Worker:
    """One resident ``worker.py`` process."""

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.model = cfg["model"]
        self.proc: subprocess.Popen | None = None
        self.log_path = self._log_path(cfg)
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._log_file = None
        self.peak_bytes = 0
        self.load_seconds: float | None = None
        self._start()

    @staticmethod
    def _log_candidates(cfg: dict[str, Any]) -> list[Path]:
        name = Path(cfg["model"]).name.replace(" ", "_")[:60]
        bases = []
        env = os.environ.get("QWEN21_MLX_LOG_DIR", "").strip()
        if env:
            bases.append(Path(os.path.expanduser(env)))
        bases += [LOG_DIR, Path(tempfile.gettempdir()) / "qwen21-mlx-logs"]
        return [base / f"worker-{name}.log" for base in bases]

    def _log_path(self, cfg: dict[str, Any]) -> Path:
        """First writable location wins; a read-only install still gets diagnostics."""
        for path in self._log_candidates(cfg):
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.is_file() and path.stat().st_size > 8 << 20:
                    path.unlink()
                with open(path, "ab"):
                    pass
                return path
            except OSError:
                continue
        return Path(os.devnull)

    def _args(self) -> list[str]:
        cfg = self.cfg
        args = [
            find_python(),
            str(NODE_DIR / "worker.py"),
            "--model",
            cfg["model"],
            "--quantize",
            str(cfg.get("quantize", "auto")),
            "--cache-limit-gb",
            str(float(cfg.get("cache_limit_gb", 2) or 0)),
        ]
        if cfg.get("base_model"):
            args += ["--base-model", str(cfg["base_model"])]
        if cfg.get("vae_tiling"):
            args.append("--vae-tiling")
        if cfg.get("low_ram"):
            args.append("--low-ram")
        lora_paths = list(cfg.get("lora_paths") or [])
        lora_scales = list(cfg.get("lora_scales") or [])
        for path, scale in zip(lora_paths, lora_scales):
            args += ["--lora", str(path), str(scale)]
        if lora_paths and not cfg.get("lora_bake", False):
            args.append("--no-bake-lora")
        if cfg.get("verbose"):
            args.append("--verbose")
        return args

    def _start(self) -> None:
        args = self._args()
        _say(f"[qwen21-mlx] 启动 MLX worker: {Path(self.model).name}", flush=True)
        self._log_file = open(self.log_path, "ab", buffering=0)
        self.proc = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log_file,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
        )
        threading.Thread(target=self._reader, name="qwen21-mlx-reader", daemon=True).start()
        self._wait_ready()

    def _reader(self) -> None:
        assert self.proc and self.proc.stdout is not None
        for line in self.proc.stdout:
            self._queue.put(line)
        self._queue.put(None)

    def _next_frame(self, deadline: float, should_cancel: Callable[[], bool] | None) -> dict[str, Any]:
        while True:
            timeout = max(0.0, min(0.25, deadline - time.time()))
            try:
                line = self._queue.get(timeout=timeout)
            except queue.Empty:
                line = ""

            if line is None:
                raise Qwen21RunError(self._death_message())
            if line:
                try:
                    frame = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(frame, dict):
                    return frame

            if should_cancel and should_cancel():
                self.kill()
                raise Qwen21RunError("已取消（ComfyUI 中断），MLX worker 已停止。")
            if time.time() > deadline:
                self.kill()
                raise Qwen21RunError(
                    "等待 MLX worker 超时，已停止。最近日志：\n" + self._log_tail()
                )

    def _wait_ready(self) -> None:
        deadline = time.time() + 900
        while True:
            frame = self._next_frame(deadline, None)
            kind = frame.get("type")
            if kind == "ready":
                self.load_seconds = frame.get("load_seconds")
                self.peak_bytes = int(frame.get("peak_bytes") or 0)
                _say(
                    f"[qwen21-mlx] 模型就绪：{Path(self.model).name} "
                    f"(quant={frame.get('bits') or 'auto'}, "
                    f"载入 {frame.get('load_seconds')}s, "
                    f"MLX 峰值 {self.peak_bytes / 1e9:.1f} GB)",
                    flush=True,
                )
                return
            if kind == "fatal":
                raise Qwen21RunError(
                    f"模型加载失败：{frame.get('message')}\n{frame.get('traceback', '')}"
                )
            if kind == "status" and frame.get("phase") == "loading":
                _say("[qwen21-mlx] 正在加载 Qwen-Image-2.1 权重（首次约 1-2 分钟）...", flush=True)

    def _death_message(self) -> str:
        code = self.proc.poll() if self.proc else None
        return f"MLX worker 已退出（exit={code}）。最近日志：\n{self._log_tail()}"

    def _log_tail(self, lines: int = 25) -> str:
        try:
            with open(self.log_path, "rb") as fh:
                fh.seek(0, os.SEEK_END)
                size = fh.tell()
                fh.seek(max(0, size - 12000))
                text = fh.read().decode("utf-8", errors="replace")
        except OSError:
            return "(日志不可用)"
        rows = [r for r in text.replace("\r", "\n").splitlines() if r.strip()]
        return "\n".join(rows[-lines:]) or "(日志为空)"

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def generate(
        self,
        params: dict[str, Any],
        on_progress: Callable[[int, int], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
        verbose: bool = False,
    ) -> dict[str, Any]:
        if not self.alive():
            raise Qwen21RunError(self._death_message())

        request_id = uuid.uuid4().hex
        payload = json.dumps({"id": request_id, "op": "generate", "params": params}, ensure_ascii=False)
        assert self.proc and self.proc.stdin is not None
        try:
            self.proc.stdin.write(payload + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise Qwen21RunError(f"MLX worker 写入失败：{exc}\n{self._log_tail()}") from exc

        deadline = time.time() + 60 * 60  # a 40-step 2048² run can take a while
        started = time.time()
        first_step_at: float | None = None
        last_total = 0
        announced: set[str] = set()
        while True:
            frame = self._next_frame(deadline, should_cancel)
            kind = frame.get("type")

            if kind == "progress":
                step = int(frame.get("step", 0))
                total = int(frame.get("total", 0))
                now = time.time()
                rate = (now - first_step_at) / (step - 1) if first_step_at and step > 1 else None
                if first_step_at is None:
                    first_step_at = now
                last_total = max(last_total, total)
                _say(f"[qwen21-mlx] 采样 {render_progress(step, total, rate)}", flush=True)
                if on_progress:
                    on_progress(step, total)
                continue

            if kind == "status":
                phase = frame.get("phase")
                label = {
                    "encoding": "编码提示词 / 参考图",
                    "denoising": "开始去噪",
                    "decoding": "VAE 解码",
                }.get(str(phase), str(phase))
                if phase not in announced:
                    announced.add(str(phase))
                    _say(f"[qwen21-mlx] {label}", flush=True)
                if phase == "decoding" and last_total and first_step_at:
                    rate = (time.time() - first_step_at) / (last_total - 1) if last_total > 1 else None
                    _say(f"[qwen21-mlx] 采样 {render_progress(last_total, last_total, rate)}", flush=True)
                continue

            if frame.get("id") not in (None, request_id):
                continue  # stale frame from a killed run

            if kind == "error":
                raise Qwen21RunError(
                    f"生成失败：{frame.get('message')}\n{frame.get('traceback', '')}"
                )
            if kind == "result":
                self.peak_bytes = max(self.peak_bytes, int(frame.get("peak_bytes") or 0))
                _say(
                    f"[qwen21-mlx] 完成 {frame.get('width')}x{frame.get('height')} "
                    f"用时 {frame.get('elapsed')}s，MLX 峰值 {self.peak_bytes / 1e9:.1f} GB",
                    flush=True,
                )
                return frame

            if verbose:
                _say(f"[qwen21-mlx] {frame}", flush=True)

    def stop(self) -> None:
        if not self.proc:
            return
        try:
            if self.proc.poll() is None and self.proc.stdin:
                self.proc.stdin.write(json.dumps({"op": "shutdown"}) + "\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            pass
        finally:
            self.kill()

    def kill(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.kill()
                self.proc.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                pass
        if self._log_file:
            try:
                self._log_file.close()
            except OSError:
                pass
            self._log_file = None


# ------------------------------------------------------------- lifecycle ----


def _key(cfg: dict[str, Any]) -> str:
    return "|".join(
        [
            str(cfg["model"]),
            str(cfg.get("quantize", "auto")),
            str(bool(cfg.get("vae_tiling"))),
            str(bool(cfg.get("low_ram"))),
            str(cfg.get("cache_limit_gb", 2)),
            ",".join(str(p) for p in (cfg.get("lora_paths") or [])),
            ",".join(str(s) for s in (cfg.get("lora_scales") or [])),
            str(bool(cfg.get("lora_bake"))),
        ]
    )


def _lock_for(key: str) -> threading.Lock:
    with _REGISTRY_LOCK:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = _LOCKS[key] = threading.Lock()
        return lock


def get_worker(cfg: dict[str, Any]) -> Worker:
    key = _key(cfg)
    with _REGISTRY_LOCK:
        worker = _WORKERS.get(key)
        if worker is None or not worker.alive():
            worker = Worker(cfg)
            _WORKERS[key] = worker
        return worker


def run_generation(
    cfg: dict[str, Any],
    params: dict[str, Any],
    on_progress: Callable[[int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Run one generation, serialised per model configuration."""
    key = _key(cfg)
    lock = _lock_for(key)
    if not lock.acquire(timeout=1800):
        raise Qwen21RunError("该模型正被另一个生成占用，等待超时。")

    try:
        worker = get_worker(cfg)
        try:
            result = worker.generate(
                params,
                on_progress=on_progress,
                should_cancel=should_cancel,
                verbose=bool(cfg.get("verbose")),
            )
        except Qwen21RunError:
            worker.kill()
            raise
        finally:
            if not cfg.get("keep_resident", True):
                worker.stop()
        return result
    finally:
        lock.release()


def free_workers(model: str | None = None) -> int:
    """Stop resident workers (all, or only those serving ``model``)."""
    stopped = 0
    with _REGISTRY_LOCK:
        for key, worker in list(_WORKERS.items()):
            if model and model not in key:
                continue
            worker.stop()
            _WORKERS.pop(key, None)
            stopped += 1
    return stopped


@atexit.register
def _shutdown() -> None:  # pragma: no cover - process teardown
    try:
        free_workers()
    except Exception:
        pass
