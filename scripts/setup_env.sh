#!/usr/bin/env bash
# Prepare the Qwen-Image-2.1 MLX node pack.
#
#   bash scripts/setup_env.sh            # probe interpreter + weights, print status
#   bash scripts/setup_env.sh --weights  # ...and download the recommended 4-bit edit package
#
# Nothing is installed into ComfyUI's own environment: mflux pins its own
# mlx/transformers, ComfyUI pins different ones, so the worker runs in its own
# interpreter (this machine already has ~/mflux-venv).

set -euo pipefail

NODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${QWEN21_MLX_MODELS_DIR:-$HOME/ComfyUI-Shared/models/MLX}"
MODEL_DIR="$MODELS_DIR/JoyFusionAI__Qwen-Image-2.1-Edit-MLX-4bit"
REPO="JoyFusionAI/Qwen-Image-2.1-Edit-MLX-4bit"
WANT_WEIGHTS=0

for arg in "$@"; do
  case "$arg" in
    --weights) WANT_WEIGHTS=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

say() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

[ "$(uname -s)" = "Darwin" ] || die "只支持 macOS（MLX 需要 Apple Silicon）。"
[ "$(uname -m)" = "arm64" ] || die "需要 Apple Silicon (arm64)，当前 $(uname -m)。"

# ------------------------------------------------------------- interpreter --
PY="${QWEN21_MLX_PYTHON:-}"
if [ -z "$PY" ]; then
  for cand in "$NODE_DIR/.venv/bin/python" "$HOME/mflux-venv/bin/python" "$HOME/ComfyUI-Shared/mflux-venv/bin/python"; do
    if [ -x "$cand" ]; then PY="$cand"; break; fi
  done
fi
if [ -z "$PY" ]; then PY="$(command -v python3 || true)"; fi
[ -n "$PY" ] && [ -x "$PY" ] || die "找不到 python3。"

"$PY" - <<'EOF' || die "该 Python 里没有 mflux 的 Qwen-Image-2.1 模型。请 pip install -U 'mflux>=0.24' 或用 QWEN21_MLX_PYTHON 指定。"
import importlib.metadata as md
import mlx.core as mx
import mflux.models.qwen21.variants.edit.qwen_image_21_edit  # noqa: F401
print("mflux", md.version("mflux"), "| mlx", md.version("mlx"), "| device", mx.default_device())
EOF

say "Python: $PY"

# -------------------------------------------------------------- download ----
if [ "$WANT_WEIGHTS" = "1" ]; then
  if [ -d "$MODEL_DIR/transformer" ]; then
    say "权重已存在：$MODEL_DIR"
  else
    say "下载 $REPO -> $MODEL_DIR（约 9.6GB，4bit，含 Qwen3-VL 视觉塔）"
    mkdir -p "$MODEL_DIR"
    HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" HF_HUB_DISABLE_XET=1 \
      "$PY" -m huggingface_hub.commands.huggingface_cli download "$REPO" \
        --local-dir "$MODEL_DIR" --max-workers 4
  fi
fi

# ---------------------------------------------------------------- inspect ---
if [ -d "$MODEL_DIR/text_encoder" ]; then
  CAP="$("$PY" - "$MODEL_DIR" <<'EOF'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
index = root / "text_encoder" / "model.safetensors.index.json"
keys = json.loads(index.read_text())["weight_map"]
visual = sum(k.startswith("visual.") for k in keys)
language = sum(k.startswith("language_model.") for k in keys)
print(("edit" if visual and language else "txt2img")
      + f" (visual={visual}, language_model={language})")
EOF
)"
else
  CAP="缺失"
fi

cat <<EOF

$(say "状态")
  Python : $PY
  模型   : $MODEL_DIR
  能力   : $CAP

冒烟测试（不经过 ComfyUI，直接用 mflux；首次约 1 分钟加载）：
  "$PY" -m mflux.models.qwen21.cli.qwen21_edit_generate \\
      --model "$MODEL_DIR" --base-model qwen-image-2.1 \\
      --prompt "把外套改成黑色皮衣，保持人物身份、姿势和背景不变" \\
      --steps 4 --guidance 1.0 --seed 42 --vae-tiling \\
      --image-paths /path/to/reference.png --output /tmp/qwen21_smoke.png

然后把整个目录放进 ComfyUI/custom_nodes/（本目录已就位）并重启 ComfyUI。
EOF
