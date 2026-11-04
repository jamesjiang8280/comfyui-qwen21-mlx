# comfyui-qwen21-mlx

在 ComfyUI 里用 **Apple MLX（mflux）** 跑 **Qwen-Image-2.1**：文生图 + 指令式图像编辑，
一套常驻 worker 同时服务两条链路。

```
ComfyUI 节点（PyTorch 进程）
   │  JSONL over stdin/stdout
   ▼
worker.py（mflux / MLX 进程，常驻）
   │  首次加载 ~9.6GB 4bit 权重
   ▼
Qwen-Image-2.1（Qwen3-VL 视觉塔 + DiT + VAE） ──▶ PNG ──▶ ComfyUI IMAGE
```

ComfyUI 自带的 Qwen 节点跑 PyTorch/MPS；这里把扩散计算交给 MLX，并把模型留在显存里，
第二张图开始不再付 1 分钟的加载成本。节点层不 import `mlx`/`mflux`（它们和 ComfyUI 的
`transformers`/`mlx` 版本会打架），只通过子进程协议通信。

## 节点

| 节点 | 作用 |
| --- | --- |
| **Qwen-Image-2.1 MLX 模型加载器** | 扫描 `~/ComfyUI-Shared/models/MLX`，检测模型能力（是否含视觉塔），输出配置对象 |
| **Qwen-Image-2.1 MLX 文生图** | prompt / 尺寸 / 步数 / seed → IMAGE，可选 step cache |
| **Qwen-Image-2.1 MLX 图像编辑** | 1~10 张参考图 + 指令，可选 mask（auto_mask / 扩写 / 自检取决于权重是否有 lm_head），输出 IMAGE + alpha MASK |
| **Qwen-Image-2.1 MLX LoRA 加载器** | 单个 LoRA：选 `models/loras` 下的 `.safetensors`，调强度/开关/是否烘焙 |
| **Qwen-Image-2.1 MLX LoRA Stack** | 一个节点叠 4 个 LoRA（空槽=`none`），可串联多个 Stack 节点继续叠 |
| **Qwen-Image-2.1 MLX 释放内存** | 停止常驻 worker，交还统一内存（换模型/跑别的负载前用） |

## 安装

```bash
cd ~/ComfyUI/custom_nodes
git clone https://github.com/jamesjiang8280/comfyui-qwen21-mlx.git
```

重启 ComfyUI（Desktop 里退出再开，或重启后端）后，节点出现在 **Qwen21 MLX** 分类下。

### mflux 解释器

worker 需要独立的 mflux 环境（和 ComfyUI 的 python 分开，避免 `mlx`/`transformers` 版本冲突）：

```
~/mflux-venv/bin/python     # mflux 0.24.0 + mlx 0.32.4，已实测
```

查找顺序：`QWEN21_MLX_PYTHON` → 本包 `.venv` → `~/mflux-venv` → `~/ComfyUI-Shared/mflux-venv`
→ ComfyUI 自己的 python。想指定别的环境：

```bash
export QWEN21_MLX_PYTHON=/path/to/python-with-mflux
```

自检：

```bash
bash custom_nodes/comfyui-qwen21-mlx/scripts/setup_env.sh
```

## 模型

推荐（含 Qwen3-VL 视觉塔，可做指令编辑）：

```
~/ComfyUI-Shared/models/MLX/JoyFusionAI__Qwen-Image-2.1-Edit-MLX-4bit   # 9.6GB, 4bit
```

- 加载器会用 `text_encoder/model.safetensors.index.json` 里的键判断能力：
  同时有 `visual.*` 与 `language_model.*` 才是完整 **Edit** 包；
  只有语言模型的包只能文生图 / 普通重绘，编辑节点会直接报错而不是假装能编辑。
- `quantization=auto` 表示沿用权重自带的量化元数据。**已经是 Q4 的包不要再选 4**。
- **lm_head 很重要**：这个 4bit 包只有 `visual.*` + `language_model.*`，没有 `lm_head`。
  指令编辑（图生图 + 指令）正常，但 `auto_mask` / `enhance_prompt` / `verify` 需要模型生成文本，
  用不了；编辑节点会在开始去噪前直接报错，而不是白跑一轮。局部修改请用手动 mask。
- LoRA 用独立的 **LoRA 加载器**（单个）或 **LoRA Stack**（最多 4 个槽）节点：下拉来自 ComfyUI 的
  `loras` 目录（含 extra_model_paths），每个槽独立强度；`enabled` 做 A/B，`bake=false` 用运行时适配器
  （显存更省）。同一节点里填多个槽就是一次加载多个 LoRA，worker 日志会逐个打印 `Applied to N layers`；
  4 个槽不够时把两个 Stack 节点串起来即可。

## 工作流

```
user/default/workflows/Qwen Image 2.1 文生、图像编辑一体化.json
```

节点包里也有一份：`custom_nodes/comfyui-qwen21-mlx/workflows/qwen_image_21_mlx.json`。

- 两条链路共用一个加载器 / 一个常驻 worker；LoRA Stack 串在加载器和两条分支之间。
- 默认在 Stack 的 `lora_1` 接 `Qwen-image-2.1-turbo8_lora_8step`，8 步出图；`lora_2~4` 空槽为 `none`，
  可以再叠风格/画质 LoRA（记得同时调低各自 strength）。关掉 Stack 的 `enabled` 就是 40 步基础模型。
- 文生图：1024²、8 步、guidance 1.0。
- 编辑：参考图接 `image_1`，多图接 `image_2`~`image_10`；`width/height=0` 表示按参考图比例自动。
- 局部修改：MASK 接 `mask`（白=重绘，黑=保留）。本权重不支持 `auto_mask`。
- `strength=1.0` 完全按指令重画；`0.6~0.8` 保留更多原图，同时跳过前段高噪声步数（更快）。

## 实测（M5 / 24GB，4bit，磁盘缓存已热）

| 场景 | 耗时 | MLX 峰值 |
| --- | --- | --- |
| 512² 文生图 2 步（含首次加载） | 8.9 s | 9.9 GB |
| 1024² 编辑 2 步（1 张参考图，含首次加载） | 40.2 s | 10.2 GB |
| 1024² 编辑 2 步（第二次请求，复用 transformer） | 39.7 s | 13.9 GB |
| 768² 编辑 2 步（经 ComfyUI 完整链路） | 28.6 s | 13.6 GB |
| 512² 编辑 8 步 + turbo8 LoRA（含加载） | 34.8 s | 11.6 GB |
| 1024² 编辑 8 步 + turbo8 LoRA（含编码器重载） | 104 s | 11.6 GB |
| 参考：原 PyTorch/MPS 一体工作流 40 步 | 6 min 34 s | — |

基础模型稳态约 **8.4 s/步 @1024²**（第 1 步含权重/编码开销），40 步约 **5.5~6 分钟**；
接 turbo8 LoRA 后 8 步约 **1 分 45 秒**。提速手段，按性价比排序：

1. 用 **LoRA 加载器** 接 8 步蒸馏 LoRA（本机已验证 `Qwen-image-2.1-turbo8_lora_8step`，227/227 层匹配）；
2. `strength=0.7~0.8`：跳过前 20~30% 的步数，重绘类任务几乎无损；
3. `use_step_cache=True`（≥10 步才生效）：约 1.4x，画面有轻微差异；
4. 降到 768² 出图再放大。

## 两个关键设计

**编码后释放文本编码器。** Qwen3-VL 编码器占 4.8GB，只在 prompt 编码时需要。worker 在进入
去噪循环前把它卸掉（mflux CLI 的 MemorySaver 同款做法），下一个请求再从磁盘按需重载：

| | 编码器常驻 | 编码后释放 |
| --- | --- | --- |
| 1024² 第 2 步 | 14.8 s | 8.3 s |
| 峰值 | 15.0 GB | 10.2 GB |

**每次请求重建回调注册表。** 避免释放器/进度回调跨请求累积，也避免失败请求留下半套回调。

## 日志与排障

worker 的 stdout/stderr 写到：

```
custom_nodes/comfyui-qwen21-mlx/logs/worker-<模型名>.log
```

目录不可写时自动落到 `$TMPDIR/qwen21-mlx-logs/`（可用 `QWEN21_MLX_LOG_DIR` 覆盖）。

| 现象 | 处理 |
| --- | --- |
| 提示找不到 mflux 环境 | 设 `QWEN21_MLX_PYTHON=~/mflux-venv/bin/python`，或 `pip install -U 'mflux>=0.24'` |
| 首次生成特别慢（1~2 分钟） | 正常：权重首次从磁盘载入；之后再生成走常驻 worker |
| 内存告警 / 系统卡顿 | 加载器打开 `low_ram` 或 `vae_tiling`，`cache_limit_gb` 设 2~4；或先跑「释放内存」节点 |
| 编辑节点报“没有视觉塔” | 换完整 Edit 包（`text_encoder` 内含 `visual.*`） |
| 报 `no text-encoder lm_head` | 该权重缺 Qwen3-VL 的文本生成头：关掉 auto_mask / enhance_prompt / verify，或改用手动 mask |
| LoRA 不生效/层匹配少 | 确认是 Qwen-Image-2.1 的 diffusers 键名 LoRA；mflux 日志会打印 `Applied to N layers` |
| 中断后下一张变慢 | 中断会杀掉 worker，下次重新加载；属预期 |
| `mask` 没效果 | 确认白色=要改的区域；`mask` 与 `auto_mask` 同时接时 mflux 只用 mask |

## 已知限制

- 每个模型配置同一时刻只跑一个请求（节点内加锁串行），并发队列会排队。
- 输入是 batch 时只取第一帧作为参考图。
- 输出是 RGBA：`image` 是白底合成后的 RGB，透明通道在第二个输出 `alpha`（MASK）。
- `enhance_prompt` / `verify` / `auto_mask` 需要 lm_head；本机这个 4bit 包没有，节点会提前拦截。
- LoRA 烘焙（`bake=true`）会把被折叠加固的 4bit 层提到 8bit，显存明显变高（512² 实测峰值 16.1GB vs
  运行时适配器 11.6GB），默认不烘焙。
- 依赖 mflux 的 Qwen-Image-2.1 实现（0.24 实测）；升级 mflux 后若内部接口变动，需要同步 `worker.py` 的重载逻辑。
