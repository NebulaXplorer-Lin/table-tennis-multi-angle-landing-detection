# RTX 5070 Ti 本地 uv 环境

本地开发环境位于仓库 `.venv`，依赖由 `pyproject.toml` 和 `uv.lock` 固定。
适用于 Linux x86_64、RTX 5070 Ti 16GB；安装时检测到驱动 595.91.07。
`nvidia-smi` 显示的 CUDA 13.2 是驱动支持上限，项目使用 PyTorch 自带的 CUDA 12.8 运行库。

| 组件 | 本地版本 |
|---|---|
| Python | 3.10（具体补丁版本由 uv 安装） |
| PyTorch / torchvision | 2.7.1+cu128 / 0.22.1+cu128 |
| NumPy | 1.26.4 |
| ONNX | 1.16.2 |
| ONNX Runtime GPU | 1.22.0，CUDA 12 / cuDNN 9 |
| OpenCV headless / SciPy | 4.11.0.86 / 1.15.3 |
| PyAV | 15.1.0 |

## 安装与检查

在仓库根目录运行：

```bash
uv sync --locked
uv run --locked python scripts/check_local_env.py
```

uv 从官方 CUDA 12.8 索引下载 torch 和 torchvision，其他包使用 PyPI。
不需要另装系统 CUDA Toolkit；编译自定义 CUDA 扩展时才需另行配置工具链。
自检包含 CUDA 卷积反向传播、torchvision GPU NMS、两份实际 ONNX 模型推理、
ORT profiling 中的 CUDA 节点执行记录，以及本地训练视频的 PyAV 解码。

也可以 `source .venv/bin/activate`；IDE 解释器选择 `.venv/bin/python`。
在受限沙箱中 GPU 可能不可见，请在可访问宿主 GPU 的终端执行自检。

## 本地运行官方算法

两个新增的 `participant/solution_local.py` 继承官方算法，仅替换 `prepare()`，
使用 ONNX Runtime CUDA 加载同一份权重；初始化失败时明确报错，禁止 session 运行时整体回退。
ORT 仍可能把个别形状运算安排到 CPU，这与整个网络回退 CPU 不同。

```bash
uv run --locked python pingpang/run.py \
  --input pingpang/public_data --output out/local_pingpang \
  --solution participant.solution_local:Solution --decoder cpu

uv run --locked python basketball/run.py \
  --input basketball/public_data --output out/local_basketball \
  --solution participant.solution_local:Solution --decoder cpu
```

这里 `--decoder cpu` 仅指视频解码；模型仍然在 GPU 推理。PyAV wheel 自带解码库，
无需系统安装 ffmpeg。`imageio-ffmpeg` 另外提供独立 ffmpeg 可执行文件，路径可查看：

```bash
uv run python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())'
```

官方 `--decoder gpu` 路径通过 PATH 查找 `ffmpeg`，且需要其支持 NVDEC；
本次检查发现随包 ffmpeg 7.0.2 的 `-hwaccels` 只有 vdpau，不支持 CUDA 解码，
因此本地命令明确使用 `--decoder cpu`。没有安装或替换系统 ffmpeg。

## 本机验证记录（2026-10-13）

- Python 3.10.21；GPU capability `(12, 0)`；torch 2.7.1+cu128。
- CUDA 卷积前向、反向梯度和 torchvision GPU NMS 通过。
- `ball.onnx` 输出 `(1, 3, 288, 512)`，profiling 记录 311 个 CUDA 节点事件。
- `player.onnx` 输出 `(1, 300, 6)`，profiling 记录 290 个 CUDA 节点事件。
- 两个 ONNX 文件检查通过，推理输出均为有限数值；PyAV 成功解码训练视频。
- 两任务官方文件完整性校验通过。
- 完整运行器抽样：乒乓球 125 帧 / 4 落点、篮球 750 帧 / 7592 框，
  两者视频状态均为 `ok`，输出 schema 错误数均为 0。
  结果在 `out/local_smoke/{pingpang,basketball}/output/`；这是环境与流程测试，未计算准确率。

## 与官方提交环境的边界

- 本地环境不是官方评测镜像。不要在 `sport-base:v1` 中执行本项目的 `uv sync`。
- 官方 `solution.py`、`core/`、`run.py`、环境检查及完整性检查脚本保持原样。
- 官方环境检查锁定 torch 2.1.2+cu118 / TensorRT 8.6，会拒绝本地环境，这是预期行为。
- 本地未安装 TensorRT：8.6 不适合 Blackwell，10.x 又更换了官方代码使用的 binding API。
- `.dockerignore` 排除 `.venv`、uv 配置和本地入口；官方入口仍使用原 TensorRT 路径。
- 提交前必须在 4090 官方镜像中验证 ONNX → TensorRT、FP16 数值差异、最终指标和总耗时。
  本地 ORT 通过不代表官方 TensorRT 验收通过。

版本依据：[PyTorch 官方安装矩阵](https://pytorch.org/get-started/previous-versions/)、
[ORT CUDA/cuDNN 兼容表](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)、
[TensorRT 8 → 10 Python 接口迁移](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/api/tensorrt-8x-to-10x-python-api.html)。

## 本轮复核（2026-10-15）

`pyproject.toml` / `uv.lock` 与 `.venv` 一致，实际为 PyTorch 2.7.1+cu128。
受限沙箱内 `torch.cuda.is_available()` 为 false 且 NVML 初始化失败；在允许访问
宿主 GPU 的执行环境中可识别 RTX 5070 Ti，CUDA 前向/反向及四组真实样本检查通过。
这属于设备可见性限制，不需要改动 CUDA wheel 或重新安装依赖。

四组完整视频预训练前测速命令：

```bash
.venv/bin/python -m ttnet.events.benchmark_comparison --device cuda \
  --video 02_001 --video 22_001 --output out/results/new_comparison_runtime
```

若受限沙箱不可见 GPU，应在可访问宿主 GPU 的终端运行。两个 PP 视频共 762 帧，
PyTorch AMP 完整推理 A/B/C/D 分别约 98.31/89.99/90.63/86.46 FPS；这是已训练定位器
加初始化事件头的运行检查，包含原视频解码和序列化，不包含初始化、画图和磁盘写入，
不代表事件准确率或整个验证集的最终测速。PP 原始帧号/真实时间戳模式支持非等间隔 PTS。
