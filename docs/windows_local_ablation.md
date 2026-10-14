# Windows 本地运行与 MIN_GAP 参数敏感性测试

本机为 RTX 5060 Laptop。本地使用 Python 3.10、torch 2.7.1 CUDA 12.8、ONNX Runtime GPU 1.22.0。推理模型仍为官方 ball.onnx，FP32、关闭 TF32；视频由 CPU/PyAV 解码。运行规则继承官方 Solution，只替换模型准备方法，并通过 LOCAL_MIN_GAP 设置间隔。

## 准备环境

在仓库外层 participant 的 PowerShell 中执行：

```powershell
python -m pip install uv
uv sync --locked
uv run --locked python scripts/check_local_env.py
```

`uv sync --locked` 会使用 uv.lock 中固定的版本准备项目 .venv；如锁文件与项目要求不一致，会报错而不是修改锁文件。首次需要下载 Python 3.10 和 GPU 依赖，可能较慢、占用数 GB 空间。不要手改 uv.lock，不要在比赛服务器固定环境执行 uv sync。

本地只具备数据和模型时才可运行。视频、模型被 Git 忽略，不会随 clone 获取。放置位置：

```text
pingpang/participant/weights/ball.onnx
pingpang/public_data/manifest.json
pingpang/public_data/gt_public.jsonl
pingpang/public_data/videos/...
```

检查脚本验证 GPU 矩阵计算、真实 ONNX 模型输出、CUDA 卷积节点执行记录以及视频首帧解码。显示 LOCAL ENV CHECK PASSED 才继续。失败时提供完整错误，不自动降级为 CPU 模型推理。

## 先建立本地原始对照

```powershell
uv run --locked python tools/run_local_ablation.py --gaps 10
```

先确认 15 条视频成功。然后进行同一环境下的参数对比：

```powershell
uv run --locked python tools/run_local_ablation.py --gaps 14 12 10 8 6
```

每组一个独立进程，只改变最小落点间隔。每次调用生成新的时间编号，避免覆盖已有结果。

脚本沿用 `run_local_ablation.py` 文件名，但这项测试严格属于参数敏感性分析（Parameter Sensitivity Analysis）：保留间隔过滤机制，只改变参数值，没有移除算法组件。

如果 uv 尚未加入 PATH，可把命令中的 `uv` 替换为 PowerShell 写法 `& "$env:USERPROFILE\.local\bin\uv.exe"`。不需要手动激活外层目录的虚拟环境；`uv run` 会选择本项目的 `.venv`。

## 产物

- `experiments/records/<编号>/config.json`：设置。
- `experiments/records/<编号>/run.json`：运行状态、命令、环境和文件身份。
- `experiments/records/<编号>/evaluation.json`：本地估算效果与真实运行速度。
- `experiments/*_comparison.csv`、`*_comparison.md`：本次各组比较表。
- `out/experiments/<编号>/eval/public/`：预测、状态、日志。

本地 F1 采用帧容差 ±1、距离 50 像素、有效标注一对一最大匹配；未匹配预测若落入同样容差的 invalid GT 则忽略。官方忽略点算法未提供，因此本地分数只用于一致的对照，非正式成绩。

参数选择使用公开验证集，需要在实验报告声明；反复调参可能过拟合公开验证集。不要拿服务器 TensorRT baseline 与本机 ORT 改版作单变量比较。本地 FPS 不代表官方速度。

本次新增入口不属于比赛默认启动路径，Docker 排除这些辅助工具，官方框架不改。最终选定规则后，还要在官方 4090/TensorRT 环境重新验证。
