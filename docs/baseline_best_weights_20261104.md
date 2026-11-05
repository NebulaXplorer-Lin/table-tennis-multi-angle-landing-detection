# 历史 baseline 最佳一组权重

更新日期：2026-11-04（Asia/Shanghai）。按用户要求，整理目录和压缩包仅保留内部验证最高的 **TCN seed44，第 16 轮**。

| 项目 | 内容 |
| --- | --- |
| 权重 | `out/weights/baseline_best_20261104/best.pt` |
| 压缩包 | `out/weights/baseline_best_20261104.tar.gz` |
| 内部落点宏 F1 | **81.10%**（原始值 0.8110289623627295） |
| Bounce / Net 阈值 | **0.3 / 0.5** |
| Global / Local 定位阈值 | 0.7 / 0.7 |
| seed / 最佳轮次 | 44 / 16 |
| 原始来源 | `result/tcn_frames_lr0.0003_d0.1/seeds/seed44/best.pt` |
| SHA256 | `f4a694eb28d511493a28fa11076276213d89132800e24e059df558b6e7dd37b2` |

这一组采用三帧 Global/Local、历史轨迹规则关联、旧桌框几何、真实时间编码和 13 帧 TCN。`best.pt` 内含 **Global、Local、桌框头、事件头**全部模型参数，完整推理直接加载这一份文件。

内部验证采用固定 PP 40 个视频、原帧 ±1、原图 50px、一对一匹配、三视角宏 F1；81.10% 是历史主基线 TCN 三种子中的最高验证成绩。该 seed44 没有对应的外测记录。

包内保留 `config.yaml`、`evidence/metrics.json`、`evidence/best_selection.json`、`manifest.json`、`verification.json` 和 `SHA256SUMS`。原配置中的历史路径用于追溯，完整模型可通过以下方式加载：

```python
from pathlib import Path
import torch
from ttnet.events.engine import EventSystem

path = Path("out/weights/baseline_best_20261104/best.pt")
state = torch.load(path, map_location="cpu", weights_only=False)
system = EventSystem(state["config"], torch.device("cpu"), initialize=False)
system.load_state_dict(state["system"], strict=True)
system.configure(state["config"]["mode"], False)
thresholds = state["thresholds"]  # [Bounce, Net] = [0.3, 0.5]
```

解压后在整理目录执行 `sha256sum -c SHA256SUMS` 可核对文件。已检查权重与原件逐字节一致、历史评估哈希一致、配置/轮次/阈值一致，以及严格 CPU 加载。本次未重跑精度评估。

本次精简范围是上述整理目录和压缩包；原始实验目录及历史记录保留。
