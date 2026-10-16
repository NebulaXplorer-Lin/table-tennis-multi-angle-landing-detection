# 20261016 桌框 / 自回归 / 事件头实验结果

实验根目录：`out/experiments/20261016_table_ar`；本次归档整理其中的有潜力结果。

## 结论速览

| 方案 | 内部验证宏 F1（seed42） | 三种子均值±std | 说明 |
|---|---:|---:|---|
| tcn_frames_lr0.0003_d0.1 | 79.56% | 78.77% ± 2.30pp | **推荐**（TCN+桌框+时间编码） |
| scene_box | 78.73% | — | 桌框几何分支正收益 |
| transformer_frames_lr0.0001_d0.1 | 75.03% | 75.72% ± 1.80pp | 对比（落后 TCN 4.5pp） |
| baseline_B | 77.96% | 76.41% ± 1.11pp | 原始基线 |

## 核心结论

1. **桌框位置/尺度（box）有正收益（+0.76pp）**，加球台相对关系反而略降；桌框预测质量（IoU 38.75%）不是瓶颈。
2. **自回归替换规则打分失败**（beam1/4/8 均 ~66%，落后 12pp），根因是 null 分支欠训练（训练集 null 目标仅 0.7%）。
3. **TCN 显著优于 Transformer**（统一前端后 +4.53pp），且跨调参网格、跨种子、跨外部集一致。
4. 最终推荐 `tcn_frames_lr0.0003_d0.1`，三种子平均 78.77%、全部 ≥50 FPS。

## 目录

- [tcn_frames_lr0.0003_d0.1](./tcn_frames_lr0.0003_d0.1/README.md) — TCN finalist（推荐）
- [scene_box](./scene_box/README.md) — 桌框场景融合（box）
- [transformer_frames_lr0.0001_d0.1](./transformer_frames_lr0.0001_d0.1/README.md) — Transformer finalist（对比）
- [baseline_B](./baseline_B/README.md) — 原始基线 B（参考）

