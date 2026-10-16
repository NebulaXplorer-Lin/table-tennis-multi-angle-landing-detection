# 原始基线 B（参考）（baseline_B）

原事件头（旧 EventHead）+ 规则关联，无场景、无时间编码，作为本次所有改进的参考基线。

## 结果（内部 PP 验证集，完整视频，落点宏 F1）

| 指标 | 值 |
|---|---:|
| 落点宏 F1（seed42） | 0.7796 |
| tp / fp / fn | 148 / 41 / 41 |
| e2e FPS | 79.7525 |

| 视角 | F1 |
|---|---:|
| side | 0.7981 |
| diagonal | 0.7477 |
| front | 0.7931 |

## 多种子（内部验证宏 F1）

| seed | 宏 F1 |
|---:|---:|
| 42 | 0.7796 |
| 43 | 0.7585 |
| 44 | 0.7542 |

## 文件说明

- `best.pt`：事件头权重（seed42）。
- `attempts/`：逐 epoch 训练日志；`training_curve.svg`：验证曲线。
- `metrics.json`：连续验证集完整评估；`external/`：TTGames test + PP public；`latency.json`：逐帧延迟。
- `overlays/`：错误叠图（`index.html` 浏览器查看，mp4 里红/绿框标出预测与真值，FP/FN 一目了然）。
- `seeds/`：多种子权重与训练记录。
