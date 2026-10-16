# Transformer finalist（对比）（transformer_frames_lr0.0001_d0.1）

与 TCN 同前端、同时间编码的 Transformer 时序头（2 层、4 头），受控对比中落后 TCN 约 4.5pp，保留用于对照。

## 结果（内部 PP 验证集，完整视频，落点宏 F1）

| 指标 | 值 |
|---|---:|
| 落点宏 F1（seed42） | 0.7503 |
| tp / fp / fn | 135 / 36 / 54 |
| e2e FPS | 79.8307 |

| 视角 | F1 |
|---|---:|
| side | 0.7553 |
| diagonal | 0.7308 |
| front | 0.7647 |

## 多种子（内部验证宏 F1）

| seed | 宏 F1 |
|---:|---:|
| 42 | 0.7503 |
| 43 | 0.7395 |
| 44 | 0.7818 |

## 外部评估（锁定阈值，未调参）

| 数据集 | 宏 F1 |
|---|---:|
| TTGames test | 0.8150（落点 0.9683 / 擦网 0.6617） |
| PP public | 0.6792 |

事件头参数量：314,210；头部前向 p50：0.5323 ms

## 文件说明

- `best.pt`：事件头权重（seed42）。
- `attempts/`：逐 epoch 训练日志；`training_curve.svg`：验证曲线。
- `metrics.json`：连续验证集完整评估；`external/`：TTGames test + PP public；`latency.json`：逐帧延迟。
- `overlays/`：错误叠图（`index.html` 浏览器查看，mp4 里红/绿框标出预测与真值，FP/FN 一目了然）。
- `seeds/`：多种子权重与训练记录。
