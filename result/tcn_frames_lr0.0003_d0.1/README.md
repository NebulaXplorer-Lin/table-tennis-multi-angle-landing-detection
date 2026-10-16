# TCN finalist（推荐）（tcn_frames_lr0.0003_d0.1）

受控对比的推荐方案：TCN 时序头 + 桌框几何(box) + 时间编码 + frames 窗口 + lr3e-4 + dropout0.1。三种子平均 78.77%，是本次实验的最优结果。

## 结果（内部 PP 验证集，完整视频，落点宏 F1）

| 指标 | 值 |
|---|---:|
| 落点宏 F1（seed42） | 0.7956 |
| tp / fp / fn | 144 / 34 / 45 |
| e2e FPS | 77.5279 |

| 视角 | F1 |
|---|---:|
| side | 0.7847 |
| diagonal | 0.7451 |
| front | 0.8571 |

## 多种子（内部验证宏 F1）

| seed | 宏 F1 |
|---:|---:|
| 42 | 0.7956 |
| 43 | 0.7563 |
| 44 | 0.8110 |

## 外部评估（锁定阈值，未调参）

| 数据集 | 宏 F1 |
|---|---:|
| TTGames test | 0.8353（落点 0.9453 / 擦网 0.7254） |
| PP public | 0.7097 |

事件头参数量：197,858；头部前向 p50：0.3896 ms

## 文件说明

- `best.pt`：事件头权重（seed42）。
- `attempts/`：逐 epoch 训练日志；`training_curve.svg`：验证曲线。
- `metrics.json`：连续验证集完整评估；`external/`：TTGames test + PP public；`latency.json`：逐帧延迟。
- `overlays/`：错误叠图（`index.html` 浏览器查看，mp4 里红/绿框标出预测与真值，FP/FN 一目了然）。
- `seeds/`：多种子权重与训练记录。
