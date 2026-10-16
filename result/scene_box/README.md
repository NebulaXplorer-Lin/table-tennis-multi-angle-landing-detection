# 桌框场景融合（box）（scene_box）

在规则关联 + TCN 基础上加入桌框位置/尺度几何分支（scene_mode=box），相对无场景基线 +0.76pp，是场景融合的正收益代表。

## 结果（内部 PP 验证集，完整视频，落点宏 F1）

| 指标 | 值 |
|---|---:|
| 落点宏 F1（seed42） | 0.7873 |
| tp / fp / fn | 152 / 48 / 37 |
| e2e FPS | 85.3916 |

| 视角 | F1 |
|---|---:|
| side | 0.7873 |
| diagonal | 0.7358 |
| front | 0.8387 |

## 多种子（内部验证宏 F1）

| seed | 宏 F1 |
|---:|---:|
| 42 | 0.7873 |

## 文件说明

- `best.pt`：事件头权重（seed42）。
- `attempts/`：逐 epoch 训练日志；`training_curve.svg`：验证曲线。
- `metrics.json`：连续验证集完整评估；`external/`：TTGames test + PP public；`latency.json`：逐帧延迟。
- `overlays/`：错误叠图（`index.html` 浏览器查看，mp4 里红/绿框标出预测与真值，FP/FN 一目了然）。
