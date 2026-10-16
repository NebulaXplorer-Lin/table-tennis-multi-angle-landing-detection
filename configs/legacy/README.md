# 历史方案

此处保留整理前配置，供核查及旧入口使用。新的完整方案从 `../experiments/` 选择。

| 方案 | 实现 / 历史入口 | 状态 |
|---|---|---|
| 单帧Residual FPN | `ttnet/src/models/localizer.py`；历史checkpoint自带配置 | Global完成，Local暂停 |
| fixed3 | `fixed3.json` | 当前定位基线 |
| fps_mask9 | `fps_mask9.json` | 已有接口，未完成训练验证 |
| 空间多尺度 | `multiscale.json` | 已有配置，未验证收益 |
| TCN frozen / joint | `event.json` + `ttnet.events.train --mode ...` | frozen完成，joint暂停 |
| 官方球定位ONNX参考 | `pingpang/participant/ball_detector.py`；历史评估脚本 | 历史对照 |
| YOLO球定位 | `scripts/run_yolo_ball*.py` | 历史实验，原结果保留 |
| YOLO + 落点规则 | `pingpang/participant/solution.py`、`solution_local.py` | 当前赛事入口 |

历史权重、划分和评估协议从 [统一实验索引](../../experiments/index.md) 查询。旧路径通过兼容链接保留；根目录源数据和赛事框架不变。
