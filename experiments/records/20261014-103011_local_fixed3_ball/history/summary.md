# local 定位结果

状态：completed；数据集 f2552a61b4598b68b115854a6ec1995f61c1283eab673c7c376d47c2e2cf2c65；固定留出 40 视频、6292 帧。

| 项目 | 数值 |
|---|---:|
| best epoch | 13 |
| Global 阈值 | 0.7 |
| Local 阈值 | 0.7 |
| 最终球心 F1，10 px | 0.850786 |
| 最终球心 F1，50 px | 0.860632 |
| 匹配误差中位数 / P90，10 px | 1.0556053138120334 / 2.332162632107244 |
| 最长连续漏检，10 px | 22 |
| ROI 覆盖率 | 0.8721237268955111 |
| 原始视频完整定位 FPS | 141.21 |
| 峰值 GPU 显存 bytes | 54212608 |
| 参数量，Global | 553985 |
| Global GFLOPs（Conv2d） | 5.5562 |

性能为本机 PyTorch FP16 AMP，含 PyAV 解码、预处理、定位和 JSON 序列化；不含磁盘写入与事件模型。GPU 可能与其他任务共享，不能代表赛事硬件结果。AP 不适用于球心热图。

历史基线 F1@10px：0.977598；当前差值：-0.126812。视频划分相同，历史基线的解码与推理精度不同；这是历史对照，不是本次重新运行基线。当前时序模式：fixed3。

[逐视频指标](evaluations/best_val/per_video.json) · [完整指标](evaluations/best_val/metrics.json) · [错误索引](evaluations/best_val/errors.json) · [pred / label 预览](evaluations/best_val/index.html) · [测速](evaluations/benchmark/metrics.json) · [ONNX 对齐](exports/manifest.json)

未知标签不计入 P/R/F1，不把缺标判为背景。所有评分错误都有原图 pred / label 叠加图。未来事件接口已缓存验证；本阶段不识别触台。
