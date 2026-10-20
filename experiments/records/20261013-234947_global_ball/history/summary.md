# global 定位结果

状态：completed；数据集 4cd159537792c3b54edf7c3fea4768c2cfd7cc4b74ab46bdeef174ee060fb244；固定留出 40 视频、6292 帧。

| 项目 | 数值 |
|---|---:|
| best epoch | 6 |
| Global 阈值 | 0.7 |
| Local 阈值 | None |
| 最终球心 F1，10 px | 0.448305 |
| 最终球心 F1，50 px | 0.461713 |
| 匹配误差中位数 / P90，10 px | 1.8559890424459167 / 3.336128045020221 |
| 最长连续漏检，10 px | 45 |
| ROI 覆盖率 | None |
| 原始视频完整定位 FPS | 319.94 |
| 峰值 GPU 显存 bytes | 28336640 |
| 参数量，Global | 245057 |
| Global GFLOPs（Conv2d） | 2.3310 |

性能为本机 PyTorch FP16 AMP，含 PyAV 解码、预处理、定位和 JSON 序列化；不含磁盘写入与事件模型。GPU 可能与其他任务共享，不能代表赛事硬件结果。AP 不适用于球心热图。

历史基线 F1@10px：0.977598；当前差值：-0.529293。视频划分相同，历史基线的解码与推理精度不同；这是历史对照，不是本次重新运行基线。单帧双分支与历史三帧定位器的输入也不同。

[逐视频指标](evaluations/best_val/per_video.json) · [完整指标](evaluations/best_val/metrics.json) · [错误索引](evaluations/best_val/errors.json) · [pred / label 预览](evaluations/best_val/index.html) · [测速](evaluations/benchmark/metrics.json) · [ONNX 对齐](exports/manifest.json)

未知标签不计入 P/R/F1，不把缺标判为背景。所有评分错误都有原图 pred / label 叠加图。未来事件接口已缓存验证；本阶段不识别触台。
