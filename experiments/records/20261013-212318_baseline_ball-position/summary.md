# Baseline 小球位置检测验证

使用原始 ball.onnx 和未改动的 BallDetector，仅执行球定位。三帧 RGB 输入 1×9×288×512、FP32（CUDA EP 默认允许 TF32）、热图 >0.7，全部阈值以上像素的加权质心；ch1 对齐输入帧号−1。未运行触桌/落点规则。

## 数据与结果

复用固定 yolo_ball_v1 验证清单，40 个完整视频，6292 个原始帧，5302 个有球标注帧，990 个未知帧不计精度。train/val 与已有实验清单 SHA256 一致，视频和标注身份已核对。沿用已有场景分组 pilot，未重新划分。baseline 权重的训练来源和是否包含这些视频未知，因此这是本数据上的验证对照，不能据此证明独立泛化。

| 原图球心距离 | TP | FP | FN | Precision | Recall | F1 | 匹配误差中位数 / P90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ≤10px | 5084 | 15 | 218 | 0.997058 | 0.958883 | 0.977598 | 1.269 / 2.601px |
| ≤50px | 5098 | 1 | 204 | 0.999804 | 0.961524 | 0.980290 | 1.271 / 2.626px |

10px 匹配覆盖率 95.89%；最长连续未匹配 11 帧（未知帧处断开）。203 个有标签帧没有输出，另有 15 个输出位置未匹配。所有 21 个有标签末帧均计漏检，因为官方路径没有未来帧输入，不人为补黑帧。中间帧预测不使用当前帧 GT。输出最多一个球心，远离全部 GT 的一个输出同时贡献 FP 和 FN。

热图没有 bbox，mAP 留空；没有可靠无球标签，无球误报率留空。下游 conf≥0.75 的定位诊断另存 evaluations/onnx_fp32_ball.json（同一预测，无额外调阈值推理）。

## 速度与计算量

RTX 5070 Ti / ONNX Runtime CUDA：GPU 绑定输入输出的同步前向 p50 6.564ms、p95 7.299ms（warmup 50、3×200 次，含 CPU 调度，不含传输/预处理）。六个场景各首段完整视频，CPU 解码至 JSONL 写盘 E2E 111.23 FPS，包含 RGB 转换、预处理、传输、推理和解码球心，排除初始化及可视化。初始化 0.345s。

ONNX 权重/偏置元素 1.292659M；Conv/MatMul 27.105959 GMAC / 54.211918 GFLOPs（1 MAC=2 FLOPs），未计激活、融合 Add、SE Mul、pool、resize 和形状运算。ORT 显存不由 PyTorch allocator 管理，峰值显存未测，留 null。与 YOLO 的后端和原生输入不同，速度仅作本机各自流水线对照；没有进行赛事 TensorRT/4090 测试。

## 产物

- [全部叠加视频和代表帧](../../../out/experiments/20261013-212318_baseline_ball-position/eval/onnx_fp32_ball/index.html)：绿色 GT / 红色预测，覆盖所有错误和未知帧。
- [逐视频指标](../../../out/experiments/20261013-212318_baseline_ball-position/eval/onnx_fp32_ball/per_video.json)、[逐组指标](../../../out/experiments/20261013-212318_baseline_ball-position/eval/onnx_fp32_ball/per_group.json)、[全部错误索引](../../../out/experiments/20261013-212318_baseline_ball-position/eval/onnx_fp32_ball/errors.json)。
- [逐帧预测](../../../out/experiments/20261013-212318_baseline_ball-position/eval/onnx_fp32_ball/predictions.jsonl)、[完整评估](evaluations/onnx_fp32_ball.json)、[实际配置](config.yaml)。
- [日志与源码快照](../../../out/experiments/20261013-212318_baseline_ball-position/attempts/001)、[测速原始记录](../../../out/experiments/20261013-212318_baseline_ball-position/benchmark/onnx_fp32_ball)。

未修改官方检测器、solution 或受保护框架，完整性校验通过。未训练或调整 baseline。

## 核验与复现

5 项回归测试通过，覆盖真实 BallDetector 的三帧时序、reset、原图缩放、严格热图阈值与多 blob 解码，以及一对一匹配、FP/FN 和未知帧漏检区间。逐帧预测已用独立距离计算复核 TP/FP/FN；40 个 H.264 叠加视频的帧数与尺寸均匹配原视频，全部 218 个错误帧均有索引和叠加画面；已人工查看成功、漏检和定位偏差样本。

结论：该数据上的球定位覆盖率较高，10px F1 为 97.76%；实际 Solution 使用 conf≥0.75 后，定位 Precision/Recall/F1 为 99.74% / 94.57% / 97.09%。这两组成绩都不代表触桌/落点检测成绩。

```bash
.venv/bin/python scripts/test_baseline_ball.py
.venv/bin/python scripts/evaluate_baseline_ball.py
```

第二条命令自动创建新实验编号，不覆盖已有结果。原始执行版本保存于 attempts/001/evaluate_baseline_ball.py；当前脚本仅补充了完全未知视频在 HTML 目录中“未评分”的展示。
