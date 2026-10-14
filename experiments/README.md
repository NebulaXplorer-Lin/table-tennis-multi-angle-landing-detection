# 球定位实验

本轮任务：YOLO 作为单帧球定位模型，暂不执行触桌识别。

`comparison.csv` 是汇总源，`comparison.md` 由运行脚本生成。所有 AP / P / R / F1 存 0–1；缺失指标留空 / null。每次运行在 records 下登记状态、配置、环境和评估结果，大产物在 out/experiments。

## ball_v1 协议

- YOLO11n COCO 预训练，替换为 ball 单类别；首轮 640 输入、15 epoch 上限、AdamW、seed=42、FP32。训练 3,162 张，验证 5,302 张，按场景整体划分；不进行触桌标签推导。优化器每步累积由 Ultralytics nbs=64 和 warmup 决定，实际完整参数保存在 config.yaml。
- best 按验证 mAP50–95 选择。AP 使用置信度下限 .001、NMS IoU .7、max_det=10；报告 Ultralytics AP 曲线上最佳 F1 处的 P/R，该 P/R 不是置信度 .25 的固定阈值结果。
- 另报固定 conf=.25 下 bbox IoU≥.5 的 P/R/F1；球心距离≤10、≤50 **原图像素**的 P/R/F1、匹配球心误差中位数/P90和覆盖率。匈牙利算法最大化有效匹配数量，再最小化距离（框匹配最大化 IoU）。混合 720p/1080p，不归一化像素阈值。
- 只评估有球框帧，其他球形物体或未标球可能影响 FP；无球误报率无法可靠计算，留 null。连续漏检在未知帧/视频边界断开，不跨未知帧补轨迹。
- 前向测速为 1×3×640×640、FP32、融合网络，warmup=50、3×200 次 CUDA events。E2E 为六个留出目录各首段完整视频，包含 CPU 解码、预处理、传输、预测、NMS、回传和 JSONL 写盘，不包含初始化；单帧 predictor 使用 rect=True，实际输入为按 stride 对齐的 letterbox 矩形，与前向固定方形测速不同。
- THOP 对未融合单类别网络统计参数 / MACs，1 MAC=2 FLOPs；不保证覆盖 functional attention matmul 等操作。记录单独的前向与训练显存，GPU 为 RTX 5070 Ti，不能代表赛事 4090。
- 不涉及 ONNX/TensorRT 部署。Ultralytics 结束训练时会 strip best/last 的优化器，因此这两个文件适合推理，不能保证精确恢复训练；本轮未要求恢复。

## 重跑

实验依赖隔离在 `out/yolo-deps`，版本清单位于 `out/yolo_setup/requirements.txt`；使用本项目 `.venv` 的 torch / numpy / OpenCV。不会修改赛事固定环境。

```bash
PYTHONPATH=out/yolo-deps .venv/bin/python scripts/prepare_yolo_ball.py
# 使用全新的时间戳实验编号，先创建日志父目录，再重定向：
.venv/bin/python scripts/run_yolo_ball.py YYYYMMDD-HHMMSS_yolo11n_ball-640 > out/experiments/YYYYMMDD-HHMMSS_yolo11n_ball-640/attempts/001/console.log 2>&1
```

官方模型与调用参考：[YOLO11](https://docs.ultralytics.com/models/yolo11/)、[Train](https://docs.ultralytics.com/modes/train/)、[Val](https://docs.ultralytics.com/modes/val/)。固定实际使用的 ultralytics=8.3.225；不依赖文档中后续版本的行为。
