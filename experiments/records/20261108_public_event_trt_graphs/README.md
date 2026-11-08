# 事件 TensorRT 和 CUDA Graph 加速入口

使用入口：`participant.solution_public_trt_graphs:Solution`。

此入口继承已验证的 Local TensorRT 版本，增加完整 gated/TCN 事件引擎，并对 Global、桌面 YOLO、事件引擎使用固定缓冲区及 CUDA Graph。Local 每帧可能执行两个裁剪，需要保留各自描述特征，因此继续使用原 TensorRT 调用方式。

GPU 特征缓存实验没有显示加速收益，未用于本入口。球/桌面特征存储、裁剪、事件阈值和 NMS 沿用原流程。原来的 `participant.solution_public_trt_local:Solution` 保留用于回退。

服务器运行示例：

```bash
cd /home/lirong/pingpong-public-ee754ef
export PYTHONUTF8=1
export PYTHONDONTWRITEBYTECODE=1
export YOLO_CONFIG_DIR=$PWD/out/yolo-settings
export TTNET_CHECKPOINT=/home/lirong/pingpong-tcn/out/new-public-weight-20261105/20261105-145517_public_event_best/best.pt
export TTNET_TRT_DIR=$PWD/out/public-trt-001/engines
export TTNET_LOCAL_TRT_DIR=$PWD/out/public-local-trt-001/engines
export TTNET_EVENT_TRT_DIR=$PWD/out/public-event-trt-001/engines
/opt/conda/envs/conda_env/bin/python pingpang/run.py --input /workspace/pingpong/pingpang/public_data --output out/NEW_UNIQUE_OUTPUT --solution participant.solution_public_trt_graphs:Solution --decoder gpu </dev/null
```

`NEW_UNIQUE_OUTPUT` 使用新的输出目录。需要同一 checkpoint、相容的 TensorRT 8.6.1 和 RTX 4090 引擎；换 GPU/运行环境应重新构建及验证。

事件模型使用 FP16 执行并保护显式 LayerNorm 算术为 FP32，未使用 INT8/INT4。模型导出副本把 Where 的标量零显式转换成与特征一致的类型，以兼容当前 PyTorch ONNX 导出和 TensorRT 解析；原始模型源码未修改。

CUDA Graph 输出缓冲区会复用，本入口按帧顺序执行，每次结果在下次对应图调用前被消费；不能直接把该封装用于并发调用或未复制的跨帧结果缓存。现有流式窗口/NMS 等待仍存在，吞吐 FPS 不代表零延迟。

实验记录：`out/public-event-trt-001/acceptance.json`。`baseline1/2/3` 和 `graphs1/2/3` 为正常速度轮；`capture`、`shadow`、`graphs-shadow` 和 `event-profile` 是诊断轮，不作为正式 FPS。

开发复现：新输出目录先运行 `capture_event_trt.py`，再运行 `build_event_trt.py`，用 `evaluate_event_variants.py` 做完整输出、精度、速度对照。抓取窗口来自推理流程，没有使用真实标签构造网络输入。输入抓取文件仅供构建验证，完成后清理。
