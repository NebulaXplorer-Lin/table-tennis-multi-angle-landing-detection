# TTNet 第一阶段：球定位

独立的数据与训练工程。原始视频只读；不覆盖 YOLO 数据或结果，不修改赛事框架和 `Solution`。事件模型尚未实现，本阶段提供可复用的视觉特征接口。

## 数据

当前数据集为 `datasets/ball_v2`，覆盖全部 154 视频的 23,690 帧：20,846 帧有有效球框，2,844 帧缺标且标记 unknown。固定视频划分 114 train / 40 val，沿用用户批准的历史场景分组；与当前 SOP 的逐文件夹 80/20 默认不同，不重新划分。不能据此声称独立比赛场次测试成绩。

数据集只保存原始帧与原始标签信息。`labels/<video_id>.jsonl` 保留每一帧的 fps、fps 精确分数、基于 PTS 的名义帧率、原图分辨率、球框/球心、监督状态与原始其他类别标注；不保存模型输入尺寸、固定缩放变换或 Global 输入图像路径。坐标使用原图像素，时间戳相对首帧。missing ball ≠ negative。

v2 的 `frames/` 通过相对符号链接复用 v1 的原始 PNG，不重复解码或复制图片；v2 不包含 `global_inputs/`。v1 作为历史数据版本保留，与已完成或暂停的训练权重对应。共享原图不可修改，也不能只删除 v1 的原图目录。

`metadata/` 保存数据身份、每视频来源哈希、实际解码帧数、PTS 审计和处理版本。`splits/` 是 TTNet 自己的固定视频清单。轻量来源清单另存 `configs/frozen_split.json`，正常训练不再依赖 YOLO 文件。

处理可按视频恢复；已完成视频会验证现有 PNG。来源、图像处理或采样策略变化时必须更换数据版本，不能覆盖现有身份。

```bash
.venv/bin/python -m ttnet.data_tools.prepare --pilot 4
.venv/bin/python -m ttnet.data_tools.prepare --workers 4
.venv/bin/python -m ttnet.data_tools.migrate_raw  # 已有 v1 时生成 v2，复用原图
.venv/bin/python -m ttnet.data_tools.audit --full-images
.venv/bin/python -m unittest discover -s ttnet/tests -v
```

## 模型和训练

两路独立的残差编码器（16/32/64/96，GroupNorm）+ 多尺度 FPN + 单通道 sigmoid 热图。Global 输入宽×高 512×288，输出 256×144；Local 输入 256×256，输出 128×128。1080p 的 ROI 边长 256 px，其他分辨率按高度缩放，越界部分补黑，裁剪中心不移动。

最强局部峰附近 7×7 加权定位，不混合远处候选。Global 负责逐帧重捕获；Local 不可靠时返回有效 Global；两者均无效时返回缺失。

`global_input` / `local_input` 是训练配置中的 `[width,height]`，不是 dataset metadata。DataLoader 从原图按当前输入尺寸执行 letterbox 或 ROI resize，同步变换原图球心并生成热图；推理时使用同一变换逆映射。改变输入尺寸不需要重建数据集。评估、测速及导出读取 checkpoint 中的输入尺寸。

默认维持 Global 512×288、Local 256×256。可通过 `global_train_scales` / `local_train_scales` 列出多个输入尺寸；同一 micro-batch 采用相同尺寸，批次间随机选择，验证始终使用固定的 `global_input` / `local_input`。`configs/multiscale.json` 提供 Global 384×216、512×288、640×360 的示例，尚未启动或验证该配置的训练效果。输入宽、高必须不小于32且可被4整除，Local 输入为正方形；`local_crop_side_1080` 独立控制原图 ROI 范围（默认256 px），改变 Local 输出尺寸不会改变裁剪视野。

AdamW，lr 3e-4、weight decay 1e-4，AMP、有效 batch 8；因 GPU 正被其他任务使用，micro-batch 2、累积 4 次。前景/背景分别归一化 MSE，sigma=1.5 输出像素。水平翻转和轻量亮度增强。最多 80 epoch，12 epoch 验证指标未改善早停。Local 前 10 epoch 使用 ±20% ROI 边长的 GT 中心抖动，随后使用冻结 Global 的预测裁剪；切换后重置选优与早停计数。无有效 Global 裁剪的样本屏蔽 Local 损失，已知球在裁剪之外的样本监督为空 ROI。

Global best 按原图 50px 球心 F1；Local best 按组合输出 10px F1。阈值在固定 val 上从 0.01/0.02/0.05/0.1/0.2/0.3/0.5/0.7/0.9 中选优。每视频完整推理，未知帧不作为误报背景。框 AP 留空。

```bash
.venv/bin/python -m ttnet.pipeline
```

GPU 命令需要可访问 NVIDIA 设备的执行环境。管线依次完成小样本检查、Global、Global 评估/导出/测速、Local 小样本检查、Local、组合评估/导出/测速。所有产物存 `ttnet/runs/<Tokyo时间编号>/`。`run.json` 只有在评估、错误可视化、导出和测速齐全后才标记 completed。

独立执行与恢复：

```bash
.venv/bin/python -m ttnet.training.train --stage global --run ttnet/runs/NEW_GLOBAL_RUN
.venv/bin/python -m ttnet.training.train --stage global --run ttnet/runs/EXISTING_GLOBAL_RUN --resume
.venv/bin/python -m ttnet.training.train --stage local --run ttnet/runs/NEW_LOCAL_RUN --global-checkpoint ttnet/runs/GLOBAL_RUN/checkpoints/best.pt
.venv/bin/python -m ttnet.evaluation.evaluate --run ttnet/runs/LOCAL_RUN --global-checkpoint ttnet/runs/GLOBAL_RUN/checkpoints/best.pt --local-checkpoint ttnet/runs/LOCAL_RUN/checkpoints/best.pt --amp
.venv/bin/python -m ttnet.evaluation.export --run ttnet/runs/LOCAL_RUN --global-checkpoint ttnet/runs/GLOBAL_RUN/checkpoints/best.pt --local-checkpoint ttnet/runs/LOCAL_RUN/checkpoints/best.pt
.venv/bin/python -m ttnet.evaluation.benchmark --run ttnet/runs/LOCAL_RUN --global-checkpoint ttnet/runs/GLOBAL_RUN/checkpoints/best.pt --local-checkpoint ttnet/runs/LOCAL_RUN/checkpoints/best.pt --amp
```

`config.yaml` 使用 JSON 格式（YAML 1.2 的子集），无需引入 PyYAML。恢复校验配置与数据身份，保留 optimizer/scaler/RNG，并创建新的日志 attempt。ONNX 为 opset17，输出 heatmap 与128维 features，动态 batch；验证真实图像与 batch2 的 ORT CPU FP32 / PyTorch CPU FP32 数值对齐。目标赛事 TensorRT 尚需后续部署阶段单独验证。

当前 Global 已完成，Local 与自动管线按用户要求保持暂停。`--resume` 未指定 dataset 时沿用原 run 的数据版本，不会把旧 checkpoint 静默切到 v2；新运行默认使用 v2。旧版本固定512×288输入与运行时缩放的像素一致性已纳入测试。

## 未来事件接口

`FeatureWindowCache` 用模型训练过的 FPN 特征做 2×2 空间池化，每分支128维，拼为256维。没有未训练投影层。每帧保留定位结果与裁剪变换；缓存身份绑定数据集和两路 checkpoint 哈希。reset 清除视频状态。

固定9槽位，中心第5槽。按中心前后33.3ms选最近边界真实帧：120/60/30fps对应约9/5/3帧；25fps为3帧、跨度80ms。缺失槽填零并显式 frame_valid；无有效 Local 特征使用 local_valid。输入保留真实时间差，不通过重复帧伪造时间分辨率。高于120fps的输入对两侧各保留最多4个真实帧。

`window(center)` 返回 features `[1,9,256]`、frame_valid/local_valid/time_offsets_sec `[1,9]`、中心帧身份、定位和裁剪变换。第一阶段窗口为离线缓存接口；若后续用于流式事件推理，需要等待足够未来帧，不能把当前临时尾部窗口当作已确认完整窗口。

## 结果解释

完整错误图片和索引位于每个 run 的 `visualizations/`、`evaluations/best_val/errors.json`；预览页同时显示预测和标签。未知标签状态明确显示，不凭缺标认定误检。

测速分开报告同步前向 p50/p95（50 warmup，3×200）及原视频解码至坐标 JSON 序列化的 E2E FPS。PNG 输入评估吞吐另报，不能冒充原视频 E2E。GPU 可能共享，本机速度不等于赛事速度。历史基线来自保存记录，解码/精度/帧窗口不同，比较限制写入结果。
