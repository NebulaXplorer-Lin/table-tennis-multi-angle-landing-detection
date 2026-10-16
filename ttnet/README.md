当前使用方法以 [工作区说明](../docs/workspace.md) 为准；配置入口在根目录 `configs/`，当前数据位于 `data/datasets/`，新实验位于 `out/experiments/`。下文保留历史接口和模型说明；旧目录通过归档兼容链接保留。

# TTNet 第一阶段：球定位

独立的数据与训练工程。原始视频只读；不覆盖 YOLO 数据或结果，不修改赛事框架和 `Solution`。第二阶段已增加视觉与轨迹融合 EventHead，提供冻结定位和联合微调两种训练路径，见 [第二阶段说明](../docs/eventhead_training.md)。

查看最终结果请直接打开 [output/README.md](output/README.md)，从任务汇总进入权重、eval、错误图册、Recall / F1、延迟与 FPS；[RESULTS.md](RESULTS.md) 是训练进度和最终结果的统一入口。

## 目录与最终交付

```text
ttnet/
  src/                         # 模型、训练、评估、数据工具、测试
    configs/                   # 默认配置、实验配置与固定划分来源清单
  data/
    datasets/ball_v1, ball_v2/  # 原图、标签、窗口、metadata、splits
    cache/                     # 可重建的缩放帧与裁剪缓存
  runs/<run_id>/               # 各次启动/续训记录、日志、工作权重、中间产物
  output/
    README.md                  # 所有最终结果和完整训练任务的索引
    latest.json                # 最新完整 Global / Local 结果编号
    comparison.csv             # Precision / Recall / F1 / FPS 对比
    <run_id>/
      summary.md               # 一次训练的最终报告，直接链接下面各项
      weights/                 # best、last；Local 还包含配套 global_best.pt
      eval/                    # 完整评估、逐视频指标、错误索引/图册、测速原始数据
      error_samples/           # pred / label 叠加图与已有慢放片段
      predictions/             # 逐帧预测
      exports/                 # ONNX 与数值对齐记录
      feature_cache/           # 特征、定位和接口身份记录
      stats/                   # summary.json 总统计、training.json 每轮记录
      config.yaml, run.json, environment.json, manifest.json
    <pipeline_id>/summary.md   # 一次完整任务的 Global / Local 总入口
    diagnostics/               # 独立诊断分析
    .history/                  # 同一任务后续更新前的结果快照
```

代码的实际目录是 `src/`，公开模块名仍为 `ttnet`：所有 `python -m ttnet...` 命令继续有效，从项目根目录执行。根目录 `datasets`、`cache` 仅为旧路径的兼容符号链接；新任务实际使用 `data/`，保留兼容入口以支持历史 config 与 checkpoint。

训练结束、评估、导出、测速、错误检查和管线结束后自动刷新结果。运行中的任务与 smoke 不发布为最终任务；缺项、暂停或失败任务保存为 `partial`，未测指标留空。已完成结果复制到 output，浏览报告和使用权重无需依赖 runs；Local 组合定位所需的两路权重均齐全。更新会保留旧快照，不覆盖其他任务。

已有历史结果或新增离线统计可以直接整理，不会重新训练或推理：

```bash
.venv/bin/python -m ttnet.results
.venv/bin/python -m ttnet.results --run ttnet/runs/<run_id>
```

`manifest.json` 记录来源 run、结果状态、权重 SHA256、文件清单和缺项；`stats/summary.json` 汇总 Precision / Recall / F1、TP/FP/FN、误差、ROI 覆盖率、错误分类、E2E FPS、前向延迟、参数/FLOPs、显存和硬件。轻量代码与配置纳入版本管理，数据、缓存和 output 大产物不纳入 Git。

错误检查可通过 `python -m ttnet.evaluation.review_errors --run runs/<run_id>` 生成（从项目根目录执行时使用 `--run ttnet/runs/<run_id>`）。输出 `evaluations/best_val/errors.html`，按视频、错误类型、GT 是否进入 ROI 及最终来源分支筛选全部错误；另保存四类代表片段的 GT / Pred / ROI / Global 慢放 MP4。读取本次评估的已保存预测和原始帧，不重新推理。

## 数据

当前数据集为 `data/datasets/ball_v2`，覆盖全部 154 视频的 23,690 帧：20,846 帧有有效球框，2,844 帧缺标且标记 unknown。固定视频划分 114 train / 40 val，沿用用户批准的历史场景分组；与当前 SOP 的逐文件夹 80/20 默认不同，不重新划分。不能据此声称独立比赛场次测试成绩。

数据集只保存原始帧与原始标签信息。`labels/<video_id>.jsonl` 保留每一帧的 fps、fps 精确分数、基于 PTS 的名义帧率、原图分辨率、球框/球心、监督状态与原始其他类别标注；不保存模型输入尺寸、固定缩放变换或 Global 输入图像路径。坐标使用原图像素，时间戳相对首帧。missing ball ≠ negative。

v2 的 `frames/` 通过相对符号链接复用 v1 的原始 PNG，不重复解码或复制图片；v2 不包含 `global_inputs/`。v1 作为历史数据版本保留，与已完成或暂停的训练权重对应。共享原图不可修改，也不能只删除 v1 的原图目录。

`metadata/` 保存数据身份、每视频来源哈希、实际解码帧数、PTS 审计和处理版本。`splits/` 是 TTNet 自己的固定视频清单。轻量来源清单另存 `src/configs/frozen_split.json`，正常训练不再依赖 YOLO 文件。

处理可按视频恢复；已完成视频会验证现有 PNG。来源、图像处理或采样策略变化时必须更换数据版本，不能覆盖现有身份。

```bash
.venv/bin/python -m ttnet.data_tools.prepare --pilot 4
.venv/bin/python -m ttnet.data_tools.prepare --workers 4
.venv/bin/python -m ttnet.data_tools.migrate_raw  # 已有 v1 时生成 v2，复用原图
.venv/bin/python -m ttnet.data_tools.audit --full-images
.venv/bin/python -m unittest discover -s ttnet/src/tests -v
```

## 模型和训练

两路独立的残差编码器（16/32/64/96，GroupNorm）+ 多尺度 FPN + 单通道 sigmoid 热图。Global 输入宽×高 512×288，输出 256×144；Local 输入 256×256，输出 128×128。1080p 的 ROI 边长 256 px，其他分辨率按高度缩放，越界部分补黑，裁剪中心不移动。

当前默认模型为前后多帧定位：逐帧共享同一编码器参数，在四层空间特征上分别融合有序帧特征、frame_valid 和实际 time_offsets，预测中心帧热图。每层用1×1投影和残差卷积完成时序空间融合，后续FPN和热图头保持原结构。三帧模式每分支553,985参数，9槽位模式644,033参数；历史单帧每分支245,057参数仍可按旧checkpoint配置加载。

| temporal_mode | 图像输入 | 中心槽 | 后续特征窗口 | 当前状态 |
|---|---|---:|---|---|
| fixed3 | `[B,3,3,H,W]`，所有fps取原生前一帧、当前帧、后一帧 | 2 | `[B,3,256]` | 优先训练 |
| fps_mask9 | `[B,9,3,H,W]`，以中心前后33.3ms取真实帧，缺失槽mask | 5 | `[B,9,256]` | 已接通配置，暂不训练 |
| single | `[B,3,H,W]` | 当前帧 | 历史接口 | 兼容历史权重 |

固定三帧的首尾跨度随帧率变化：25/30/60/120fps分别约80/66.7/33.3/16.7ms；9槽位方案保持约66.7ms，25fps最接近的真实帧跨度约80ms。两种方案都监督中心帧，视频首尾填零并mask，unknown标签帧仍可作为真实视觉上下文。增强在整个窗口内使用同一翻转与亮度参数。Local一个窗口用同一原图ROI，不能逐帧用各自GT球心居中，以免消除运动线索。

`src/configs/fixed3.json` 是本次训练配置；`src/configs/fps_mask9.json` 是保留的对照开关。本轮固定空间输入尺寸，不同时启用空间多尺度训练。

最强局部峰附近 7×7 加权定位，不混合远处候选。Global 负责逐帧重捕获；Local 不可靠时返回有效 Global；两者均无效时返回缺失。

`global_input` / `local_input` 是训练配置中的 `[width,height]`，不是 dataset metadata。DataLoader 从原图按当前输入尺寸执行 letterbox 或 ROI resize，同步变换原图球心并生成热图；推理时使用同一变换逆映射。改变输入尺寸不需要重建数据集。评估、测速及导出读取 checkpoint 中的输入尺寸。

默认维持 Global 512×288、Local 256×256。可通过 `global_train_scales` / `local_train_scales` 列出多个输入尺寸；同一 micro-batch 采用相同尺寸，批次间随机选择，验证始终使用固定的 `global_input` / `local_input`。`src/configs/multiscale.json` 提供 Global 384×216、512×288、640×360 的示例，尚未启动或验证该配置的训练效果。输入宽、高必须不小于32且可被4整除，Local 输入为正方形；`local_crop_side_1080` 独立控制原图 ROI 范围（默认256 px），改变 Local 输出尺寸不会改变裁剪视野。

AdamW，lr 3e-4、weight decay 1e-4，AMP、有效 batch 8；micro-batch 2、累积 4 次。前景/背景分别归一化 MSE，sigma=1.5 输出像素。水平翻转和轻量亮度增强。当前每阶段最多 20 epoch；平均训练 loss 和验证 F1 分别跟踪显著改善，loss 相对下降超过 1%、F1 增加超过 0.002 才重置对应计数。两项均连续 5 epoch 没有显著改善时早停；这是训练 loss 的辅助判断，best 仍按验证 F1 与误差选取。Local 前 10 epoch 使用 ±20% ROI 边长的 GT 中心抖动，随后使用冻结 Global 的预测裁剪；切换后重置选优与两项早停计数。无有效 Global 裁剪的样本屏蔽 Local 损失，已知球在裁剪之外的样本监督为空 ROI。

Global best 按原图 50px 球心 F1；Local best 按组合输出 10px F1。阈值在固定 val 上从 0.01/0.02/0.05/0.1/0.2/0.3/0.5/0.7/0.9 中选优。每视频完整推理，未知帧不作为误报背景。框 AP 留空。

```bash
.venv/bin/python -m ttnet.pipeline
```

GPU 命令需要可访问 NVIDIA 设备的执行环境。管线依次完成小样本检查、Global、Global 评估/导出/测速、Local 小样本检查、Local、组合评估/导出/测速。启动记录与中间产物存 `ttnet/runs/<Tokyo时间编号>/`，最终结果自动整理到 `ttnet/output/<同一编号>/`，完整训练任务另有 pipeline 汇总入口。`run.json` 只有在评估、错误可视化、导出和测速齐全后才标记 completed。

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

用户调整停止策略后，使用 `--resume --config ttnet/src/configs/fixed3.json --allow-stopping-change` 恢复；只允许改变 `epochs`、`patience`、`early_stopping`，旧值与新值写入本次 attempt 的 `stopping_change.json`。旧 checkpoint 没有联合早停状态时，从已完成的 epoch 日志重建计数；新 checkpoint 直接保存联合状态。

历史单帧 Global 已完成，历史单帧 Local 保持暂停；当前新建三帧实验按用户要求优先训练，不恢复旧实验。`--resume` 未指定 dataset 时沿用原 run 的数据版本，不会把旧 checkpoint 静默切到 v2；新运行默认使用 v2。旧版本固定512×288输入与运行时缩放的像素一致性已纳入测试。

## 未来事件接口

`FeatureWindowCache` 用模型训练过的 FPN 特征做 2×2 空间池化，每分支128维，拼为256维。没有未训练投影层。每帧保留定位结果与裁剪变换；缓存身份绑定数据集和两路 checkpoint 哈希。reset 清除视频状态。

事件特征窗口与定位模式一致：fixed3为3槽位、中心第2槽；fps_mask9为9槽位、中心第5槽，按实际时间选帧。缺失槽填零并显式 frame_valid，无有效Local特征使用 local_valid。输入保留真实时间差，不通过重复帧伪造时间分辨率。每个128维描述符已经包含自身定位窗口的上下文，因此后续把多个描述符组成事件窗口时，整体视觉感受时间范围会比单个定位窗口更宽。

`window(center)` 返回 features `[1,T,256]`、frame_valid/local_valid/time_offsets_sec `[1,T]`（T=3或9）、中心帧身份、定位和裁剪变换。第一阶段窗口为离线缓存接口；若后续用于流式事件推理，需要等待足够未来帧，不能把当前临时尾部窗口当作已确认完整窗口。

## 结果解释

最终错误图片和索引位于每个 output 任务的 `error_samples/`、`eval/best_val/errors.json`；预览页同时显示预测和标签。未知标签状态明确显示，不凭缺标认定误检。

测速分开报告同步前向 p50/p95（50 warmup，3×200）及原视频解码至坐标 JSON 序列化的 E2E FPS。多帧测速使用逐视频离线缓冲，包含解码、预处理及窗口组装；尚非赛事流式部署。PNG 输入评估吞吐另报，不能冒充原视频 E2E。GPU 可能共享，本机速度不等于赛事速度。历史基线来自保存记录，解码/精度/帧窗口不同，比较限制写入结果。
