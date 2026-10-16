# 四组事件对照：运行与标注接口

本轮代码使用现有 `.venv` 与工作区数组库。人工标注进行中时可以运行候选诊断、自动测试和真实样本检查；A/B/C/D 的正式训练会拒绝缺失或未复核的 PP 标注。

## 配置与检查

```bash
.venv/bin/python -m ttnet.cli list
.venv/bin/python -m ttnet.cli train --config configs/experiments/event_comparison_D.json --dry-run
.venv/bin/python -m unittest discover -s ttnet/src/tests
.venv/bin/python -m ttnet.events.verify_comparison \
  --historical ttnet/output/events/20261014-181231_event25_frozen/checkpoints/best.pt \
  --output out/results/new_code_check/real_samples.json
```

历史机器的 `ttnet/output` 是兼容链接；跨机器应显式覆盖 `global_checkpoint`、`local_checkpoint`。不会移动或覆盖历史 A0。

配置可覆盖 `tracker.type`、`event_head.type`、`association.beam_size`、`association.motion_weight`、`transformer.layers` 等已声明参数。B/D 限制为 frozen；joint 不参与本轮。

## 三帧候选诊断

```bash
.venv/bin/python -m ttnet.evaluation.candidate_audit \
  --config configs/experiments/event_comparison_A.json \
  --device cpu --output out/results/new_fixed3_candidates
```

默认完整推理固定 PP val 全部视频。`--video 02_001` 可做单视频检查，结果状态明确标为子集。目录包含 `summary.json`、`per_video.json`、`candidate_samples.jsonl`、`error_index.jsonl`、逐视频预测与完整叠加 MP4、`index.html`。不接受旧单帧模型，不以 GT 决定候选选择。

候选统计表在置信度 0.7/0.1/0.01 下分别给出 Top-1/3/5 的 10/50px 命中及候选 ROI 覆盖。`localization` 是实际当前路径结果，候选覆盖是 oracle 上限，二者不能互换。诊断计时包含画图和编码，不作为 50 FPS 推理验收。

优先判断多候选轨迹关联的定位收益，可直接运行配对入口：

```bash
.venv/bin/python -m ttnet.evaluation.trajectory_comparison \
  --config-a configs/experiments/event_comparison_A.json \
  --config-b configs/experiments/event_comparison_B.json \
  --annotations data/annotations/pp_bounce_confirmed_v1.json \
  --output out/results/new_trajectory_AB --device cuda
```

同一固定40视频完整推理，输出10/50px球心F1、任一/选中Local ROI覆盖率、连续定位漏检及无观测长度、触台±1帧定位、视角/分辨率统计。错误跳转定义为相邻有效正标签帧的预测位移与GT位移偏差超过50原图像素，真实时间间隔不大于1.5帧；不将真实快速运动直接算作错误跳转。`paired_changes.jsonl` 区分恢复/退化帧，A/B分别提供覆盖全部评分错误的完整视频。`runtime.json` 另跑相同硬件与后端的全部原视频解码至JSON；事件头相同且未训练，定位诊断不能替代最终事件选型。

## PP 连续标注

如已有人工标注清单，直接转换到以下结构，不重新挑选回合。需要空白模板时：

```bash
.venv/bin/python -m ttnet.events.continuous template \
  --annotations data/annotations/pp_continuous_v1.json
```

模板确定性地从既有 train/val 各选 12/6 个完整视频，优先分散比赛组；`view` 留空、状态待复核。它不会猜测视角或触台标签，不能代替正在进行的人工标注。若目标文件存在，命令拒绝覆盖。

顶层字段：`schema: 1`、`version`、`ball_dataset_id`、`ball_split_hashes`、`videos`。哈希由模板工具填写。每个视频条目示例（视频/哈希须使用真实值）：

```json
{
  "video_id": "00_001",
  "split": "train",
  "match_group": "00",
  "view": "view_1",
  "source": "data/pp_train_data/rallies_videos/00/001.mp4",
  "source_sha256": "由模板保留的实际 SHA256",
  "status": "reviewed",
  "reviewed_by": "人工复核者",
  "net_reviewed": false,
  "reviewed_intervals": [[0, 59]],
  "unknown_intervals": [[25, 30]],
  "events": [{"kind": "bounce", "frame_id": 12, "x": 450.0, "y": 280.0}],
  "review_tags": ["racket", "floor", "occlusion", "video_boundary"]
}
```

帧号从 0 开始，是原视频帧号；区间两端都包含。坐标是原图像素，不是 Global/Local 输入坐标。`reviewed_intervals` 内未列出事件的时刻才是确定背景；unknown 覆盖时仍为未知。必须逐帧覆盖完整回合：每帧至少属于 reviewed 或 unknown。confirmed event 不得落在 unknown。未核验 Net 时不允许填写 Net 事件，Net loss 与分类统计均屏蔽。视角名称由人工保持一致，缺视角会报告实际宏平均范围。

```bash
.venv/bin/python -m ttnet.events.continuous validate \
  --annotations data/annotations/pp_continuous_v1.json
```

检查器核对原视频内容哈希、固定球定位数据身份、划分文件哈希、视频归属、比赛组、视角、人工复核状态、区间和坐标。训练记录保存整个标注文档身份与采样身份；修改标签后必须新建实验，旧 checkpoint 评估会拒绝身份不一致。

## 正式训练

标注文件完成复核后依次执行四组，示例：

```bash
.venv/bin/python -m ttnet.cli train --config configs/experiments/event_comparison_A.json --device cuda
.venv/bin/python -m ttnet.cli train --config configs/experiments/event_comparison_B.json --device cuda
.venv/bin/python -m ttnet.cli train --config configs/experiments/event_comparison_C.json --device cuda
.venv/bin/python -m ttnet.cli train --config configs/experiments/event_comparison_D.json --device cuda
```

`--set pp_annotations=实际路径` 可使用其他标注版本；四组必须相同。`--set seed=43` / `44` 用于后续 A 与最佳新方案复验。旧训练命令和旧 checkpoint 仍使用旧协议，不会自动获得新监督。

B/D 第一次缓存会从原始 OpenTTGames 视频流式补齐连续上下文，已有稀疏缓存像素优先读取，不生成第二份原帧 NPY。只保存目标帧的特征到 `data/cache/features/`。A/C 和 B/D 各自共享特征，轨迹参数或实现变更自动换缓存身份。每视频从第 0 帧开始，事件窗口只读取缓存，不重置关联器。

训练每个 optimizer 边界定期保存 optimizer、scaler、RNG、epoch/step 与累计 loss，可用 `ttnet.cli train --run 实验目录 --resume` 续训。PP 主指标按 F1、负 FP 排序；连续验证恶化会触发 patience，稀疏 BCE 仅作单独诊断。

## 连续验证与 A0

新模型：

```bash
.venv/bin/python -m ttnet.cli evaluate --run out/experiments/实际实验ID --device cuda
```

该入口采用 checkpoint 保存的 PP 阈值，输出至该实验的 `evaluations/continuous/`。直接入口也可用于 A0，默认只在 PP val 选择历史模型的 Bounce 阈值：

```bash
.venv/bin/python -m ttnet.events.comparison evaluate --scheme A0 \
  --checkpoint ttnet/output/events/20261014-181231_event25_frozen/checkpoints/best.pt \
  --annotations data/annotations/pp_continuous_v1.json \
  --output out/results/new_A0_continuous --device cuda
```

PP 连续评估保留原始帧号和 PTS，允许非等间隔时间戳；不会用容器平均 FPS 重写事件位置。结果包括逐视角/分辨率连续触台与落点、球心 F1/ROI 覆盖/连续漏检/错误跳转/触台邻域误差、原视频完整推理 FPS、峰值显存与未来帧等待说明，以及所有评分错误的视频索引和叠加视频。预测缺少球坐标时仍计入连续触台检测，不能伪造成成功落点。

## 最终泛化与汇总

四组及种子选型结束后才运行独立测试与 PP public，保留 checkpoint 阈值，不扫描外部标签：

```bash
.venv/bin/python -m ttnet.events.evaluate --run out/experiments/实际实验ID --device cuda
.venv/bin/python -m ttnet.events.comparison report \
  --results out/experiments/实际A/evaluations/continuous/metrics.json \
            out/experiments/实际B/evaluations/continuous/metrics.json \
            out/experiments/实际C/evaluations/continuous/metrics.json \
            out/experiments/实际D/evaluations/continuous/metrics.json \
  --output out/results/new_comparison
```

报告输出 `comparison.csv`、`comparison.md`、`selection.json`，拒绝混用不同 PP 标注或后处理。按实际硬件/后端/精度判断测速可比性；缺四组或多种子结果会保持待验证状态。最终需人工结合逐视角退化、误报、速度与波动确认选型；未获得稳定收益前保留 A。

## 本轮代码验证记录

- 全套 73 项自动测试通过，包括补充的真实缺帧 mask 与非等间隔原视频 PTS 测试。
- CPU 和 RTX 5070 Ti 上，四组真实视频帧前向/反向、checkpoint 恢复均通过；A 与历史 A0 的视觉特征、观测及 logits 逐值一致。
- 固定 PP val 40 视频、6,292 帧完成候选诊断，5,302 个正标签，808 个评分错误；逐个核对完整 MP4 帧数和错误索引覆盖。
- Global 阈值 0.7 时，Top-1/3/5 的 10px 覆盖为 84.01%/89.72%/90.44%，50px 为 85.14%/91.36%/92.19%，ROI 上限为 87.21%/93.68%/94.47%。Top-5 的 10px oracle 可恢复 341/848 个 Top-1 漏命中（40.21%），不是实际关联提升。
- GPU 两视频测速 A/B/C/D 为 98.31/89.99/90.63/86.46 FPS；事件头未正式训练，正式四组准确率及多种子选型仍为空。

产物：`out/results/20261015_fixed3_candidates_full/`、`out/results/20261015_comparison_code_checks/`、`out/results/20261015_comparison_cuda_runtime_native/`。
