# 独立特征编码器＋MLP与调色增强

2026-11-04用户要求尝试独立特征提取后接MLP，并加入调色增强。当前任务：[状态](../out/experiments/20261104-160636_independent_mlp_color/task.json) · [日志](../out/launches/20261104-160636_independent_mlp_color/driver.log) · [配置](../configs/table/independent_mlp_color.json) · [训练代码](../ttnet/src/scene/train_independent_mlp.py)。

## 模型与训练范围

单帧RGB 512×288 → 深度可分离CNN（16/32/64/96通道）→ FPN输出32×72×128空间图 → 16×4全图池化 → 展平2048 → Linear128＋SiLU → Linear4＋Sigmoid有序四角。

模型共305,295参数，其中编码器/FPN42,507、MLP262,788。编码器与MLP均从头初始化，全部接受桌框监督；不加载或更新B球定位器，不依赖YOLO或预训练权重。首个成功优化步将编码器和MLP的前后权重hash及梯度范数写入`first_update.json`。

保留32通道空间输出，可供后续ROI与上下文特征提取；checkpoint记录`native_depthwise_fpn_mlp_v1`结构身份与pool尺寸，独立前端加载器以及事件checkpoint重建路径按结构身份恢复。回归头每图输出一个框，候选score=1是存在性约定，不是校准的框质量分数。

## 数据增强

| 项目 | 范围/规则 |
| --- | --- |
| 水平翻转 | 50%，同步变换GT四角 |
| 平移 | 横纵各±5%原图尺寸 |
| 等比缩放 | 0.9–1.1，围绕图像中心 |
| 调色概率 | 80%，各操作随机执行顺序 |
| 亮度 / 对比度 | 各0.8–1.2倍 |
| 饱和度 | 0.75–1.25倍 |
| 色相 | ±0.03周，即约±10.8° |

几何变换作用在RGB和bbox上；调色在letterbox后的有效内容区执行，不改变bbox和外层letterbox黑边。重复样本的随机种子同时绑定seed、epoch、视频/帧和draw索引，因此同一帧重复曝光会获得不同的图像扰动。只在训练侧启用增强，验证保持原图变换。`augmentation_examples/`保存8个实际训练draw的原图/增强图与对应GT。

首轮发现边缘样本`08_004:1`的框紧贴底边，特定缩放/平移将其完全裁出图像；原失败任务`20261104-160259_independent_mlp_color`和日志保留。修复为：随机几何若使正框裁后宽或高不足1原图像素，回退到原几何位置，仍可调色；每轮记录`geometry_fallbacks`。使用新的当前任务从头重启，没有修改GT或复用不完整epoch权重。

## 采样、优化与评价

- 固定`pp_table_reannotated_v1`原114/40视频划分，unknown不参与训练；验证覆盖全部6,292帧，5,101个有效框评分。
- seed42，每轮4,096次曝光、薄框410次（10.01%）；薄框单帧最多4次、单视频最多123次薄框曝光，其余普通帧无放回采样。保存实际抽样清单、唯一帧/视频覆盖及分组统计。
- batch16，每轮256个完整优化批次，无末尾单样本。编码器与MLP均AdamW lr=3e-4、wd=1e-4（bias/norm不decay）；3轮warmup＋cosine；AMP，最多60轮，按视频均IoU早停patience10。
- 损失沿用最新MLP配方：L1＋GIoU＋0.25×相对纵向中心SmoothL1＋0.25×对数高度SmoothL1；归一化高度下限0.02。
- 主`best.pt`按验证视频均IoU选型，同时保存`best_video.pt`、`best_frame.pt`和`best_thin.pt`，以及完整last/optimizer/scaler/RNG和逐轮全量分桶指标。
- 最终对主best做train/val完整评分，保存逐帧预测、全错误索引与逐视频最差叠图；与旧独立密集CNN、冻结B＋4×4 MLP及冻结B＋薄框16×4 MLP比较。

本轮合并改变编码器、读出和增强/训练配方，评价的是组合实验，不能据此单独归因调色或结构收益。原训练薄框仍集中在源组13，调色与增采不会制造新的真实场景。本次不追加单元测试，不启动事件头训练。

恢复同一任务：

```bash
.venv/bin/python -u -m ttnet.scene.train_independent_mlp \
  --config configs/table/independent_mlp_color.json \
  --run out/experiments/20261104-160636_independent_mlp_color \
  --device cuda --resume
```

## 完成结果

2026-11-04 16:15完成。第11轮触发早停，主best及帧均/薄框best均为第1轮；已完成主best的全量train/val评分，并保存31个包含IoU<0.5错误的验证视频的最差叠图，其余视频仍完整评分。主best训练帧均IoU64.29%、视频均60.78%，验证如下：

| 路线 | 验证帧均IoU | 验证视频均IoU | 薄框IoU | 验证IoU≥0.5比例 |
| --- | ---: | ---: | ---: | ---: |
| 原独立CNN＋中心热图/密集框 | 19.29% | 15.05% | 2.40% | 3.33% |
| 冻结B＋4×4 MLP | 46.37% | 36.18% | 6.36% | 56.85% |
| 冻结B＋薄框16×4 MLP | 48.25% | 37.19% | 23.53% | 45.62% |
| **独立编码器＋MLP＋调色** | **41.86%** | **42.89%** | **43.24%** | **18.11%** |

薄框IoU≥0.5比例为13.04%（82/629），此前两只B＋MLP均为0；主best验证IoU≥0.75比例为0。新路线提高了视频均与薄框均IoU，但帧均和总体过0.5比例下降，不能直接认定优于全部旧路线。

| 验证视角/源组 | 帧数 | 新路线均IoU | 说明 |
| --- | ---: | ---: | --- |
| 正视/组12 | 628 | 58.51% | 高于B薄框组合50.85%，低于B原MLP66.34% |
| 斜视 | 2,118 | 40.99% | 组21升至46.17%，但组03仅29.40% |
| 侧视 | 2,355 | 38.20% | 薄框改善，组22却降至36.36%，该组仍无IoU≥0.5 |
| 薄框 | 629 | 43.24% | 组02为36.03%，组04为43.81% |

早期验证仍有明显波动；后续训练loss降低但视频均未超过首轮，说明独立特征与调色并未消除泛化/优化问题。当前保留多路线权重和全部逐轮指标，不替换事件前端，不追加训练。

产物：[主best](../out/experiments/20261104-160636_independent_mlp_color/best.pt) · [完整评分](../out/experiments/20261104-160636_independent_mlp_color/metrics.json) · [机器对照报告](../out/experiments/20261104-160636_independent_mlp_color/comparison.md) · [逐帧预测](../out/experiments/20261104-160636_independent_mlp_color/evaluation/predictions.jsonl) · [叠图索引](../out/experiments/20261104-160636_independent_mlp_color/evaluation/index.html) · [真实增强样例清单](../out/experiments/20261104-160636_independent_mlp_color/augmentation_examples/manifest.json)。
