# 桌框 pred / GT 检查（2026-10-18）

[打开完整图册](../../out/results/20261018_table_box_review/index.html) · [六组代表帧总览](../../out/results/20261018_table_box_review/overview.png)

绿色为人工桌框GT，红色为预测桌框。所有框均映射回原图坐标；点击图册图片可打开大图。TableHead没有质量置信度输出，图中不伪造置信度。

后续[label时序核查](../../docs/table_label_temporal_20261018.md)发现原始标签存在异常，例如04_004第115帧的table框落在地板上。因此绿色仅表示原始标注，并不保证标注正确；原IoU结果保留，错误案例须结合画面复核。

## 已输出

- 18组静态叠图：覆盖3个视角、6个场景组，每组选IoU的P10/P50/P90代表帧。每组有完整画面和局部放大，共36张图片。
- [侧视对照](../../out/results/20261018_table_box_review/sheets/side.png)、[斜视对照](../../out/results/20261018_table_box_review/sheets/diagonal.png)、[正视对照](../../out/results/20261018_table_box_review/sheets/front.png)。同一行从左至右为组内较差、中位、较好。
- 40个完整H.264叠加视频，6292帧，支持0.25×慢放；全部已重新解码并核对帧数。
- [所有IoU<0.5错误帧索引](../../out/results/20261018_table_box_review/errors.json)：3162帧，均由完整视频覆盖。
- [来源与样本明细](../../out/results/20261018_table_box_review/manifest.json)：坐标、IoU、中心偏移、来源哈希与验证记录。

本次复用桌框best checkpoint的已保存验证预测，没有重新推理或训练。带GT的5101帧使用原桌框评估的精确预测；其余1191帧只补充同桌框权重对应的scene_box完整轨迹预测，GT明确标为UNKNOWN，不参与IoU或错误统计。

## 建议先看

| 代表帧 | 画面上的主要现象 | 原图叠加 | 桌框放大 |
|---|---|---|---|
| 02_003，42帧，侧视 | 薄桌面预测过高，覆盖桌下区域 | [原图](../../out/results/20261018_table_box_review/images/02_median_02_003_000042.jpg) | [放大](../../out/results/20261018_table_box_review/zoom/02_median_02_003_000042.jpg) |
| 04_007，202帧，侧视 | 预测框过高，并有中心偏移 | [原图](../../out/results/20261018_table_box_review/images/04_median_04_007_000202.jpg) | [放大](../../out/results/20261018_table_box_review/zoom/04_median_04_007_000202.jpg) |
| 12_003，13帧，正视 | 预测明显向画面下方偏移 | [原图](../../out/results/20261018_table_box_review/images/12_median_12_003_000013.jpg) | [放大](../../out/results/20261018_table_box_review/zoom/12_median_12_003_000013.jpg) |
| 21_004，145帧，斜视 | 预测偏上且偏窄，遗漏较大桌面区域 | [原图](../../out/results/20261018_table_box_review/images/21_median_21_004_000145.jpg) | [放大](../../out/results/20261018_table_box_review/zoom/21_median_21_004_000145.jpg) |
| 22_002，47帧，侧视 | IoU相对较高，仍漏掉桌面的左侧部分 | [原图](../../out/results/20261018_table_box_review/images/22_median_22_002_000047.jpg) | [放大](../../out/results/20261018_table_box_review/zoom/22_median_22_002_000047.jpg) |

这些是各组中位IoU代表帧；图册另有较差与较好帧，避免只展示极端失败样本。组内较好不表示预测达标：所有有GT帧的平均IoU为0.387，最高约0.678，没有IoU≥0.75的帧。02组只有46帧有桌框GT，不能将该组样本外推为全部画面。

## 后续训练采用三阶段流程

1. 训练Global/Local定位、桌框与特征提取器，先解决上述几何偏差。
2. 冻结前端，提取预测轨迹、预测桌框以及可选桌面ROI特征，训练事件投影、融合层与TCN。
3. 从前两阶段权重开始，用少量训练集数据在线重算特征做端到端FT；保留定位/桌框辅助监督、较低的视觉编码器学习率和未FT版本作回退。

FT子集仅从固定训练集选取，覆盖三视角、正常背景与困难例；40视频验证集不变。具体子集大小与微调超参数尚未确定，本次不启动新训练。

实现与消融建议见[4090训练优化文档](../../docs/4090_training_optimization_20261017.md)。

用户后续明确先改善桌框定位：[定位改进方案与根因诊断](../../docs/table_localization_improvement_20261018.md)。新诊断包括训练/验证IoU差距、薄框覆盖和坐标往返检查；尚未启动新定位模型训练。
