# 完整版本与实验结果

更新：2026-10-20。组员先看[Quick intro](../docs/quick_intro.md)，训练见[指南](../docs/training_guide.md)。

后续开发以三种子完整的TCN主基线为参照。EMA是单种子候选，内部只少1个FP，两套外测与主基线相同；保留两版，不把小幅单种子变化解释为稳定收益。

| 版本 | 内部seed42宏F1 | 三种子均值±std | 完整视频FPS | 作用 |
| --- | ---: | ---: | ---: | --- |
| [TCN+box+时间编码](tcn_frames_lr0.0003_d0.1/README.md) | 79.56% | 78.77% ± 2.30pp | 77.53 | 当前开发主基线 |
| [TCN+桌框窗口EMA](20261017-112200_tcn_fast_refine/README.md) | 79.69% | 仅seed42 | 86.88 | 单种子候选，新增best.pt与完整metrics快照 |
| [baseline_B](baseline_B/README.md) | 77.96% | 76.41% ± 1.11pp | 79.75 | 原轨迹关联基线 |
| [scene_box](scene_box/README.md) | 78.73% | — | 见版本metrics | 桌框几何消融与桌框错误复核 |
| [Transformer](transformer_frames_lr0.0001_d0.1/README.md) | 75.03% | 75.72% ± 1.80pp | 79.83 | 完整受控对照 |

内部验证为固定PP 40视频、原帧±1、原图50px、一对一匹配、三视角宏平均。TCN与EMA的PP public落点宏F1均为70.97%，OpenTTGames test稀疏锚点分类宏F1均为83.53%；两种任务不能混作同一成绩。

历史速度来自RTX 5070 Ti、PyTorch AMP，包含原视频解码、定位、关联、事件头与JSON序列化，排除初始化/磁盘写入。不同轮次测量不用于推断EMA加速，也不代表4090比赛实测。事件窗需6帧未来上下文，定位器另需1帧。

## 已完成实验给出的信息

- 桌框box几何在当前对照中小幅正收益；relative模式未更好。旧桌框验证平均IoU仅38.75%，且有薄框泛化与错标问题，不能据此判定桌框精度不是瓶颈。[桌框复核](scene_box/TABLE_REVIEW.md)与[定位改进诊断](../docs/table_localization_improvement_20261018.md)用于下一步。
- 当前AR beam1/4/8约66%，未优于规则关联；后续null修复也未通过预设门槛，未继续构建轨迹缓存或训练事件头。
- 困难背景采样候选宏F1为74.17%，退化；TCN在同前端、同时间编码对照及三种子汇总中优于当前Transformer。
- 全随机联合训练已停止，没有完成验证，已移出结果首页和日常实验目录。

本目录保留权重、配置、训练曲线、评估、种子与完整pred/GT错误视频的入口。权重和视频不随Git传输。原始历史run.json可能停留在训练结束状态，完成证据以metrics、评估receipt和[整理后索引](../experiments/index.md)为准。

完整历史分析：[A/B/C/D与定位对照](../docs/experiment_comparison_20261016.md)、[桌框/AR/事件头实现](../docs/table_ar_experiments_20261016.md)。后续以[三阶段路线](../docs/next_steps.md)为准。
