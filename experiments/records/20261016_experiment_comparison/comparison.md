# 乒乓球定位与落点检测：已完成实验对比

截至 2026-10-16T17:10:25+08:00（北京时间）。汇总已有实际产物，本次不启动新训练、不重新选择阈值。

当前结论：本轮同一 PP 内部验证集上，B（多候选轨迹关联＋TCN）最好，落点三视角宏平均 F1 为 **77.96%**，完整视频 **86.54 FPS**。这是一轮 seed42 验证成绩；多种子与当前四组的外部泛化评估尚未完成。

历史官方 ONNX 定位器在 PP 相同验证视频上的10px球心F1为97.76%，仍高于自训练定位器；其训练来源未知，不能据此证明独立泛化。球定位、稀疏事件分类、连续触台和落点成绩分别列出。

机器可读汇总：[comparison.csv](</home/radno/projects/pingpong/experiments/records/20261016_experiment_comparison/comparison.csv>)；数据清单与完整身份：[datasets.json](</home/radno/projects/pingpong/experiments/records/20261016_experiment_comparison/datasets.json>)。表中百分数由0～1原始值格式化，空白/“—”表示未测，不表示0。

## 1. 数据集与划分

| 编号 | 来源与版本 | 训练部分 | 验证/测试部分 | 标注与用途 |
|---|---|---|---|---|
| D1 | PP 球定位；pp_ball_v1/v2；同一固定视频划分 | 114视频；17398帧；15544有球GT | val 40视频；6292帧；5302有球GT、990 unknown | 球心/框；无可靠无球负类，unknown不评分 |
| D2 | PP 新增连续事件；pp_bounce_confirmed_v1 | D1的114视频；613触台；600有空间参考 | D1的40视频；203触台；189有空间参考 | 按用户确认触台已标全；ignore/空导出保持unknown；Net屏蔽 |
| D3 | OpenTTGames event25_v3 | game_1–4；4场；3352原始anchor | val game_5：450 anchor；test_1–7：464 anchor | 25 FPS，13帧窗口；Bounce/Net/empty；稀疏分类 |
| D4 | PP public：pingpang_landing_v1_public | 未加入本轮事件训练 | 15视频；2109帧；92有效落点、12忽略标签 | 0°/45°/90°各5视频；历史迁移/规则落点评估 |
| D5 | 历史YOLO全量训练版本 | PP全部154视频，20846有球标注帧 | 评估用D4；没有按D1留出内部val训练 | 与D1内部验证训练边界不同，不能并入同一公平训练排名 |

D1本地训练与留出评估复用固定114/40清单；本轮未重分。官方权重仅在val评估，训练来源未知。v1/v2的数据身份不同，但train/val清单哈希一致。早期YOLO640只从训练部分按每5帧采样，实际3162张，val为5302张有球帧。

D2的空间参考是**触台事件帧的原球心GT**，不是新增人工精标接触坐标；13个训练事件、14个验证事件缺球心，仍用于时间监督/评分，空间评分排除。`02_004`空标注导出整段保持unknown；显式ignore优先。front/diagonal/side为粗视觉分组，不等同D4精确角度名称。

| PP集合 | front视频 | diagonal视频 | side视频 | 触台事件 | 空间可评分事件 |
|---|---|---|---|---|---|
| train | 81 | 18 | 15 | 613 | 600 |
| val | 8 | 14 | 18 | 203 | 189 |

D3原始anchor计数：训练Bounce1289 / Net986 / empty1077；验证248 / 184 / 18；测试240 / 180 / 44。位移软标签扩展后训练窗口12452、验证2178、测试2144；扩展窗口不是独立事件数量。每个epoch平衡/混合采样，当前四组实际使用7734个窗口，两个来源各50%。

数据入口：[PP train清单](</home/radno/projects/pingpong/data/datasets/pp_ball_v2/splits/train.jsonl>)；[PP val清单](</home/radno/projects/pingpong/data/datasets/pp_ball_v2/splits/val.jsonl>)；[连续触台标签](</home/radno/projects/pingpong/data/annotations/pp_bounce_confirmed_v1.json>)；[TTGames场次划分](</home/radno/projects/pingpong/data/datasets/openttgames_event25_v3/split.json>)；[public清单](</home/radno/projects/pingpong/data/pp_public_data/manifest.json>)。

**public的历史使用情况：**它已被历史YOLO、冻结TCN和联合训练中间评估使用，也已据此分析过错误。本轮A/B/C/D尚未评估public，且本轮不使用其标签训练或选阈值；但不能把它描述为整个项目从未查看过的盲测集。

## 2. 球定位：历史模型与三帧定位器

以下精度均来自D1固定40个验证视频的已知有球标签；未知帧不当作无球。球心阈值为原图10px/50px。官方预训练来源未知；不同模型解码、边界处理与后端不同，表格用于历史对照。

| 方案 | 训练数据 | 完成轮/最佳轮 | 球心F1@10px | 球心F1@50px | 说明 |
|---|---|---|---|---|---|
| YOLO11n 640，conf .25 | COCO＋D1采样3162帧 | 6 / 1 | 0.00% | — | 严重漏检，主评估无有效框 |
| YOLO11n 640，conf .001事后诊断 | 同一权重 | 同上 | 2.35% | — | 只69个输出/63个10px TP，非新训练 |
| 官方三帧ONNX | 来源未知；本地未重训 | — | 97.76% | 98.03% | 原始官方解码，非本轮定位骨干 |
| 自训练单帧Global | D1 train114视频 | 18 / 6 | 44.83% | 46.17% | 固定工作阈值0.7 |
| 自训练三帧Global | D1 train114视频 | 20 / 20 | 84.34% | 85.48% | 固定工作阈值0.7 |
| 自训练三帧Global＋Local | D1 train114视频 | 20 / 13 | 85.08% | 86.06% | 固定工作阈值0.7 |

YOLO640框架记录AP50=20.85%、AP50–95=6.69%，但默认阈值球心召回为0；事后本地101点AP诊断为0.99%/0.43%，协议不同。原报告已指出低召回下的库PR尾部插值问题，这些AP数值不能替代实际球心命中率。热图模型没有bbox，AP留空。

三帧Global到Local的10px F1提升约0.74个百分点、50px提升约0.59个百分点；增加Local只带来有限总体提升，候选与搜索区域错误仍是重要来源。

来源：[YOLO首轮与AP审计](</home/radno/projects/pingpong/experiments/records/20261013-163117_yolo11n_ball-640/summary.md>)；[官方定位完整报告](</home/radno/projects/pingpong/experiments/records/20261013-212318_baseline_ball-position/summary.md>)；[单帧Global](</home/radno/projects/pingpong/ttnet/output/20261013-234947_global_ball/summary.md>)；[三帧Global](</home/radno/projects/pingpong/ttnet/output/20261014-095202_global_fixed3_ball/summary.md>)；[三帧Global/Local](</home/radno/projects/pingpong/ttnet/output/20261014-103011_local_fixed3_ball/summary.md>)。

## 3. 多候选轨迹关联：不训练事件头的A/B定位对照

D1同40视频、6292帧，固定同一Global/Local权重、CUDA AMP和工作阈值。B是规则关联器：Global Top5、最多8条假设、短历史代价、最多2个Local ROI；组件本身不额外训练。GT只用于诊断，不用于选候选。

| 指标 | A 独立解码 | B 多候选关联 |
|---|---|---|
| 球心F1@10px | 85.08% | 87.79% |
| 球心F1@50px | 86.06% | 89.41% |
| 任一Local ROI覆盖 | 87.21% | 93.93% |
| 最终所选ROI覆盖 | 87.21% | 90.87% |
| 触台±1帧球心F1@50px | 87.16% | 93.49% |
| 错误跳转次数 | 825 | 534 |
| 最长定位漏检段@50px（帧） | 22 | 22 |
| 定位漏检段P90（帧） | 3.0 | 5.0 |
| 最长无有效观测段（帧） | 4 | 3 |

50px评分中恢复299帧、退化114帧；错误跳转按相邻已知GT的预测位移偏差>50px统计，不把真实快速运动直接算成错误。三个粗视角和720p/1080p均提升。最长漏检仍22帧，剩余漏检段P90由3变5，说明仍有持续跟错问题。

候选oracle诊断（历史CPU FP32，D1正标签5302帧），Global阈值0.7：

| 候选数 | 10px候选命中 | 50px候选命中 | 候选ROI覆盖上限 |
|---|---|---|---|
| 1 | 84.01% | 85.14% | 87.21% |
| 3 | 89.72% | 91.36% | 93.68% |
| 5 | 90.44% | 92.19% | 94.47% |

Top5 oracle可恢复Top1的341/848个10px漏命中（40.21%），这是GT诊断上限，不是实际关联增益。历史CPU诊断与当前CUDA AMP有4帧有效坐标差异>1px，因此两次A错误计数808/809不可当作同后端数值对齐。

[A/B完整报告](</home/radno/projects/pingpong/out/results/20261016-154803_trajectory_priority_seed42/summary.md>)；[恢复与退化逐帧清单](</home/radno/projects/pingpong/out/results/20261016-154803_trajectory_priority_seed42/paired_changes.jsonl>)；[A全部错误视频](</home/radno/projects/pingpong/out/results/20261016-154803_trajectory_priority_seed42/A/index.html>)；[B全部错误视频](</home/radno/projects/pingpong/out/results/20261016-154803_trajectory_priority_seed42/B/index.html>)；[候选oracle完整产物](</home/radno/projects/pingpong/out/results/20261015_fixed3_candidates_full/index.html>)。

## 4. 本轮A0/A/B/C/D：训练设置与轮数

| 方案 | 定位/轨迹 | 事件头 | 事件训练数据 | 实际完成/最佳轮 |
|---|---|---|---|---|
| A0 | 独立解码 | 历史TCN | 仅D3；本轮不重训 | 17 / 12（历史） |
| A | 独立解码 | TCN | D3＋D2，来源各50% | 30 / 25 |
| B | 轨迹关联 | TCN | D3＋D2，来源各50% | 14 / 9 |
| C | 独立解码 | Transformer | D3＋D2，来源各50% | 15 / 10 |
| D | 轨迹关联 | Transformer | D3＋D2，来源各50% | 24 / 19 |

四组均冻结现有三帧Global/Local；只随机初始化并训练事件头。TCN约17.56万参数；Transformer为2层Encoder、128维、4头、FFN256、dropout0.1，约29.98万参数。输入均为13帧、256维视觉描述符＋9维运动/有效性/真实时间信息。

AdamW，lr=3e-4，weight_decay=1e-4，AMP，seed42，有效batch8、micro_batch8；最多30轮，patience5。PP完整验证的三视角落点宏F1选择checkpoint，同分优先FP更少；Bounce阈值网格0.1～0.9，Net默认0.5且PP不监督Net。A/C共用定位特征，B/D共用关联特征，窗口不重置轨迹状态。

每轮7734个窗口、约967次更新；定位器特征提前缓存，因此事件头训练和缓存验证每轮约3～4秒。累计逐轮耗时A约98秒、B约53秒、C约57秒、D约93秒，不含特征准备、初始化和训练后原视频完整评估。

本轮best checkpoint均已保存并完成完整视频评估。部分旧run.json仍写trained_pending_eval；此文依据实际metrics.json和队列DONE结果判定“已评估”，不将旧状态字段误当成未完成。

## 5. 本轮主结果：PP内部验证集完整落点检测

评估数据为D2 val的40完整视频/6292帧；189个空间GT。原视频帧号±1帧、原图距离≤50px、最大基数一对一匹配，unknown不计FP。主指标为front/diagonal/side各自F1的宏平均；表中P/R及TP/FP/FN为合并计数，因此与三视角宏F1不是同一个聚合方式。A0仅在当前PP val选择阈值，不更新权重。

| 方案 | 落点三视角宏F1 | 合并P | 合并R | TP | FP | FN | Bounce阈值 |
|---|---|---|---|---|---|---|---|
| A0 | 50.86% | 42.24% | 67.72% | 128 | 175 | 61 | 0.2 |
| A | 72.44% | 70.16% | 70.90% | 134 | 57 | 55 | 0.3 |
| B | 77.96% | 78.31% | 78.31% | 148 | 41 | 41 | 0.4 |
| C | 55.85% | 76.03% | 48.68% | 92 | 29 | 97 | 0.3 |
| D | 65.63% | 68.09% | 67.72% | 128 | 60 | 61 | 0.3 |

B相对A：宏F1增加 **5.52个百分点**，FP由57降至41，FN由55降至41。A相对A0增加 **21.58个百分点**，包含新增PP监督和重新训练协议的收益，不能全部归于架构。D相对C提升9.78个百分点，关联在两种事件头上均有收益；C/D均未超过对应TCN。

| 方案 | front落点F1 | diagonal落点F1 | side落点F1 | 720p合并落点F1 | 1080p合并落点F1 |
|---|---|---|---|---|---|
| A0 | 48.72% | 49.03% | 54.83% | 54.55% | 37.84% |
| A | 77.19% | 72.22% | 67.91% | 71.60% | 64.29% |
| B | 79.31% | 74.77% | 79.81% | 79.75% | 70.18% |
| C | 50.00% | 52.38% | 65.17% | 62.64% | 40.00% |
| D | 66.67% | 54.87% | 75.36% | 72.96% | 40.68% |

B相对A三个视角均提升。1080p空间GT仅28个，720p为161个，分辨率子组应结合样本量解释。最终落点使用事件时刻预测球心，不包含额外人工接触点回归。

## 6. 连续触台与TTGames稀疏分类分别看

连续触台只判断时间，203个GT；不要求球坐标存在。数据仍为D2 PP val，不能与D3稀疏分类混为同一种F1。

| 方案 | 连续触台P | 连续触台R | 连续触台F1（合并） | TP / FP / FN |
|---|---|---|---|---|
| A0 | 46.13% | 70.44% | 55.75% | 143 / 167 / 60 |
| A | 72.64% | 71.92% | 72.28% | 146 / 55 / 57 |
| B | 79.40% | 77.83% | 78.61% | 158 / 41 / 45 |
| C | 76.38% | 47.78% | 58.79% | 97 / 30 / 106 |
| D | 69.90% | 67.49% | 68.67% | 137 / 59 / 66 |

TTGames分类：只统计原始已标注anchor。历史A0在TTGames val选BCE和阈值；当前A/B/C/D在PP val选checkpoint与Bounce阈值、Net固定0.5。两批阈值/选模协议不同，下面不构成严格的同协议架构排名。

| 模型 | 评估数据 | 样本数 | Bounce F1 | Net F1 | 两类宏F1 | 记录性质 |
|---|---|---|---|---|---|---|
| A0历史 | D3 game_5 val | 450 | 96.49% | 95.82% | 96.16% | 历史独立val选模 |
| A0历史 | D3 test_1–7 test | 464 | 96.88% | 79.61% | 88.24% | 历史独立测试 |
| A | D3 game_5 val | 450 | 96.67% | 61.65% | 79.16% | 当前最佳PP模型的缓存验证快照 |
| B | D3 game_5 val | 450 | 93.79% | 55.69% | 74.74% | 当前最佳PP模型的缓存验证快照 |
| C | D3 game_5 val | 450 | 92.74% | 86.93% | 89.83% | 当前最佳PP模型的缓存验证快照 |
| D | D3 game_5 val | 450 | 86.23% | 37.17% | 61.70% | 当前最佳PP模型的缓存验证快照 |

当前A/B/C/D未完成D3 test_1–7的独立测试；不要把A0历史test成绩归给新模型。C的TTGames分类较好，但PP连续触台漏检106/203；时间诊断放宽到±5帧仍只命中97个，并非大量预测仅偏一两帧。

Transformer原因尚未证实：当前TCN分别投影视觉256→96与运动9→32；Transformer直接拼接265维投影至128再加时间编码，因此本轮比较的是两套完整事件头，不能将差距只归因于attention。[Transformer诊断](</home/radno/projects/pingpong/out/results/20261016_transformer_diagnosis/summary.md>)。

## 7. 历史PP public落点评估（单独列示）

数据为D4：15完整视频、三个角度各5个，92有效落点、12忽略标签。训练来源、事件规则、抽帧和忽略匹配实现存在差异，不能与第5节PP内部验证成绩直接排名。

| 方案 | 训练数据/阶段 | 0° F1 | 45° F1 | 90° F1 | 角度宏F1 | TP / FP / FN |
|---|---|---|---|---|---|---|
| YOLO11n 1280＋原启发式 | D5全量154视频；第32/80轮暂停 | 17.54% | 40.86% | 57.58% | 38.66% | 43 / 81 / 49 |
| 历史冻结TCN A0 | D3事件训练＋PP定位预训练；best12 | 33.33% | 41.10% | 55.17% | 43.20% | 68 / 157 / 24 |
| 历史联合微调TCN | D3；仅第2轮中间评估，后暂停 | 3.17% | 16.33% | 26.92% | 15.47% | 16 / 105 / 76 |

YOLO记录采用“先忽略再匹配”本地近似评分；历史TCN采用先匹配有效GT、再由忽略GT吸收剩余预测。联合模型仅中间结果，不能据15.47%断言联合训练最终无效。YOLO全量训练使用了D1内部val对应视频，因此其public结果不代表D1固定留出协议。

历史A0 public 43.20%与当前A0内部val 50.86%是**同一历史权重在两个不同数据集/阈值协议下的成绩**；不能把差值当成模型训练收益。当前新四组尚未运行public迁移评估。

[YOLO public原始记录](</home/radno/projects/pingpong/experiments/records/20261013-215749_yolo11n_full1280_public/evaluations/epoch032_public_paused.json>)；[历史冻结TCN完整报告](</home/radno/projects/pingpong/ttnet/output/events/20261014-181231_event25_frozen/summary.md>)；[联合版第2轮public指标](</home/radno/projects/pingpong/ttnet/runs/20261014-181231_event25_joint/evaluations/epoch002_public/public/metrics.json>)。

## 8. 速度与batch测试：按范围区分

均为本机RTX 5070 Ti；不同精度、后端、视频集合和是否包含事件头不可直接当作同一基准。

| 模型/测试 | 数据范围 | 后端/精度 | 完整视频FPS | 包含内容 |
|---|---|---|---|---|
| YOLO640，conf .25 | D1六段代表视频 | PyTorch FP32 | 316.73 | 解码到写盘；几乎无检测输出 |
| 官方三帧ONNX | D1六段代表视频 | ORT CUDA FP32/TF32 | 111.23 | 解码、定位、写盘 |
| 自训练单帧Global | D1固定val，按各自benchmark记录 | PyTorch AMP | 319.94 | 纯球定位，不含事件头 |
| 自训练三帧Global | D1固定val，按各自benchmark记录 | PyTorch AMP | 236.90 | 纯球定位，不含事件头 |
| 自训练三帧Global＋Local | D1固定val，按各自benchmark记录 | PyTorch AMP | 141.21 | 纯球定位，不含事件头 |
| A0 | D2全部40原视频/6292帧 | PyTorch AMP | 99.24 | 解码＋定位/轨迹＋事件头＋JSON序列化 |
| A | D2全部40原视频/6292帧 | PyTorch AMP | 100.53 | 解码＋定位/轨迹＋事件头＋JSON序列化 |
| B | D2全部40原视频/6292帧 | PyTorch AMP | 86.54 | 解码＋定位/轨迹＋事件头＋JSON序列化 |
| C | D2全部40原视频/6292帧 | PyTorch AMP | 99.34 | 解码＋定位/轨迹＋事件头＋JSON序列化 |
| D | D2全部40原视频/6292帧 | PyTorch AMP | 85.03 | 解码＋定位/轨迹＋事件头＋JSON序列化 |
| 历史YOLO1280＋启发式 | D4全部15视频 | ORT CUDA FP32 | 66.20 | 历史完整落点管线 |
| 历史A0 public | D4全部15视频 | PyTorch AMP | 97.14 | 历史完整落点管线 |
| 历史joint第2轮 | D4全部15视频 | PyTorch AMP | 77.96 | 与继续训练共享GPU，不作独立速度排名 |

本轮A/B/C/D全部超过50FPS；B相对A约降低13.92%。PyTorch分配器记录的完整推理峰值约48～49MiB，不是整张显卡总占用，也不包含桌面/驱动/其他进程。轨迹额外未来等待0帧；原三帧定位需1帧未来，中心13帧事件窗口需6帧，组合理论等待7帧；执行器仍是离线整视频。

特征提取batch实测：D1 val `22_001`前128帧，CUDA AMP，只计特征提取组件：

| batch | 组件FPS | PyTorch峰值MiB | 相比batch1坐标变化>1px |
|---|---|---|---|
| 1 | 79.67 | 41.7 | 0 |
| 8 | 133.80 | 286.3 | 0 |
| 16 | 133.09 | 569.1 | 0 |
| 32 | 132.29 | 1131.0 | 0 |
| 64 | 126.59 | 2255.0 | 0 |

采用feature_batch8、micro_batch8、validation_batch128；有效batch始终为8、学习率不变。事件头同一有效batch8的前后向耗时micro1/2/4/8分别为12.21/5.26/2.68/1.55ms，不含optimizer.step。AMP描述符最大差异约1.03e-4；128帧中超过1px的坐标变化为0。

早期两视频762帧的初始化事件头运行检查A/B/C/D为98.31/89.99/90.63/86.46FPS，只是运行检查；第5节当前已训练模型的40视频完整评估为正式本轮速度记录。[batch原始记录](</home/radno/projects/pingpong/out/results/20261016_training_preflight/batch_benchmark.json>)。

## 9. 工程验证、数据合并与中断记录

| 验证项 | 数据/对象 | 已有结果 | 证据 |
|---|---|---|---|
| 自动测试 | 模型/缓存/连续评分/轨迹边界等 | 最近完整记录86项：85通过、1项旧图片测试跳过 | [测试日志](</home/radno/projects/pingpong/out/experiments/20261016-154803_trajectory_priority_seed42/tests.log>) |
| 四组真实样本检查 | PP val 02_001真实帧，CPU与CUDA | 前向、梯度、恢复、mask、因果状态与A历史兼容通过；不是精度成绩 | [CUDA真实样本](</home/radno/projects/pingpong/out/results/20261015_comparison_code_checks/real_samples_cuda.json>) |
| 候选诊断 | D1 val全40视频 | 旧CPU诊断808错误帧均有完整叠加视频 | [候选视频索引](</home/radno/projects/pingpong/out/results/20261015_fixed3_candidates_full/index.html>) |
| A/B诊断视频核验 | D1 val每方案40视频 | 80视频帧数/末帧解码/错误引用通过；A809、B658个10px错误 | [核验记录](</home/radno/projects/pingpong/out/results/20261016-154803_trajectory_priority_seed42/overlay_verification.json>) |
| 新事件标注合并 | D1＋event_annotations全部154视频 | 固定114/40划分；原粗标记层和确认后的监督层分开 | [合并记录](</home/radno/projects/pingpong/experiments/records/20261016-150030_pp_event_merge/summary.md>) |
| 重复数据清理/缓存读取 | PP与TTGames原视频、共享NPY | 清理144788文件，释放约49.6GiB；122项历史保护产物核对通过 | [数据清理报告](</home/radno/projects/pingpong/docs/data_cleanup.md>) |

中断/未完成实验不作为成功模型：

| 实验 | 实际状态 | 本次处理 |
|---|---|---|
| 官方ONNX第一次启动 20261013-212251 | 失败；随后212318成功重跑 | 不重复统计精度 |
| 单帧Local 20261014-001042 | 完成2轮后用户暂停 | 没有完整可比较验证成绩，留空 |
| YOLO1280 20261013-215749 | 计划80轮，完成32轮后暂停 | 保留第32轮public中间评估 |
| 联合TCN 20261014-181231_joint | 完成2轮，第3轮step400暂停 | 只列第2轮public中间评估 |
| 20261016-152348 的A | 在特征准备阶段被batch优化替代 | 没有事件训练精度，不计额外A结果 |
| 20261016-153213 的A/B恢复 | 缓存准备/进程恢复，保留attempt和检查点 | 本轮最终结果按实际best checkpoint计一次 |
| 多次smoke与pipeline封装 | 小规模检查或调度器，不是独立模型对照 | 检查通过列入工程验证，不冒充新精度实验 |

完整可发现运行目录的状态见 [运行清单](</home/radno/projects/pingpong/experiments/records/20261016_experiment_comparison/inventory.json>)。

## 10. 当前结论与未完成项

1. 自训练三帧定位明显好于自训练单帧Global，但仍未超过官方ONNX在相同PP视频上的定位验证成绩；官方权重来源、后端和边界处理差异需要保留。
2. 本轮新增PP事件监督后，A相对A0的内部落点宏F1由50.86%到72.44%；它体现新增监督＋重新训练的综合收益。
3. 在本轮固定协议、seed42下，轨迹关联在两种事件头上均带来配对提升：A→B、C→D均提高。B的77.96%/86.54FPS是当前验证候选最佳，尚未证明多种子稳定性。
4. 当前Transformer两组不如对应TCN，主要问题包含连续触台漏检；输入融合和时间编码也与TCN不同，原因待受控消融，不能推广成Transformer普遍不适合。
5. 尚未完成：A/B的seed43、44；当前新四组在OpenTTGames test_1–7、PP public上的最终泛化/迁移；最终逐视角、多种子选型。当前保留A为默认，B为待复验候选，不宣布部署替换。

本轮边界仍是冻结定位器、13帧事件窗口；新骨干、联合微调、桌面几何、时间偏移回归及部署集成为后续阶段。

## 11. 本轮模型与结果入口

| 方案 | 配置 | 权重 | 指标 | 可视化 | checkpoint SHA256前16位 |
|---|---|---|---|---|---|
| A | [配置](</home/radno/projects/pingpong/out/experiments/20261016-153213_event_comparison_b8_seed42_A/config.yaml>) | [best权重](</home/radno/projects/pingpong/out/experiments/20261016-153213_event_comparison_b8_seed42_A/checkpoints/best.pt>) | [完整指标](</home/radno/projects/pingpong/out/experiments/20261016-153213_event_comparison_b8_seed42_A/evaluations/continuous/metrics.json>) | [错误视频](</home/radno/projects/pingpong/out/experiments/20261016-153213_event_comparison_b8_seed42_A/evaluations/continuous/index.html>) | 404aa06b48411b23 |
| B | [配置](</home/radno/projects/pingpong/out/experiments/20261016-153213_event_comparison_b8_seed42_B/config.yaml>) | [best权重](</home/radno/projects/pingpong/out/experiments/20261016-153213_event_comparison_b8_seed42_B/checkpoints/best.pt>) | [完整指标](</home/radno/projects/pingpong/out/experiments/20261016-153213_event_comparison_b8_seed42_B/evaluations/continuous/metrics.json>) | [错误视频](</home/radno/projects/pingpong/out/experiments/20261016-153213_event_comparison_b8_seed42_B/evaluations/continuous/index.html>) | e1516722aafe8493 |
| C | [配置](</home/radno/projects/pingpong/out/experiments/20261016-153213_event_comparison_b8_seed42_C/config.yaml>) | [best权重](</home/radno/projects/pingpong/out/experiments/20261016-153213_event_comparison_b8_seed42_C/checkpoints/best.pt>) | [完整指标](</home/radno/projects/pingpong/out/experiments/20261016-153213_event_comparison_b8_seed42_C/evaluations/continuous/metrics.json>) | [错误视频](</home/radno/projects/pingpong/out/experiments/20261016-153213_event_comparison_b8_seed42_C/evaluations/continuous/index.html>) | a5cc400d2dd97f27 |
| D | [配置](</home/radno/projects/pingpong/out/experiments/20261016-153213_event_comparison_b8_seed42_D/config.yaml>) | [best权重](</home/radno/projects/pingpong/out/experiments/20261016-153213_event_comparison_b8_seed42_D/checkpoints/best.pt>) | [完整指标](</home/radno/projects/pingpong/out/experiments/20261016-153213_event_comparison_b8_seed42_D/evaluations/continuous/metrics.json>) | [错误视频](</home/radno/projects/pingpong/out/experiments/20261016-153213_event_comparison_b8_seed42_D/evaluations/continuous/index.html>) | 8a0ed2d210b512c0 |

A0当前内部评估：[指标](</home/radno/projects/pingpong/out/experiments/20261016-153213_event_comparison_b8_seed42/A0_continuous/metrics.json>)；[错误视频](</home/radno/projects/pingpong/out/experiments/20261016-153213_event_comparison_b8_seed42/A0_continuous/index.html>)。完整权重SHA256、数据和来源均保存于CSV及JSON，避免仅凭实验名称认模型。

完整身份：

| 对象 | 身份 |
|---|---|
| PP球定位v2 dataset_id | f2552a61b4598b68b115854a6ec1995f61c1283eab673c7c376d47c2e2cf2c65 |
| 历史PP球定位v1 dataset_id | 4cd159537792c3b54edf7c3fea4768c2cfd7cc4b74ab46bdeef174ee060fb244 |
| PP train清单SHA256 | 2a09b6f47b95ff93ed1b6bd4ced3fab8c02f71601b1f66ba83e375ccb4c16b23 |
| PP val清单SHA256 | badc1fc0c1f8f002683e3108443137b0210077e84ac1608ea60985540330749f |
| TTGames dataset_id | 470164adc9e57e3820a4fc7ce574f2baa4a42f2c30fb61ea6719aa9265ed976a |
| TTGames split_id | openttgames_event25_match_v1 |
| 确认版PP事件annotation_id | f6fad6f8ae7b9cabb26ecf614b9b36d6d7e12480874f3cff9cdab8ba59990ad1 |
| 本轮合并训练dataset_id | e3ebe4a71a0be7fef612f44f67b470785ef6d86176be2c2f4aa3e7111e820626 |

本报告生成器：[build_experiment_comparison.py](</home/radno/projects/pingpong/scripts/build_experiment_comparison.py>)。来源清单与内容哈希保存于 [sources.json](</home/radno/projects/pingpong/experiments/records/20261016_experiment_comparison/sources.json>)。
