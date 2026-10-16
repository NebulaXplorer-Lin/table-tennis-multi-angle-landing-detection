当前配置与命令入口见 [工作区说明](workspace.md)；下文保留本轮事件实验协议与历史入口。

# EventHead 第二阶段

当前采用两个配对实验：冻结 Global/Local 只训练 Head，以及从相同初始权重联合微调三部分。定位器来自已完成的三帧 PP 定位实验；不是从随机初始化联合训练。

## 数据与25 FPS监督

固定清单见 `ttnet/src/configs/event_split.json`：game_1–4 训练，game_5 验证，test_1–7 独立测试。整场视频不跨集合；PP public 只做迁移评估，不参与权重或阈值选择。OpenTTGames 是事件中心稀疏标注，完整视频未标注处不默认当背景。

`event25_v3` 是当前正式数据版本；v1/v2 是准备阶段的中间版本，不用于本次训练。原始120 FPS视频、标注及既有版本均保留。

- 25 FPS目标时刻为 `k/25`，选择最近120 FPS原帧，无插帧。验证实际 PTS 与采样计划一致，保存源帧号及真实时间。
- PP public已是名义25 FPS，保留全部原帧和原帧号。容器平均帧率的微小偏差不用于截断尾帧；依然校验逐帧PTS。
- 原始事件时间通过解码PTS保留。以实际采样时刻计算 `max(0, 1 - abs(t_sample - t_event)/0.04)`，而非先把事件时刻取整再生成标签。
- 每个正事件提供最近目标帧及前后2帧的训练中心；远离正事件的已知类别目标为0，其他类别按mask屏蔽。明确 empty_event 仅监督其中心，事件未标注时刻保持unknown。
- 原始标签映射到同一25 FPS帧时合并正事件；与正事件冲突的empty_event不作为负样本。v3共5处映射碰撞。
- 事件分类的硬标签与训练软标签分开；验证分类指标仍按已知事件类别统计。
- 逐帧球位置只使用实际选中原帧上的坐标，缺标或单轴为-1保持unknown，双轴-1为明确无球。不用接触瞬间坐标替代其他时刻图像的球位置。

| 集合 | 比赛数 | 标注时刻 | Bounce | Net | empty_event |
|---|---:|---:|---:|---:|---:|
| train | 4 | 3352 | 1289 | 986 | 1077 |
| val | 1 | 450 | 248 | 184 | 18 |
| test | 7 | 464 | 240 | 180 | 44 |

此表为25 FPS映射合并后的标注时刻数，不等于训练窗口数。验证集empty_event较少，不能据其样本F1证明完整视频误报率。

## 架构与梯度

每个事件输入13个目标帧（约前后240 ms）。每帧定位器自身再使用前后各1帧，实际图像上下文和离线等待时间约前后280 ms。

视觉分支：Global128维 + Local128维描述符，分别LayerNorm后投影到96维。轨迹分支：归一化x/y、vx/vy、置信度、位置有效、速度有效、Local有效、相对中心实际时间，共9维，经9→32→32 MLP。速度按真实时间差计算；缺失不插值，不跨缺失求速度。

拼接128维后进入三个残差TCN块，每块一个kernel=3卷积，dilation为1/2/3，逐时刻LayerNorm、SiLU、Dropout0.1。取中心帧输出Bounce/Net两个logits。输入及每个TCN块都屏蔽无效帧，无效Local描述符清零。

- frozen：定位器冻结并保持eval；缓存身份绑定数据与两路权重、提取配置、代码哈希。只训练Head。
- joint：在线执行两路定位，事件损失通过视觉描述符回传编码器；热图用有标注帧监督。硬解码、ROI坐标及轨迹输入停止梯度，事件损失不穿过裁剪位置。
- 联合损失为事件masked BCE + Global定位MSE + Local定位MSE，各项按有效监督归一化；缺失Global裁剪时屏蔽Local损失。

## 训练、评估与运行

实际默认配置见 `ttnet/src/configs/event.json`。AdamW，Head lr=3e-4，联合定位lr=3e-5，weight decay=1e-4，micro-batch=1、累积8、有效batch=8，AMP，seed42，最多30 epoch。两版按相同seed/epoch生成相同平衡采样顺序；三组按锚点等量采样，再为正事件随机选择一个中心偏移。

每轮验证masked BCE最低保存best；连续5轮未相对改善1%早停。保存best、last、optimizer/scaler/RNG，续训使用同一run的新attempt。正式训练前完成两版真实视频smoke与标签叠加检查。

```bash
.venv/bin/python -m ttnet.events.prepare
.venv/bin/python -m unittest discover -s ttnet/src/tests -v
.venv/bin/python -u -m ttnet.events.pipeline --phase smoke
.venv/bin/python -u -m ttnet.events.pipeline --phase train --smoke-task ttnet/runs/<smoke_pipeline>
# 单模型精确续训
.venv/bin/python -m ttnet.events.train --mode joint --run ttnet/runs/<joint_run> --resume
```

管线保存所有子进程stdout/stderr和退出状态。两版依次训练和评估，不自动追加其他实验。

- OpenTTGames：标注锚点分类P/R/F1、逐视频结果；事件附近±2帧扫描的时间偏移及±1帧命中率；有可靠标签帧的定位F1与误差。未测完整连续事件F1。
- 留出的game_5完整视频另做逐帧推理与未评分事件输出；不以缺标判误报。
- PP public：±1帧、50px、一对一最大匹配，忽略标记的干扰GT吸收剩余近邻预测；报告分视角及宏平均F1。坐标缺失不伪造落点，单列统计。
- 完整视频FPS包含PyAV解码、采样、预处理、定位、Head与JSON序列化；初始化与磁盘写入另计。Head测50次warmup、3×200同步前向延迟。
- 已评分的分类、时间、球定位和PP落点错误保存pred/label叠加图、错误索引和HTML图册。

最终结果入口为 `ttnet/output/events/README.md`，CSV为对照表数据源；各run包含实际配置、权重身份、日志、曲线、错误图册及改进假设。指标不佳也保留。未完成评估不标记completed。

## 已知限制与后续判断

25 FPS抽帧不能保证输入含真实接触瞬间，也不能模拟真实25 FPS曝光。首版最终落点取事件帧球坐标，可能存在空间偏移。现有描述符经过2×2空间池化，事件关系信息可能不足。

后续改进依错误证据决定：定位召回/ROI问题优先检查阈值与域适配；事件混淆考虑ROI空间特征和明确困难负样本；时间偏差考虑多采样相位或时间偏移回归；多视角落点问题考虑补PP train连续事件标注和真实接触位置回归。这些是诊断方向，不代表已验证的提升。
