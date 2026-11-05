# PP154训练 / pp_public验证事件实验

2026-11-05，按用户新授权，完成旧B事件训练收尾后继续自主实验。旧结果与机器审计见 [收尾报告](B_event_closeout_20261105.md)。

本轮新建 `pp154_public15_event_v1`，将旧PP40验证视频及其已确认事件样本并入训练：训练为PP154＋OpenTT game1–4，OpenTT game5只保留稀疏辅助验证，test1–7保留。最终落点验证为pp_public的15个视频、2,109个完整原帧、92个有效GT与12个ignored GT。public没有监督训练样本，GT坐标不作为球位置真值或推理输入。完整清单与哈希见 [split](../experiments/splits/pp154_public15_event_v1/split.json) 和 [审计](../experiments/splits/pp154_public15_event_v1/audit.json)。

按public三角度宏F1选checkpoint与Bounce阈值，同分优先更少FP；阈值网格0.1–0.9，原帧±1、原图50px、有效GT优先的一对一匹配，ignored GT吸收其邻域内未匹配预测。这是用户指定的验证集，不能当作独立测试结果；旧PP114/40宏F1属于另一协议。源视频哈希排除字节重复，尚未建立同源重编码切片完全互斥的结论。

三组为无桌面视觉F0、已有视觉拼接concat、零初始化残差调制gated；每组42/43/44种子配对。B高分辨率SE Global、配套Local和YOLO11s桌框冻结，前端仍是旧训练划分选出的权重；本轮扩大的是事件头训练数据。13原帧TCN，AdamW lr3e-4、weight decay1e-4、batch8、AMP、最多40轮、patience8，PP与OpenTT各50%采样。gated与concat共享相同视觉输入，只增加调制分支。

旧159视频特征只在全部推理身份字段相等（仅新数据身份变化）时按视频复制，原缓存保留；public直接以原始RGB和真实PTS提取。public解码最多常驻3张RGB图，不另建整套图像缓存。每组先smoke，再训练，保存best/last和逐轮记录，最后完整在线重评、全帧缓存一致性、全部错误索引和叠加视频、测速。

本轮九组训练和完整public验证均已完成，正式审计通过6,261项检查，135条叠加视频覆盖18,981帧；两组固定旧权重参考也已完成。全部best/last、采样身份、缓存派生、全帧在线一致性、错误索引和测速已核查。详细结果见 [正式三种子报告](../out/experiments/20261105-092319_public_event_study/summary.md)。

| 方案 | public宏F1均值 ± 总体标准差 | 本机离线完整视频FPS均值 |
|---|---:|---:|
| F0 | 68.73% ± 3.42pp | 53.78 |
| concat | 74.30% ± 1.00pp | 43.29 |
| gated | 74.76% ± 2.19pp | 43.25 |

按事先确定的“最高三种子均值方案，再选方案内最高分seed”规则，打包gated seed43第28轮，单次public宏F1为77.59%，TP/FP/FN=75/25/17，Bounce/Net阈值0.3/0.5。它是当前验证选定候选：gated比concat均值仅高0.46pp，只在seed43改善，seed42/44回退，尚不能声称稳定优于concat。权重及全部错误入口见 [结果交付页](../result/public_event_20261105/README.md)。

旧F0/concat seed42固定原权重、原阈值的public成绩为75.69%/73.33%；新实验也改变了epoch/阈值选型数据，因此不是纯训练数据消融。扩大训练集没有带来普遍收益。速度优先时保留旧F0参考（本机54.40FPS）；视觉候选约43FPS，尚未达到50FPS目标。以上为本机PyTorch AMP离线观测，排除初始化、评分、叠加和写盘。

GT辅助定位诊断有88/92事件在±1帧存在50px内有效坐标，不能当作事件F1上限；九组共有14个稳定FN。后续优先研究0°事件判别、45°误报和时间上下文，并只从PP154训练视频构建困难样本，继续保留public只验证的边界。独立桌面视觉与新融合代码已完成，本轮不再扩展同一组门控实验。

任务与事先计划：[task.json](../out/experiments/20261105-092319_public_event_study/task.json) · [pretraining_plan.md](../out/experiments/20261105-092319_public_event_study/pretraining_plan.md)。[代理错误图抽查](../out/experiments/20261105-092319_public_event_study/agent_visual_review_seed42.md)记录了颜色、缩放与接受事件的读图说明；不宣称人类审核。

中断时的历史恢复入口：`.venv/bin/python -u -m scripts.run_public_event_study run --task out/experiments/20261105-092319_public_event_study`。本任务已经完成，无需再次启动协调器。
