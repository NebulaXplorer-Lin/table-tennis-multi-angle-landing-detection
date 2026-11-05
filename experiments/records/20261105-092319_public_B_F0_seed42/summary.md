# PP public事件训练结果

状态：completed（独立机器完成收据；模型run.json保留原始训练状态）。

方案F0，seed42；best epoch 15 / 完成23轮；阈值0.2。

PP public验证宏F1 66.4729%；TP/FP/FN=61/23/31；完整视频53.62FPS。

[完整三种子比较](../../../out/experiments/20261105-092319_public_event_study/summary.md) · [全部错误/完整叠加视频](../../../out/experiments/20261105-092319_public_B_F0_seed42/evaluations/public/errors.html) · [指标](../../../out/experiments/20261105-092319_public_B_F0_seed42/evaluations/public/metrics.json) · [最佳权重](../../../out/experiments/20261105-092319_public_B_F0_seed42/checkpoints/best.pt)。

训练使用全部154个PP视频；pp_public用于验证和选型，不作为独立外测。全部15视频完整帧cache/online及落点决策验收通过；所有FP/FN/无坐标事件均有图和完整视频。未宣称人类审核。
