# PP public事件训练结果

状态：completed（独立机器完成收据；模型run.json保留原始训练状态）。

方案gated，seed43；best epoch 28 / 完成36轮；阈值0.3。

PP public验证宏F1 77.5929%；TP/FP/FN=75/25/17；完整视频43.18FPS。

[完整三种子比较](../../../out/experiments/20261105-092319_public_event_study/summary.md) · [全部错误/完整叠加视频](../../../out/experiments/20261105-092319_public_B_gated_seed43/evaluations/public/errors.html) · [指标](../../../out/experiments/20261105-092319_public_B_gated_seed43/evaluations/public/metrics.json) · [最佳权重](../../../out/experiments/20261105-092319_public_B_gated_seed43/checkpoints/best.pt)。

训练使用全部154个PP视频；pp_public用于验证和选型，不作为独立外测。全部15视频完整帧cache/online及落点决策验收通过；所有FP/FN/无坐标事件均有图和完整视频。未宣称人类审核。
