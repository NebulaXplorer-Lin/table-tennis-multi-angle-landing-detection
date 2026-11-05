# PP154训练 / pp_public验证交付

旧B事件训练已完成独立收尾；本轮九组新训练、两组固定旧参考和完整产物审计均已完成。最终落点验证为pp_public15视频、2,109原帧、92有效GT与12个ignored GT；旧PP40验证已并入事件训练，public无监督训练样本。

| 方案 | 三种子宏F1均值 ± 总体标准差 | 本机离线完整视频FPS均值 |
|---|---:|---:|
| F0 | 68.73% ± 3.42pp | 53.78 |
| concat | 74.30% ± 1.00pp | 43.29 |
| gated | 74.76% ± 2.19pp | 43.25 |

按最高三种子均值方案、再选该方案内最高分seed的既定规则，交付 **gated seed43第28轮**：单次public宏F1 **77.59%**，TP/FP/FN=75/25/17，Bounce/Net阈值0.3/0.5，Global/Local阈值0.7/0.7。三帧B SE Global与Residual FPN Local、独立YOLO11s桌面视觉冻结，13原帧TCN，AdamW lr3e-4/wd1e-4、batch8、AMP、最多40轮/patience8。

这是当前public验证选定候选。gated相对concat仅seed43改善，另外两种子回退；均值高0.46pp且波动更大，不能认定稳定优于concat。Public用于epoch、阈值和方案选择，结果不作为独立测试。

旧F0/concat seed42固定原权重和阈值的公共宏F1为75.69%/73.33%；旧F0本机54.40FPS可作速度参考。新视觉候选约43FPS，尚未达到50FPS目标。速度是本机PyTorch AMP离线完整视频观测，包含解码与JSON序列化，排除初始化、评分、叠加与写盘。

完整参数已包含Global、Local、事件头及桌面前端；运行仍需本仓库代码和out/yolo-deps中的ultralytics8.3.225。原实验与权重均保留。

- [候选权重压缩包](../../out/weights/20261105-145517_public_event_best.tar.gz) · [压缩包SHA256](../../out/weights/20261105-145517_public_event_best.tar.gz.sha256)
- [best.pt](../../out/weights/20261105-145517_public_event_best/best.pt) · [配置与加载说明](../../out/weights/20261105-145517_public_event_best/README.md) · [包内校验清单](../../out/weights/20261105-145517_public_event_best/SHA256SUMS)
- [完整三种子报告](../../out/experiments/20261105-092319_public_event_study/summary.md) · [逐轮曲线](../../out/experiments/20261105-092319_public_event_study/training_curves.png) · [九组CSV](../../out/experiments/20261105-092319_public_event_study/comparison.csv)
- [候选全部错误与15条完整叠加视频](../../out/experiments/20261105-092319_public_B_gated_seed43/evaluations/public/errors.html) · [逐视频指标](../../out/experiments/20261105-092319_public_B_gated_seed43/evaluations/public/per_video.json)
- [正式机器审计](../../out/experiments/20261105-092319_public_event_study/analysis.json) · [完成收据](../../out/experiments/20261105-092319_public_event_study/completion.json) · [打包验证](../../out/weights/20261105-145517_public_event_best/verification.json)
- [旧实验收尾](../../docs/B_event_closeout_20261105.md) · [新协议与后续结论](../../docs/public_event_study_20261105.md)

九组全部best/last、逐轮采样与日志均在原运行中。135条新叠加视频共18,981帧，全部错误覆盖、完整编码后解码和全帧缓存/在线概率、坐标、落点决策验收通过；另有两份固定旧参考的完整公共评估。机器验收不等于人类语义审核。
