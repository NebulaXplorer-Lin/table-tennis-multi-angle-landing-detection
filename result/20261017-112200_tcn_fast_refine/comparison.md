# 轻量优化结果

固定 PP 114/40 视频、标签和 seed42，冻结定位器；单种子验证选型。

| 方案 | 宏 F1 | TP/FP/FN | 完整视频 FPS |
|---|---:|---:|---:|
| baseline_tcn | 79.56% | 144/34/45 | 77.53 |
| tcn_sampled | 74.17% | 137/46/52 | 86.46 |
| tcn_smoothed | 79.69% | 144/33/45 | 86.88 |

推荐：**tcn_smoothed**；权重：[best.pt](/home/radno/projects/pingpong/out/experiments/20261017-112200_tcn_fast_refine/selected/checkpoints/best.pt)。
最佳验证错误叠加：[index.html](/home/radno/projects/pingpong/out/experiments/20261017-112200_tcn_fast_refine/tcn_smoothed/evaluations/continuous/index.html)。

FPS 包含原视频 PyAV 解码、定位、关联、事件头和 JSON 序列化，不含初始化及磁盘写入。
13 帧事件窗需要未来6帧，定位器另需1帧；该吞吐不代表零前视在线延迟。

锁定阈值外测：TTGames anchor 分类宏 F1 83.53%；PP public 落点宏 F1 70.97%。
TTGames 是稀疏 anchor 分类；PP public 是历史已查看的迁移集。

## 分支记录

- AR stopped at fixed-cache gate; no trajectory cache or event-head training.
- AR 最佳合格权重：正候选命中率 90.09%，null P/R/F1 36.64%/16.16%/22.43%；门槛通过：False。

## 结果解释与验证

- 困难背景共121个；每轮PP实际为773正例、121困难背景、2973普通背景，与3867个TTGames窗口各占50%。采样候选下降，未追加调参。
- EMA的内部改善仅为02_001第21帧少一个误报；没有新增或丢失正确匹配，三视角仅side由78.47%到78.85%。两套外测与原模型相同，未观察到外部精度提升。
- 基线沿用历史完整视频评估；新候选速度满足50 FPS门槛，不将不同时间的测量解释为速度提升。
- AR按预设门槛停止：null召回16.16%低于20%；正常候选命中率90.09%，相对原模型90.71%下降0.62pp。保留GRU权重与逐轮记录，没有重建AR轨迹缓存。
- 31项相关行为及回归测试通过；每组完整验证均覆盖40视频、6292帧，40个pred/GT叠加视频及错误索引齐全。权重SHA256、固定划分、初始化、锁定阈值与外测身份一致。
- 外测只执行了一次，未根据外测结果调参或换模型。完整执行约18分钟，未触及3小时上限；没有自动提交、推送或部署。

事件头197,858参数，AMP/batch1前向p50约0.386ms、p95约0.399ms（RTX5070Ti）；这是头部前向，完整流水线速度另见上表。

原始TCN回退权重：[best.pt](/home/radno/projects/pingpong/result/tcn_frames_lr0.0003_d0.1/best.pt)。
实际生效配置：[config.yaml](./config.yaml)。代码执行快照保存在实验根目录source_snapshot，最终兼容性检查见validation.json。
