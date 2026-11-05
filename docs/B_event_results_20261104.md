# 新B定位器训练结果（2026-11-04）

已完成42/43/44三种子、三组方案，共九组训练和固定40视频完整验证。使用B高分辨率SE Global＋配套Local，冻结前端训练13原帧TCN；比较无桌框F0、YOLO桌框几何、几何＋ROI/上下文视觉concat。

| 方案 | 三种子宏F1均值 ± 标准差 | 本机完整视频FPS均值 |
|---|---:|---:|
| F0 | 79.26% ± 1.82pp | 53.20 |
| geometry | 80.08% ± 3.51pp | 43.28 |
| concat | 82.77% ± 0.82pp | 43.22 |

PP固定114/40视频；OpenTT game1–4训练/game5验证，两来源各50%。AdamW lr3e-4、weight decay1e-4、batch8、AMP、最多40轮、patience8，按原帧±1/原图50px三视角宏F1选best，Global/Local阈值均0.7。标准差为三种子总体标准差(ddof=0)。速度为RTX5070Ti、PyTorch AMP，包含解码至JSON序列化，排除初始化与写盘。

内部均值最高为concat；历史主基线三种子均值78.77%。历史同时更换了前端、桌框和事件设置，差值不代表定位器单一收益。视觉方案带来计算开销，未达到本机50FPS目标；F0保留为速度对照。当前结论仅限内部验证，未新增public/test选型或外测。

已修复NPZ重复解压与缓存/在线张量布局差异；最终九组全部通过完整帧概率、坐标与事件决策一致性检查，360条叠加视频共解码56,628帧。原布局不一致的桌框试验保留，但未混入三种子均值。元数据审计明确记录最早F0及修正版geo42的两项窄凭证例外，未伪造原始记录。

[三种子完整报告、逐种子错误入口、曲线与权重](../out/experiments/20261104-182418_B_event_exploration/replication_summary.md) · [机器可读强审计](../out/experiments/20261104-182418_B_event_exploration/replication_analysis.json)。全部best/last、逐轮采样、预测与错误索引已保留。
