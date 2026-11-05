# 新B定位器事件训练探索

2026-11-04（Asia/Shanghai），按用户授权自主选择训练方案并使用新定位器。本轮状态以[任务记录](../out/experiments/20261104-182418_B_event_exploration/task.json)为准；训练开始前已保存[诊断与假设](../out/experiments/20261104-182418_B_event_exploration/pretraining_diagnostics.md)。本轮已完成九组训练及验证，最终结论见[三种子结果](B_event_results_20261104.md)。

固定B的高分辨率SE Global（best epoch13）与配套Residual FPN Local（best epoch19），采用已完成球定位验证的0.7/0.7阈值及规则关联。球定位验证F1@50px为92.98%，不能直接作为落点成绩。原mixed-ball比较实际上已经完成，见[完整结果](../out/results/20261103-140723_mixed_ball_comparison/summary.md)。

本轮三组分别为无桌框F0、YOLO桌框几何geometry、YOLO几何与桌面视觉concat。后两组固定使用原生YOLO11s桌框第12轮best、640输入、confidence0.1，保留组22大量漏框这一已知局限。使用相同PP114/40视频划分、OpenTT game1–4训练/game5验证、两来源各50%采样、13原帧TCN、seed42、AdamW lr3e-4/wd1e-4、effective batch8，最多40轮、patience8；每轮7734个样本，事先保存40轮采样身份供验收。只按内部40视频原帧±1/原图50px的三视角宏F1选型，同分时FP更少优先，不使用public/test调参。

B事件特征缓存从原视频开头按时间顺序生成；OpenTT稀疏间隙只用于关联上下文，不另存整套RGB。独立YOLO桌面特征通过[缓存追加入口](../ttnet/src/scene/augment_event_cache.py)生成，严格核对源SHA、球权重、阈值、数据及源码身份，保留球特征和轨迹，geometry/concat复用同一追加缓存。

训练前F0小样本反向与优化检查已通过，12步事件损失0.5523→0.2562，定位器冻结不变。32项事件/轨迹检查通过。缓存追加检查覆盖身份拒绝、原数组和轨迹保留、中断恢复及batch变化拒绝。另有[8帧真实GPU AMP验收](../out/results/20261104_event_table_cache_acceptance/gpu_batch8_eight_frames.json)：批量/单帧的候选数、有效性与框坐标一致，置信度最大差5.49e-4、ROI相对L2最大5.73e-4；部分ROI不满足严格2e-4容差。最终仍须核对缓存选型结果与在线40视频完整推理，不能用这8帧替代完整验收。

在正式桌框缓存开始前，依据上述数值检查将最终提取方式固定为batch1，与在线逐帧算术一致；batch8结果仅作为数值诊断保留。等待中的协调器已单独重启，B缓存与三组模型配方保持原计划；决策和退出状态写入任务记录。

首个桌框视频提交时发现NPZ恢复函数逐帧重复解压整个数组，已修复为每个数组只读取一次；新增压缩数组读取次数回归检查，24项缓存/桌面/报告检查通过。真实3169帧恢复耗时0.30秒，输出张量核对一致。原中断日志和未提交缓存保留，随后按新源码身份重建桌框缓存；[修复证据](../out/experiments/20261104-182418_B_event_exploration/cache_restore_fix.json)。

首轮seed42完成后，F0的6292帧在线结果完全一致；桌框两组却发现差异，concat少命中`12_001:66`一个落点。已定位为缓存的转置数组保留了channels-last物理步长，而在线为连续NCHW，导致AMP执行路径不同；源像素与letterbox图像逐值一致。现已统一连续NCHW，在11个困难帧以及实际checkpoint重载的8个帧上，框、分数和桌面特征全部逐值一致。原结果保留为中间证据，按[修正版seed42任务](../out/experiments/20261104-195314_B_event_seed42_contiguous/task.json)重建桌框缓存并重训geometry/concat；F0复用已通过全帧一致性的结果。[布局修复证据](../out/experiments/20261104-182418_B_event_exploration/input_layout_fix.json)。

首轮几何组相对F0的缓存宏F1收益仅0.83pp，因此已在看到43/44任何结果前固定三组配对复核：[复核任务](../out/experiments/20261104-182418_B_event_exploration/replication.json)。最终结论使用修正输入契约后的42/43/44结果，首轮布局不一致的桌框实验不混入均值；public/test不参与选型。

重点反思切片：旧45FN中GT±1球覆盖仅38→39，但确切帧覆盖30→36；新定位器收益可能体现在时刻对齐。组22含46/189空间事件，40个事件整个13帧窗无YOLO框，其中35个为旧模型TP。除了追回旧FN，还必须检查原TP回退、侧视FP，以及无ROI时仍可用的全图context。GT仅用于诊断，不进入推理输入。

运行协调器会保留三组best/last、逐轮日志、完整验证预测、全部错误叠加视频及退出状态。恢复入口为`python -m scripts.run_b_event_study --task out/experiments/20261104-182418_B_event_exploration`；运行中的协调器持有独占锁。历史79.56%使用旧球定位器和旧桌框，本轮与历史的差异不能归因于单个结构；单种子选型也不代表稳定收益。
