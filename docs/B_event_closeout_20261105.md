# B事件旧训练收尾（2026-11-05）

旧九组实验的机器可验收尾已完成。本次只读取既有训练和评估产物，并创建独立收据；没有重训、GPU推理或更改旧受哈希保护记录。未执行人类审核；本代理检查了seed44三张代表图，机器验收不依赖用户确认。

| 方案 | 三种子宏F1均值 ± 总体标准差 | 本机完整视频FPS均值 |
|---|---:|---:|
| F0 | 79.26% ± 1.82pp | 53.20 |
| geometry | 80.08% ± 3.51pp | 43.28 |
| concat | 82.77% ± 0.82pp | 43.22 |

B高分辨率SE Global和Residual FPN Local均冻结，阈值0.7/0.7；13原帧TCN，AdamW lr3e-4、weight decay1e-4、batch8、AMP、最多40轮、patience8。旧PP114/40视频划分，OpenTT game1–4训练/game5验证，两来源各50%采样。原帧±1、原图50px一对一匹配，按三视角宏F1选best。历史主基线同时存在前端、桌框和事件设置差异，不能据差值归因于某一模块。

当前CPU强审计通过：1525项来源哈希、九组独立身份、全部best/last、逐轮采样、修正后的159视频桌框缓存、完整40视频缓存/在线概率与决策一致性、全部错误索引及九组benchmark。360条叠加视频此前逐帧解码共56,628帧，本次重新核对这些视频和解码收据的哈希；没有声称重新解码。两项seed42历史缓存凭证例外继续按原报告披露。

[机器收据](../experiments/records/20261105_B_event_closeout/evaluations/audit.json) · [逐运行CSV](../experiments/records/20261105_B_event_closeout/comparison.csv) · [旧完整三种子报告](../out/experiments/20261104-182418_B_event_exploration/replication_summary.md)。

旧run/task与analysis中的pending状态字符串保留；旧报告来源除各自task.json外均受哈希约束，不能为使状态好看而改写。本独立收据的status=completed只覆盖机器可验的旧实验工作；不表示人类签字，也不表示新pp_public协议已评估完成。旧父目录summary.md仍是退役布局的历史结果，新结论以三种子报告及本收据为入口。

下一轮最有希望先保持concat配方不变，将旧验证40视频并入训练，以独立pp_public作为新的最终验证协议，先建立可比基线。更换split_id并保存训练/验证视频与标签身份，排除同源视频重叠；旧内部指标单列。

之后优先研究事件判别及时间上下文。concat三种子在旧40视频仍有20个共同FN，其中16个在全部九组均漏检；斜视均值76.33%，低于侧视86.31%和正视85.66%。seed44的02_002:58在共享球定位下F0命中而桌框头漏检，支持先排查视觉分支压制或事件时间判别，再开展单变量的长窗口/多尺度时间上下文或残差门控视觉融合。训练可利用已授权并入训练的旧验证错误；不能将pp_public标签制成训练hard-negative清单。

组22的无ROI切片并非当前首要瓶颈：concat三个种子在整个13帧窗无桌面ROI的40个GT上均40/40命中。原始13帧在不同帧率下对应不同时间跨度，应先核对共同FN轨迹和中心事件概率，再决定时间窗口改动。静态代表图不用于宣称具体漏检原因。

50FPS仍未达到：concat平均43.22FPS，F0为53.20FPS。若目标包括速度，可在准确率基线明确后另做桌面特征稀疏更新/缓存或轻量视觉分支的单变量实验；这会改变算术或推理协议，需要重新进行缓存/在线完整帧验收。

[seed42错误入口](../out/experiments/20261104-195314_B_event_seed42_contiguous/comparison.md) · [seed43错误入口](../out/experiments/20261104-194046_B_event_seed43_replication/comparison.md) · [seed44错误入口](../out/experiments/20261104-194046_B_event_seed44_replication/comparison.md) · [seed44配对案例](../out/experiments/20261104-194046_B_event_seed44_replication/review_cases/index.html)。
