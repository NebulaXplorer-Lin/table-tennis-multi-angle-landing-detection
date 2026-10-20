# 轨迹关联优先验证：seed42

按用户要求先比较 A/B 定位，随后顺序为 A 训练、B 训练、A/B 事件验证、C/D 训练验证、A0 验证及四组汇总。

- 实时状态：[status.json](/home/radno/projects/pingpong/out/experiments/20261016-154803_trajectory_priority_seed42/status.json)
- 轨迹关联对照：[comparison.md](/home/radno/projects/pingpong/out/results/20261016-154803_trajectory_priority_seed42/comparison.md)
- 轨迹逐帧变化：[paired_changes.jsonl](/home/radno/projects/pingpong/out/results/20261016-154803_trajectory_priority_seed42/paired_changes.jsonl)
- 每组全部错误叠加视频：[A](/home/radno/projects/pingpong/out/results/20261016-154803_trajectory_priority_seed42/A/index.html)、[B](/home/radno/projects/pingpong/out/results/20261016-154803_trajectory_priority_seed42/B/index.html)
- 完整视频速度：[runtime.json](/home/radno/projects/pingpong/out/results/20261016-154803_trajectory_priority_seed42/runtime.json)

诊断在固定40个PP验证视频上完整推理；GT仅用于评分。球心10/50px F1、所有/选中ROI覆盖、GT相对错误跳转、连续漏检、触台±1帧定位和视角/分辨率分别统计。定位优劣不代替连续事件训练后的落点评价。诊断前事件头尚未产生优化器更新，A从原初始化继续并复用已完成视频缓存。

原运行目录、配置、batch8和已通过的四组启动检查保留。A/B事件训练完成前不抢先运行Transformer。seed43/44及独立test/public仍待seed42完成后执行。

## 已完成的定位对照

A/B的50px球心F1：0.8606 / 0.8941；完整视频FPS：97.78 / 88.29。

[定位详细报告](/home/radno/projects/pingpong/out/results/20261016-154803_trajectory_priority_seed42/summary.md)。定位收益不代表事件收益；A/B事件训练及后续四组/多种子仍未完成。

## 原训练队列已恢复

用户澄清自回归定位仅为讨论，并明确要求恢复原任务。已跳过完成的轨迹定位对照与A训练（30轮，best/last已保存），从B的特征准备继续；已完成视频缓存复用，未完成视频从首帧按时间顺序重建。随后执行A/B连续验证、C/D训练验证、A0与四组报告。仍为冻结定位器、事件头训练、batch8、seed42。暂停前队列计划、状态及日志保存于queue_attempts/001；本次启动信息见resume.json。
