# 历史兼容配置

当前完整方案从[配置入口](../README.md)选择；训练先读[指南](../../docs/training_guide.md)。本目录与`ttnet/src/configs/`仅用于历史checkpoint/接口兼容，不以存在配置文件代表已经完成训练验证。

| 配置或路径 | 历史用途 |
| --- | --- |
| default.json | 单帧Residual FPN |
| fixed3.json | 三帧定位原始平铺配置 |
| fps_mask9.json | 九槽时序接口，未形成完整训练版本 |
| multiscale.json | 空间多尺度候选，未验证收益 |
| event.json、event_tcn_joint.json | 旧冻结/联合事件路径；中断实验已归档 |

保留的完整实验从[实验索引](../../experiments/index.md)查看。未来新路线以[当前计划](../../docs/next_steps.md)为准。
