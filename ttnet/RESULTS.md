[统一方案与实验索引](../experiments/index.md) · [后续计划](../docs/next_steps.md)

# TTNet 结果入口

第二阶段 EventHead 已增加冻结定位与联合微调两版实验；运行与评估协议见 [说明](../docs/eventhead_training.md)，训练任务状态见 [本轮管线](runs/20261014-181231_event_train_pipeline/run.json)。最终结果单独汇总到 `output/events/README.md`，避免与球定位 F1 混用。

更新于 2026-10-14T12:28:12.994679+09:00。

[查看全部最终结果、权重、Recall / F1、测速和错误样本](output/README.md)

最终文件集中在 `output/<任务编号>/`；每个 summary.md 提供直接链接。`runs/` 保存启动、续训、日志和中间产物。

| 当前实验 | 运行状态 | 完成 epoch | 最终结果 |
|---|---|---:|---|
| 20261014-095202_global_fixed3_ball | completed | 20 | [汇总](output/20261014-095202_global_fixed3_ball/summary.md) |
| 20261014-103011_local_fixed3_ball | completed | 20 | [汇总](output/20261014-103011_local_fixed3_ball/summary.md) |
