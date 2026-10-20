# 实验记录

当前入口为[统一索引](index.md)与[结果首页](../result/README.md)。组员先读[Quick intro](../docs/quick_intro.md)。

- `splits/`：固定视频划分、原始审计和身份；继续复用，不因换模型或seed重分。
- `records/`：保留版本、完整对照及数据处理的轻量记录。
- `reports/`：已完成的历史规则诊断报告。
- `curation.json`：已核验结果快照、五套交付版本及归档实验编号。
- `workspace_comparison.csv`、`index.json`、`index.md`：由`python -m ttnet.cli results`刷新。
- `comparison.csv`及其他历史汇总：保存原评估口径，不将不同划分或任务拼成一张排名。

中断、失败启动、smoke及未形成完整评估的网格候选记录已归档；完整负结果仍保留。大权重、日志和视频在`out/`，不纳入Git。历史原始`run.json`可能早于评估，完成状态以评估metrics和`evaluation_receipt.json`为准。清理范围见[记录](../docs/workspace_cleanup_20261020.md)。
