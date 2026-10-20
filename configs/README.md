# 配置入口

| 目的 | 配置 | 状态 |
| --- | --- | --- |
| 三帧Global/Local球定位 | `experiments/ball_fixed3.json` | 当前定位前端 |
| TCN + 关联 + 旧桌框 + 时间编码 | `experiments/event_tcn_reference.json` | 当前三种子完整主基线 |
| A/B/C/D研究对照 | `experiments/event_comparison_A/B/C/D.json` | 已完成的历史对照；B无桌框 |
| 旧OpenTTGames冻结事件头 | `experiments/event_tcn_frozen.json` | 历史参考 |
| 100GB服务器定位 | `server/ball_compact.json` | 原视频读取，禁用磁盘图像缓存 |
| 服务器当前TCN参考 | `server/event_reference.json` | 使用weights/reference；旧桌框，无EMA |

`event_tcn_joint.json`和`legacy/`为兼容保留，不是当前三阶段新分支训练方案。独立单帧桌框和新融合尚无可直接启动的完整配置。

组件按实验声明的顺序合并，再应用实验`parameters`和命令行`--set`。嵌套对象合并，列表替换；未知键报错。修改共享组件会影响引用它的所有新实验，单次对照优先新方案或`--set`。

```bash
.venv/bin/python -m ttnet.cli list
.venv/bin/python -m ttnet.cli train \
  --config configs/experiments/event_tcn_reference.json \
  --set seed=43 --dry-run
```

球定位早停参数在`early_stopping`，事件早停使用顶层`patience`；尺寸列表按JSON传入，如`--set 'global_input=[640,360]'`。完整操作见[训练指南](../docs/training_guide.md)。
