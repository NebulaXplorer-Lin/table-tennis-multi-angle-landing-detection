# 配置入口

直接选择 `experiments/` 下的完整方案；`components/` 是共享默认值，`datasets/` 仅描述数据与划分。JSON组件按实验文件声明的顺序合并，再应用实验 `parameters` 和命令行 `--set`；嵌套对象合并，列表替换。

```bash
.venv/bin/python -m ttnet.cli list
.venv/bin/python -m ttnet.cli train --config configs/experiments/event_tcn_frozen.json \
  --set training.lr=0.0001 --set patience=8 --dry-run
```

球定位早停参数位于 `early_stopping`，如 `--set early_stopping.patience=8`；事件训练使用顶层 `patience`。未知键会报错。

常用覆盖：`seed`、`epochs`、`lr` / `training.lr`、`global_input`、`local_input`、`local_crop_side_1080`、`localizer.temporal_mode`。参数列表按JSON格式传入，如 `--set 'global_input=[640,360]'`。

当前组件：`localizer_type=residual_fpn`、`tracker_type=independent/association`、`event_head_type=tcn/transformer`。四组完整配置为 `event_comparison_A/B/C/D.json`；正式训练需要已复核的 PP 连续标注。运行、缓存和标注接口见 [四组对照说明](../docs/event_comparison.md)。详细命令与历史兼容说明见 [工作区说明](../docs/workspace.md)。
