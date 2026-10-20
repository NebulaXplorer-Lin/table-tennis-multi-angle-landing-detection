# 工作区说明

组员先读 [Quick intro](quick_intro.md)；数据和训练命令集中在[训练指南](training_guide.md)。

## 日常入口

| 内容 | 位置 |
| --- | --- |
| 当前完整模型结果 | [result/README.md](../result/README.md) |
| 保留实验及指标 | [experiments/index.md](../experiments/index.md) |
| 三帧球定位方案 | `configs/experiments/ball_fixed3.json` |
| 当前完整TCN参考 | `configs/experiments/event_tcn_reference.json` |
| 100GB服务器配置 | `configs/server/ball_compact.json`、`event_reference.json` |
| 下一步研发与完成条件 | [next_steps.md](next_steps.md) |

## 代码与配置

`ttnet/__init__.py`把`ttnet/src/`作为包实现，入口为`python -m ttnet.cli`。定位、轨迹、桌框、事件、数据、评估分别在`models/`、`tracking/`、`scene/`、`events/`、`data_tools/`、`evaluation/`，不按实验复制整套代码。

组合配置由`configs/components/`、`configs/datasets/`和`configs/experiments/`组成。参数按组件声明顺序合并，再应用实验`parameters`和命令行`--set`；嵌套对象合并、列表替换，未知键报错。单次实验用`--set`或新方案，不随意修改所有方案共享的默认值。

`configs/legacy/`及`ttnet/src/configs/`保留checkpoint兼容与数据固定清单；其中fps_mask9、多尺度、旧joint属于历史接口，不是当前推荐训练路线。研究模型未替换`pingpang/participant/solution.py`的YOLO规则入口。

## 数据与存储

原视频位于`data/pp_train_data`、`data/pp_public_data`和`data/ttnet`。训练数据描述为`data/datasets/pp_ball_v2`、`pp_ball_events_v1`和`openttgames_event25_v3`；确认版连续事件在`data/annotations/pp_bounce_confirmed_v1.json`。

FrameStore支持两种后端：本机已有RGB uint8分块NPY，以只读mmap读取；服务器用`video_store --publish`发布源视频索引，按真实PTS seek解码，内存有界。发布会保存原descriptor备份，不应在本机原版本上随意切换。稀疏数据的数组行号不等于视频帧号。

本机NPY库共52,573帧、约233GiB，PP位置/事件描述共用画面；它不是组员服务器的迁移前提。100GB服务器使用原视频及描述约33.74GiB，禁用图像磁盘缓存，见[服务器方案](server_100gb_20261018.md)。历史`frame_path`/`path`可能只是来源记录，实际读取通过FrameStore。

`data/cache/features/`存冻结视觉描述符，身份绑定数据、权重、轨迹、精度、读取后端与实现；这些文件不同于原图缓存。身份变化后生成新缓存，不静默复用。

## 记录与结果

- `experiments/records/<id>/`：生效配置、轻量训练/评估记录。部分历史`run.json`早于完整评估，`evaluation_receipt.json`与实际metrics提供完成证据。
- `experiments/curation.json`：保留结果快照与归档实验编号；刷新索引时读取实际运行评分，并避免把归档实验重新带回。
- `out/experiments/<id>/`：实际权重、优化器、逐轮日志、完整评估与错误视频。
- `out/launches/<id>/attempt_*/`：统一入口的stdout/stderr、命令及退出码。
- `result/`：五套完整版本的交付入口。权重和视频不纳入Git，组内另传。
- `archive/`：本机旧工作区和本次清理的可恢复副本，不是新机器必须克隆的数据。

```bash
.venv/bin/python -m ttnet.cli results
```

该命令刷新`index.md`、`index.json`和`workspace_comparison.csv`，不启动训练。历史`comparison.csv`、完整对照报告仍作为原协议的证据保留。已完成的负结果继续可查，smoke、失败启动、无完整验证的暂停实验不进入日常结果入口。

旧`ttnet/data`、`ttnet/output`等本机软链接为历史兼容；服务器使用`configs/server/`与`weights/reference/`。定位模型支持既有ONNX导出检查，整个事件系统的ONNX导出与赛事部署仍待完成。

本次整理的范围、恢复方法与检查见[工作区整理记录](workspace_cleanup_20261020.md)。
