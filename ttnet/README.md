# 研究模型代码

组员入门见[Quick intro](../docs/quick_intro.md)，命令见[数据与训练指南](../docs/training_guide.md)，完整结果见[result/README.md](../result/README.md)。`ttnet`为历史包名，当前定位器是自建Residual FPN。

## 实现导航

| 模块 | 职责 |
| --- | --- |
| `src/cli.py`、`configuration.py` | 配置展开、统一训练/评估/导出入口 |
| `src/models/localizer.py` | 三帧共享编码器、四尺度时序融合、热图与128维描述符 |
| `src/models/event.py` | 旧EventHead、当前ContextEventHead、Transformer对照 |
| `src/tracking/` | Top-K候选、规则历史关联、双ROI运行时、AR实验 |
| `src/scene/` | 旧共享特征桌框头、几何、对照与报告 |
| `src/training/` | Global/Local数据与训练 |
| `src/events/` | PP/OpenTTGames混合、特征缓存、TCN训练、连续评分、joint基础路径 |
| `src/data_tools/` | 数据描述、标签合并、NPY/原视频FrameStore、存储校验 |
| `src/evaluation/`、`src/tests/` | 定位评估与回归检查 |

`__init__.py`将`src/`暴露为包路径，在仓库根目录运行`python -m ttnet.cli`。数据和输出默认读写根目录`data/`与`out/`。本机`data`、`output`等软链接属于历史兼容，新机器不依赖它们。

当前完整方案为`configs/experiments/ball_fixed3.json`与`event_tcn_reference.json`，服务器用`configs/server/`。后续独立桌框分支和少量FT尚待实现，见[当前路线](../docs/next_steps.md)。历史小样本预检、暂停实验与旧规划已移出日常入口，见[整理记录](../docs/workspace_cleanup_20261020.md)。
