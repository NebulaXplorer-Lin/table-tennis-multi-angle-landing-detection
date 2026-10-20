# 乒乓球落点检测

面向组员的入口：[Quick intro](docs/quick_intro.md)。这里说明实验脉络、当前模型、代码结构、数据和后续路线；实际操作见[数据准备与训练指南](docs/training_guide.md)。

当前研究基线为 **三帧 Global/Local 定位 + 轨迹关联 + 桌框几何 + 13帧 TCN**。内部 PP 验证宏 F1 为79.56%，三种子均值78.77%；EMA单种子候选为79.69%。研究模型尚未替换赛事入口。

| 要做的事 | 入口 |
| --- | --- |
| 快速了解项目 | [Quick intro](docs/quick_intro.md) |
| 准备数据、训练、验证与恢复 | [训练指南](docs/training_guide.md) |
| 查看保留的完整版本、权重和错误视频 | [结果首页](result/README.md)、[实验索引](experiments/index.md) |
| 找代码、配置和产物 | [工作区说明](docs/workspace.md)、[配置说明](configs/README.md) |
| 继续开发 | [当前路线](docs/next_steps.md)、[训练白皮书](docs/training_whitepaper_20261018.md) |
| 迁移至100GB服务器 | [原视频按需解码方案](docs/server_100gb_20261018.md) |
| 了解本次清理 | [整理记录](docs/workspace_cleanup_20261020.md) |

## 本地开始

在仓库根目录执行。依赖固定于 Python 3.10 / PyTorch 2.7.1+cu128；数据和权重需另行准备。

```bash
uv sync --locked
.venv/bin/python -m ttnet.cli list
.venv/bin/python -m ttnet.cli train \
  --config configs/experiments/event_tcn_reference.json --dry-run
```

服务器遵循比赛固定环境，使用 `configs/server/`，不以本地 uv 环境升级比赛依赖。100GB服务器采用原视频解码；本机已有的233GiB NPY帧库继续保留。

## 研究代码与赛事代码

研究实现位于 `ttnet/src/`，方案位于 `configs/`，轻量实验记录位于 `experiments/`；大产物写入 `out/`，完整结果从 `result/` 浏览。项目内的 `ttnet` 是历史包名，自建定位器为 Residual FPN。

`pingpang/` 是乒乓球赛题框架，`basketball/` 是可选附加题。各自 `participant/` 是允许修改的代码区，`core/` 与框架脚本受保护。根目录 `run_all.sh` 顺序运行两题，`Dockerfile` 将当前根目录复制到 `/participant`。

当前 `pingpang/participant/solution.py` 为 YOLO + 落点规则。官方模型的本地 ORT 运行与环境自检见[本地环境说明](docs/local_environment.md)；该路径的成绩与研究 TCN 分开报告。

提交镜像基于 `sport-base:v1`，不升级比赛的 torch / numpy / onnxruntime / tensorrt。接口、评分与提交要求见 [TaskDescription.md](TaskDescription.md)，实验记录要求见 [AGENT.md](AGENT.md)。
