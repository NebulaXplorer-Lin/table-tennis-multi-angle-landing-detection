# 数据准备与训练指南

更新：2026-10-20。先读 [Quick intro](quick_intro.md) 理解模型和评估口径。本页命令在仓库根目录执行；训练示例会新建实验，不覆盖保留版本。

## 1. 环境与文件准备

本地开发环境固定在 `pyproject.toml` / `uv.lock`，Python 3.10、PyTorch 2.7.1+cu128：

```bash
uv sync --locked
.venv/bin/python scripts/check_local_env.py
.venv/bin/python -m ttnet.cli list
```

自检依赖本地模型文件和 GPU；缺少权重时先完成迁移。详细说明见[本地环境](local_environment.md)。比赛服务器进入官方 `sport-base:v1` 环境，以下 `.venv/bin/python` 换成环境内的 `python3`；比赛依赖维持固定版本，上机检查 GPU 前后向与恢复。

组员应取得三类文件：

| 内容 | 位置 | 获取方式 |
| --- | --- | --- |
| 代码、方案、轻量记录 | Git 工作区 | 检出代码；当前未提交修改由源码迁移包携带 |
| 原视频、标签、固定划分与数据描述 | `data/` | 组内迁移包和 rsync 清单；原视频也可按下节准备 |
| 成对定位权重、旧桌框与 TCN 权重 | 本机历史产物；服务器 `weights/reference/` | `reference_weights.tar.gz`，附哈希清单 |

不要复制 `.venv`。Git 不包含视频、权重和人工标注；`prepare_data.py` 也不会恢复人工事件标注或全部训练描述。

## 2. 数据准备：推荐迁移已有版本

已有数据的工作机可以生成最新源码/参考权重快照及数据白名单；命令输出一个新目录：

```bash
python3 scripts/package_server_transfer.py --profile source-video
```

按输出目录里的 `SHA256SUMS` 校验小包，将 `code.tar.gz` 与 `reference_weights.tar.gz` 解压到服务器同一工作目录。两包都含 `pingpong/` 前缀。`data_stage1.files0` 携带 PP 原视频、标注、数据描述和确认版事件；`data_stage2.files0` 携带 OpenTTGames。按[迁移说明](server_migration_20261018.md)传输，100GB 机器以[原视频方案](server_100gb_20261018.md)为准，不使用旧的 267GiB 白名单。每次代码整理后重新打包，旧包是历史快照。

服务器收到数据后建立视频索引，发布仅在服务器或新数据副本上执行：

```bash
python3 -m ttnet.data_tools.video_store --dataset data/datasets/pp_ball_v2 --publish
python3 -m ttnet.data_tools.video_store --dataset data/datasets/pp_ball_events_v1 --publish
python3 -m ttnet.data_tools.video_store --dataset data/datasets/openttgames_event25_v3 --publish
python3 -m ttnet.cli data verify --dataset data/datasets/pp_ball_v2 --full
python3 -m ttnet.cli data verify --dataset data/datasets/openttgames_event25_v3 --full
```

先只传 PP 时，OpenTTGames 两条命令留到第二批数据到达后执行。PP 应为 154 视频、23,690 帧；OpenTTGames 为 12 视频、28,883 个已准备索引帧。后者是稀疏索引，不是长视频的全帧数。

本机已有 NPY 帧库时直接校验和使用，不执行 `--publish` 切换后端。100GB 服务器使用原视频索引并关闭图像磁盘缓存，不运行 `data migrate` 或 `--prepare-training` 展开整套 RGB。完整源视频与描述约 33.74 GiB，运行空间预算约 46 GiB；实际可用空间和吞吐按上机结果核对。

查看一张真实标签帧对应的原图，核对帧号与尺寸：

```bash
.venv/bin/python -m ttnet.cli data preview \
  --dataset data/datasets/pp_ball_v2 --video 02_001 --frame 8 \
  --output out/preview/02_001_8.png
```

### 只有原始压缩包时

将 `pp_train_data.zip`、`pp_public_data.zip` 放到根目录或 `data/_archives/`：

```bash
python3 scripts/prepare_data.py --dry-run
python3 scripts/prepare_data.py
python3 scripts/prepare_data.py --check-only
```

默认还下载全部 OpenTTGames 源视频与标注；只准备 PP 可加 `--datasets pp_train_data pp_public_data`。显式来源参数与 SHA256 校验见[原数据说明](prepare_data.md)。

下一步优先取得组内已冻结的 `data/datasets/` 与 `data/annotations/`。若必须重建，旧图像路径为 `ttnet.data_tools.prepare` 和 `ttnet.events.prepare`，会写出大量图片；事件原始导出再通过 `merge_pp_events` 合并，按既有完整 Bounce 确认生成训练版。它们不构成一个适用于 100GB 机器的无缓存一键重建流程。缺人工导出或确认版时应先补齐来源，不能用空标签启动训练。

## 3. 数据约束与版本核对

- `pp_ball_v2` 用于球定位，固定 114/40 视频，训练/验证清单哈希前缀分别为 `2a09b6f47b95ff93` / `badc1fc0c1f8f002`。PP 源场景组也隔离，不能仅以相同 seed 代替清单一致。
- `pp_ball_events_v1` 保留人工粗事件及 unknown；实际事件训练配置指向 `data/annotations/pp_bounce_confirmed_v1.json`。该版仅在确认完整区间将未标 Bounce 当负类，ignore、空导出保持 unknown，PP Net 屏蔽。
- OpenTTGames `game_1–4` 训练、`game_5` 验证、`test_1–7` 外测；真实 PTS 采样到 25fps。原生帧号、source frame id、数组行号分别记录。
- 球位置缺失不等于无球；桌框缺失不等于无桌。PP 事件帧球心不是人工接触点。
- 标签修改建立新数据版本及修订记录，重建相关监督/特征缓存；前端权重、读取后端或实现身份变化后也不复用旧特征身份。

数据规模与证据见[训练白皮书](training_whitepaper_20261018.md)，事件补标细节见[合并说明](pp_event_merge.md)。

## 4. 球定位：先 Global，再 Local

本机用 `configs/experiments/ball_fixed3.json`。100GB 服务器用 `configs/server/ball_compact.json`，`resize_cache=false`、`local_crop_cache=false`、workers=2。以下先以本机配置展开参数并做预检：

```bash
.venv/bin/python -m ttnet.cli train \
  --config configs/experiments/ball_fixed3.json --stage global --dry-run
.venv/bin/python -m ttnet.cli train \
  --config configs/experiments/ball_fixed3.json --stage global --smoke
```

正式 Global 新建运行目录：

```bash
global_run="$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)_global_fixed3"
.venv/bin/python -m ttnet.cli train \
  --config configs/experiments/ball_fixed3.json --stage global \
  --run "out/experiments/$global_run"
```

Local 显式使用刚训练的 Global；同一终端下执行，先展开再移除 `--dry-run` 启动：

```bash
local_run="$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)_local_fixed3"
.venv/bin/python -m ttnet.cli train \
  --config configs/experiments/ball_fixed3.json --stage local \
  --global-checkpoint "out/experiments/$global_run/checkpoints/best.pt" \
  --run "out/experiments/$local_run" --dry-run
```

默认配方：AdamW lr=3e-4、weight decay=1e-4、effective batch=8、micro batch=2、最多20轮、AMP；Global 512×288、Local 256×256。水平翻转和亮度增强在三帧内共享，Local 训练逐步由 GT 抖动裁剪过渡到预测 ROI；验证仅使用预测。具体 early stopping 与选优设置以展开配置为准。

训练后分别评估两个阶段；Local 评估需指定正确的 Global 权重：

```bash
.venv/bin/python -m ttnet.cli evaluate --run "out/experiments/$global_run"
.venv/bin/python -m ttnet.cli evaluate --run "out/experiments/$local_run" \
  --global-checkpoint "out/experiments/$global_run/checkpoints/best.pt"
```

完整 `--stage all` 还包含定位导出和测速，首次调试优先分阶段运行。定位器 ONNX 导出不代表整个事件系统已可部署。

## 5. 事件训练：复现当前完整基线

本机新配置 `configs/experiments/event_tcn_reference.json` 对应三种子主基线的结构与配方，复用既有成对定位器和旧桌框。`event_comparison_B.json` 是更早的无桌框基线，二者不要混用。

```bash
.venv/bin/python -m ttnet.cli train \
  --config configs/experiments/event_tcn_reference.json --dry-run
.venv/bin/python -m ttnet.cli train \
  --config configs/experiments/event_tcn_reference.json --smoke

event_run="$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)_tcn_reference_seed42"
.venv/bin/python -m ttnet.cli train \
  --config configs/experiments/event_tcn_reference.json \
  --run "out/experiments/$event_run"
```

服务器改用 `configs/server/event_reference.json`，从 `weights/reference/` 读取 Global、Local 和旧桌框。迁移包提供五个参考权重；该配置不需要本机归档软链接。

如果前端已重训，需同时覆盖成对定位权重，并重新训练/核验与新 Global 空间特征匹配的桌框头；不能把旧桌框头随意搭配新编码器。单次超参覆盖例如 `--set seed=43`，不直接修改共享组件默认值。嵌套对象合并、列表整体替换，未知参数报错。

冻结训练先按预测前端生成描述符，再训练事件头；缓存身份包含权重、数据、轨迹和实现。PP/OpenTTGames 两来源各50%，AdamW lr=3e-4、effective batch=8、AMP、最多30轮、patience=5；完整 PP 内部验证选择最佳 epoch 和 Bounce 阈值，Net 阈值默认0.5。改 seed=43/44 时保持清单、采样协议与选型规则一致。

100GB 原视频后端从完整视频顺序运行关联，只保留需要的特征，不展开第二套完整 RGB。首轮先测数据等待、完整训练步、验证和显存，再安排排期。

### 完整验证与恢复

新事件运行结束后：

```bash
.venv/bin/python -m ttnet.cli evaluate --run "out/experiments/$event_run" --dry-run
.venv/bin/python -m ttnet.cli evaluate --run "out/experiments/$event_run"
.venv/bin/python -m ttnet.cli results
```

已有评估目录不覆盖。对保留 checkpoint 另做复核时使用新输出目录：

```bash
.venv/bin/python -m ttnet.events.comparison evaluate \
  --checkpoint result/tcn_frames_lr0.0003_d0.1/best.pt \
  --annotations data/annotations/pp_bounce_confirmed_v1.json \
  --output "out/results/$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)_tcn_review" \
  --scheme tcn_reference --device cuda --use-checkpoint-threshold
```

上面的保留 checkpoint 携带原本机路径，适用于当前工作区；迁移后按实际数据路径核对配置，优先用服务器参考配置建立新运行。恢复一个新实验须沿用其原始配置，不加 `--set`：

```bash
.venv/bin/python -m ttnet.cli train --run "out/experiments/$event_run" --resume --dry-run
# 确认仍是同一实验后，移除 --dry-run 执行恢复。
```

每次 attempt 的启动日志在 `out/launches/<run_id>/attempt_*/`；训练内部状态、best/last、优化器与逐轮日志在 `out/experiments/<run_id>/`。历史 run.json 可能停留在训练结束状态，完整评分以 `evaluations/continuous/metrics.json` 为准；整理后的索引优先读取评分证据。

## 6. 如何判断一次实验完成

内部评估固定 40 视频完整推理，原帧号 ±1、原图 50px、最大基数一对一匹配，三视角 F1 宏平均。同分时优先 FP 更少。验证有空间参考的189个事件进入空间评分；14个缺坐标事件单列并按既有协议忽略匹配附近的未匹配预测。OpenTTGames 稀疏分类另表报告。

保存生效配置、代码/数据/权重身份、实际命令、种子、best/last、完整逐视频指标，以及覆盖全部错误的 pred/GT 叠加视频与错误索引。不同阈值、后端和 checkpoint 各自登记，缺项就明确标未完成。

内部选型锁定后再外测，不依据 public/test 反复调参；PP public 已被历史实验查看。最终在比赛镜像测包含解码到输出的完整 FPS，目标 ≥50；初始化、写盘和未来帧等待单列。

日常复现无需运行整个 `ttnet.scene.study` 或 `fast_refine` 搜索脚本；它们会启动多组训练。新独立桌框与少量 FT 的入口待实现，按[后续计划](next_steps.md)逐项验收。
