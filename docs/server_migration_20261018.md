# 4090服务器训练迁移说明

日期 2026年10月18日

更新：服务器已确认只有100GB总磁盘，当前应使用[100GB迁移方案](server_100gb_20261018.md)与out/migration/20261018_server_100gb。下文267GiB数据方案保留作大磁盘机器的参考，不适用于本次服务器。

建议保留本地完整实验室，在服务器建立一个新的训练工作区。迁移当前源码快照、固定训练数据和少量基线权重，服务器只使用明确的配置入口。无需先大规模删代码或重构，也无需把旧实验运行目录一起搬过去。

## 1 本次检查发现的问题

- 当前存在修改未提交的训练代码，以及未纳入Git的joint、scratch和refinement等模块。仅克隆当前commit会漏掉这些文件。
- 原来的.gitignore用通用datasets和experiments目录规则，连configs/datasets和configs/experiments也被忽略。本次已添加精确例外，后续这些训练配方可以正常纳入版本管理；未自动提交。
- ttnet/data和ttnet/output是指向archive的本机兼容链接。历史配置引用/home/radno/projects/pingpong及旧ttnet/output，迁移时不能依赖这些链接仍然存在。
- 本机pyproject.toml和uv.lock用于5070 Ti的torch 2.7.1及CUDA 12.8环境。比赛镜像是torch 2.1.2及CUDA 11.8，不能在比赛机直接执行uv sync --locked。
- 独立桌框分支、新融合与少量数据微调仍需实现。迁移包提供现有定位和历史冻结事件基线，不能把历史scratch入口当作白皮书新路线的一键入口。

## 2 迁移后的目录

```text
/workspace/pingpong/
├── ttnet/src/                  当前共享训练代码，包含本地未提交模块
├── configs/server/             本次明确的服务器检查入口
├── configs/components/         入口依赖的组件配置
├── configs/datasets/           固定数据集声明
├── configs/experiments/        历史可运行配方，保留供参考
├── scripts/                    数据、打包与检查脚本
├── docs/                       白皮书和本迁移说明
├── data/                       固定数据、标签及FrameStore
├── weights/reference/          成对定位权重及历史事件参考
├── out/experiments/            到服务器后新建的训练结果
└── MIGRATION_SNAPSHOT.json     本次源码内容、哈希和原Git状态
```

源码目录还保留一些实验模块，但入口已隔离。不要通过猜文件名启动scratch.py、fast_refine.py或旧批处理。新增独立桌框方案实现后，为其建立单独配置和run目录，不覆盖历史基线。

本次新建两个配置：

| 配置 | 作用 | 边界 |
| --- | --- | --- |
| configs/server/ball_fixed3.json | 现有Global和Local定位入口 | 使用当前固定数据与fixed3配方 |
| configs/server/event_reference.json | 已有TCN、关联和旧桌框头的冻结基线 | 使用weights/reference，不是新独立桌框方案，也不包含EMA变体 |

后续若重训定位器，应让Local绑定对应Global。旧桌框头也绑定Global哈希，不能随意把新Global、旧Local和旧桌框头拼成一组。

## 3 已准备的迁移材料

打包脚本为[scripts/package_server_transfer.py](../scripts/package_server_transfer.py)。仅使用Python标准库，不训练、不上传、不删除原实验，按白名单读取实际文件，包含被忽略但必要的训练配置。

本次交付目录为out/migration/20261018_server_transfer，包含：

| 文件 | 内容 |
| --- | --- |
| code.tar.gz | 当前源码、配置、文档、数据脚本与两套赛事框架源码；含未提交改动 |
| reference_weights.tar.gz | Global、Local、旧桌框头、原TCN和EMA版TCN的5个参考checkpoint |
| data.files0 | rsync使用的NUL分隔数据文件白名单 |
| data.files.txt | 同一数据清单的可读版本 |
| data_stage1.files0 | PP部分，可先迁移并开展现有球定位训练 |
| data_stage2.files0 | TTGames部分，阶段二事件训练前补齐 |
| manifest.json | 实际文件数、字节数和各数据目录体积 |
| SHA256SUMS | 上述小文件与压缩包的SHA256 |

代码和权重包都以pingpong作为顶层目录，解压到/workspace后得到/workspace/pingpong。权重以原始字节复制到weights/reference并附来源与哈希，不修改checkpoint里的历史配置。

不迁移.venv、.git、archive、旧out/experiments、旧模型特征缓存、TensorRT engine、旧缩放输入、图册和全量结果。源码中的历史记录引用可能在服务器不可访问；正式训练以本说明和白皮书为准。两套赛事源码保留供后续集成，部署权重和公开数据另行按实际需要配置。

## 4 数据推荐直接搬现有准备结果

数据白名单包含PP原视频与原标注、PP public、已审核事件标签、三个数据描述目录、两套FrameStore，以及OpenTTGames训练和测试原视频。它不包含TTGames下载ZIP和逐文件解压的原始标注目录，因为训练和锚点评估所需标签已在数据描述中。

这套清单支持使用现有准备结果训练、从PP标注训练桌框及原视频评估。若要重新导入TTGames原始标注或运行prepare_data.py --check-only，还需另行迁移data/ttnet的对应ZIP和解压标注；缺少这些文件不等于现有训练标签缺失。

两套FrameStore约233GiB，是传输大头。它们保存已选原帧、源帧映射与真实PTS，直接迁移更容易保留数据身份。固定split、标签与事件补标都是必须项，不能只搬MP4再重新随机划分。

推荐分两批：先传约66GiB的PP数据，完成环境预检、Global和Local训练；再传约200GiB的TTGames部分。全部文件合计约267GiB，准确字节数见manifest。这样不必等全部外部事件数据到齐才开始定位阶段；冻结事件头训练前必须补齐第二批。

如果网络传不动233GiB，可以改为在服务器从原视频重建，但这是另一条需要实现和验证的数据准备路径：现有data migrate主要依赖旧ttnet/data入口，不能在不带归档的新机器上直接当作原视频重建命令。需要按已有帧号和PTS清单重解码、生成新像素身份并核对，不能宣称重建前后逐字节相同。本次未实现该替代路径。

## 5 使用SSH和rsync迁移

下面假设SSH别名为train-server，服务器工作目录为/workspace。根据实际机器替换这两处；命令目前没有执行远程操作。两端需有rsync。

先在本地确认迁移材料完整：

```bash
cd /home/radno/projects/pingpong/out/migration/20261018_server_transfer
sha256sum -c SHA256SUMS
```

先传小包与清单。在本地执行：

```bash
ssh train-server 'mkdir -p /workspace/migration /workspace/pingpong'
rsync -rt --partial --info=progress2 \
  /home/radno/projects/pingpong/out/migration/20261018_server_transfer/ \
  train-server:/workspace/migration/
```

在服务器执行以下命令。目标应为本次新建的工作区，避免覆盖另一次实验的代码：

```bash
cd /workspace/migration
sha256sum -c SHA256SUMS
tar -xzf code.tar.gz -C /workspace
tar -xzf reference_weights.tar.gz -C /workspace
cd /workspace/pingpong
python3 scripts/package_server_transfer.py --verify-root .
```

传数据前在本地做一次清单预览；确认目标目录后移除--dry-run执行。同一条命令可重复运行，中断后继续。文件白名单避免跟随本地归档链接，也避免把全部历史输出带过去。

```bash
cd /home/radno/projects/pingpong
rsync -rt --partial --info=progress2 --dry-run \
  --from0 --files-from=out/migration/20261018_server_transfer/data.files0 \
  ./ train-server:/workspace/pingpong/

rsync -rt --partial --info=progress2 \
  --from0 --files-from=out/migration/20261018_server_transfer/data.files0 \
  ./ train-server:/workspace/pingpong/
```

文件清单与相对路径、断点文件选项的语义见[rsync官方手册](https://download.samba.org/pub/rsync/rsync.1)。没有SSH时也可通过平台上传code和weights两包，数据按同一白名单分批传入相对路径；不需要把233GiB重新压成一个巨型包。

分批迁移时，将上述命令的data.files0分别替换为data_stage1.files0、data_stage2.files0，其他路径不变。第一批完成后先验证PP；第二批完成后再验证TTGames。不要对尚在传输的帧库启动训练或全量校验。

## 6 上机先检查环境与数据

使用组委会规定的镜像，保留其torch、numpy、ONNX Runtime和TensorRT版本。重新建服务器环境，不复制本地.venv，不将本机pyproject当作比赛依赖清单。

```bash
cd /workspace/pingpong
python3 -c 'import sys, torch, numpy; print(sys.version); print(torch.__version__, torch.version.cuda); print(numpy.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
python3 -m pip check
```

本地未验证新包在torch 2.1.2镜像上的完整前后向。已知现有train入口使用torch.amp.GradScaler，需要在服务器训练前适配2.1可用的torch.cuda.amp.GradScaler并检查恢复行为，见[PyTorch 2.1 AMP文档](https://docs.pytorch.org/docs/2.1/amp.html)。dry-run通过不会暴露所有运行时API问题，不能跳过真实smoke检查。

先快速检查FrameStore结构，再做一次完整SHA256校验。--full会读取约233GiB，耗时单列，建议正式训练前完成：

```bash
python3 -m ttnet.cli data verify --dataset data/datasets/pp_ball_v2
python3 -m ttnet.cli data verify --dataset data/datasets/openttgames_event25_v3
python3 -m ttnet.cli data verify --dataset data/datasets/pp_ball_v2 --full
python3 -m ttnet.cli data verify --dataset data/datasets/openttgames_event25_v3 --full
```

预期PP为154视频、23690帧，TTGames为12视频、28883帧。核对PP train与val清单哈希、事件标签及抽样overlay，数据身份详见[训练白皮书证据](training_whitepaper_20261018_evidence.json)。

## 7 检查入口再开始训练

本次只验证下面两个现有方案的参数展开；不自动启动训练：

```bash
cd /workspace/pingpong
python3 -m ttnet.cli train \
  --config configs/server/ball_fixed3.json --stage global --dry-run

python3 -m ttnet.cli train \
  --config configs/server/event_reference.json --dry-run
```

比赛版本兼容完成后，可先对现有Global入口做真实smoke：

```bash
python3 -m ttnet.cli train \
  --config configs/server/ball_fixed3.json --stage global --smoke
```

smoke是明确的小规模训练，不是只读检查。检查读图、反向、显存、best/last和日志，再按白皮书配置新建正式run。新服务器训练不要对历史run执行--resume，也不要去掉旧scratch恢复命令的保护后直接运行。

接下来的工程与训练顺序：独立桌框单帧分支及数据复核 → 球定位与桌框质量确认 → 预测特征缓存 → 新TCN拼接基线与融合消融 → 固定少量训练子集在线FT。各阶段的任务、参数、增强、预算与验收见[训练白皮书](training_whitepaper_20261018.md)。

## 8 本地代码继续修改后的同步方式

本次迁移包是固定快照。包生成后修改的代码不自动进入服务器；可以重新执行下面的打包命令，生成新的时间戳目录，或在整理并提交源码及配置后使用私有Git同步。不要反复把本地整个out目录覆盖到服务器。

```bash
cd /home/radno/projects/pingpong
python3 scripts/package_server_transfer.py
```

本次没有自动Git提交、上传或远程启动。若使用Git作为后续主同步方式，需要显式纳入所有新代码和configs/datasets、configs/experiments；数据和权重仍按清单单独管理。
