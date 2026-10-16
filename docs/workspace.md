# 工作区使用说明

当前工程使用共享实现与显式方案配置。先查看 [方案与实验索引](../experiments/index.md)，模型改进路线见 [后续计划](next_steps.md)。

## 从哪里修改

| 内容 | 入口 |
|---|---|
| 三帧 Global / Local 球定位 | `configs/experiments/ball_fixed3.json` |
| 固定定位器、训练TCN事件头 | `configs/experiments/event_tcn_frozen.json` |
| 定位器与TCN联合微调 | `configs/experiments/event_tcn_joint.json` |
| 定位尺寸、三帧模式和裁剪视野 | `configs/components/ball_fixed3.json` |
| 定位训练超参数 | `configs/components/ball_train.json` |
| 事件训练超参数与初始化权重 | `configs/components/event_train.json` |
| 数据位置及固定划分 | `configs/datasets/` |
| 历史配置与YOLO入口说明 | `configs/legacy/README.md` |

实验文件引用组件默认值，再用 `parameters` 覆盖；命令行 `--set` 最后覆盖。嵌套对象合并、列表整体替换，未知键和未实现组件会报错。修改共享组件会影响所有引用它的新实验；单次对照优先修改实验文件或使用 `--set`。

```bash
# 列举可运行方案
.venv/bin/python -m ttnet.cli list

# 只查看展开后的参数、数据、输出位置和实际命令；不会启动训练或创建实验
.venv/bin/python -m ttnet.cli train \
  --config configs/experiments/event_tcn_frozen.json \
  --set training.lr=0.0001 --set seed=42 --dry-run

# Global 或 Local 可以单独运行；all 会依次执行定位训练、评估、导出及测速
.venv/bin/python -m ttnet.cli train \
  --config configs/experiments/ball_fixed3.json --stage global --dry-run
.venv/bin/python -m ttnet.cli train \
  --config configs/experiments/ball_fixed3.json --stage local \
  --global-checkpoint ttnet/output/20261014-103011_local_fixed3_ball/weights/global_best.pt --dry-run
```

确认配置后移除 `--dry-run` 才会执行。`--smoke` 用于显式Global/Local阶段或单个事件方案的小样本检查。Transformer 和轨迹关联已接入 `event_comparison_A/B/C/D` 配置；新对照须先提供已复核 PP 连续标注，详见 [运行说明](event_comparison.md)。

## 数据与缓存

- 原视频与标注保留在 `data/pp_train_data`、`data/pp_public_data` 和 `data/ttnet`。
- 当前数据描述位于 `data/datasets/pp_ball_v2` 和 `data/datasets/openttgames_event25_v3`；标签、固定划分、真实时间戳和逻辑数据身份沿用历史版本。
- 每个描述目录的 `frame_store.json` 指向 `data/cache/frames/<storage_id>/`。每视频一个目录，每块最多32帧，NPY形状为 `[N,H,W,3]`、RGB uint8，读取采用只读mmap。
- `frame_ids.npy` 是真实帧号，`source_frame_ids.npy` 是源视频帧号，`timestamps.npy` 是真实秒数。事件数据是稀疏帧，数组行号不等于视频帧号。
- 数据描述内原有 `frame_path` / `path` 字段保留为历史来源记录；实际读取统一由 `FrameStore` 根据视频和帧号定位。
- `data/cache/inputs` 保存按模型输入尺寸生成的数组，增强与归一化在读取后执行；默认不一次生成所有可能的尺寸和裁剪。
- `data/cache/features` 保存与定位权重、精度、采样及提取实现绑定的浮点视觉特征。历史NPZ是模型描述符，不是重复画面；旧缓存归档，身份变化后不会静默复用。

两套当前有效数据的原分辨率画面共52,573帧，数组约233 GiB；PP 位置版与事件合并版共用同一帧库。2026-10-16 已校验并删除归档中的重复 PNG/JPEG、旧缩放输入和重复 PP 压缩包，释放约49.6 GiB，详见 [数据清理记录](data_cleanup.md)。兼容历史数据入口通过 `frame_store.json` 复用这套 NPY。未来重新解码视频应记录新的像素/解码身份。

```bash
# 查询存储结构，或完整校验块文件SHA256
.venv/bin/python -m ttnet.cli data verify --dataset data/datasets/pp_ball_v2
.venv/bin/python -m ttnet.cli data verify --dataset data/datasets/pp_ball_v2 --full

# 从数组导出一帧用于检查；frame是视频实际帧号
.venv/bin/python -m ttnet.cli data preview --dataset data/datasets/pp_ball_v2 \
  --video 02_001 --frame 8 --output out/preview/02_001_8.png

# 查询迁移清单；migrate支持断点续跑，并校验已完成块
.venv/bin/python -m ttnet.cli data plan
```

新机器先按原有 `scripts/prepare_data.py` 准备源视频与标注。旧帧准备工具仍作为历史兼容入口保留；`data migrate` 支持从已有帧库读取，不依赖已清理的旧图片。它不是自动下载或全量解码所有OpenTTGames视频的命令。

## 实验、结果与历史兼容

- `experiments/records/<run_id>`：实际生效的 `config.yaml`（JSON语法）、配置来源、状态及轻量评估记录。
- 各记录目录的 `history/` 保留原先纳入Git的运行文件原文；迁移对应关系在 `experiments/legacy_record_paths.json`。实际图册请从统一索引或记录的主摘要进入归档报告。
- `out/experiments/<run_id>`：权重、优化器状态、epoch日志、预测、错误图册和评估文件。
- `out/launches/<run_id>/attempt_*`：统一入口启动的stdout/stderr及退出状态。
- `out/results`：定位最终发布包；事件发布包位于其 `events` 子目录。
- `experiments/index.md`：统一浏览入口；`workspace_comparison.csv` 是该页面的表格数据源。历史 `comparison.csv` 保留，不改写原协议与数值。
- `archive/<批次>`：历史运行目录、标签、结果快照及整理前源码备份。重复训练图片和旧缩放缓存已清理，兼容数据入口复用现有 NPY；权重和报告仍保留。

```bash
# 从实验checkpoint评估 / 导出定位模型 / 测速
.venv/bin/python -m ttnet.cli evaluate --run out/experiments/<run_id> --dry-run
.venv/bin/python -m ttnet.cli export --run out/experiments/<ball_run_id> --dry-run
.venv/bin/python -m ttnet.cli benchmark --run out/experiments/<run_id> --dry-run

# 刷新索引只读历史结果，不重训、不推理、不重新发布历史快照
.venv/bin/python -m ttnet.cli results

# 恢复之前先检查展开结果；恢复不接受--set改变训练方案
.venv/bin/python -m ttnet.cli train --run ttnet/runs/<paused_run_id> --resume --dry-run
```

事件整套网络的ONNX导出目前未实现；定位器ONNX导出保持原有数值对齐检查。历史暂停实验保留原始数据路径、checkpoint和运行编号，实际恢复仍需显式执行。

赛事默认 `pingpang/participant/solution.py` 仍为YOLO与轨迹规则；研究工程尚未替换提交入口。受保护框架和固定赛事运行环境保持原样。

## 验收记录

迁移状态及逐像素核对汇总：`out/workspace/frame_migration.json`；模型、checkpoint与读取验证：`out/workspace/validation.json`；归档映射和历史文件校验：`out/workspace/current.json`。

本轮轻量验收记录：[workspace_migration.json](../experiments/workspace_migration.json)。60项测试通过，两套赛事框架完整性校验通过；真实样本CPU FP32对比的最大输出差异为0。

64帧固定随机样本的单进程读帧检查（含遍历全部RGB字节，OS缓存未控制）：PP图片约72.8帧/秒、NPY约234.6帧/秒；事件JPEG约184.5帧/秒、NPY约155.8帧/秒。该结果只描述本次读帧检查，不是训练吞吐或完整模型FPS。
