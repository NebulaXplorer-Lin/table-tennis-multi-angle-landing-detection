# PP 球位置与粗事件标注合并

2026-10-16 已将项目根目录 `event_annotations/` 合并到原固定球定位划分，输出为 `data/datasets/pp_ball_events_v1/`。原始位置数据、人工导出文件保留；新版本复用原 NPY 帧库。

| 集合 | 视频 | 全部帧 | Bounce 正帧 | 明确 Bounce 负帧 | ignore 帧 | Net 标记正帧 |
|---|---:|---:|---:|---:|---:|---:|
| train | 114 | 17,398 | 613 | 150 | 41 | 30 |
| val | 40 | 6,292 | 203 | 0 | 2 | 10 |

所有视频、原帧号、真实时间戳、球框、球心、位置标签状态与时序窗口沿用原数据。train/val 清单逐字节一致，比赛组不跨集合。标注按原始帧号直接加入，没有重采样、平移、插值或扩张到邻近帧。

## 事件语义

- `table_contact` / `table_bounce` / `touch_table` / `table` 统一为 `bounce`。
- `net_contact` / `net hit` / `touch_net` / `net` 统一为 `net`；标签保留，Net 监督按本轮既定规则继续屏蔽。
- `racket_hit`、`floor_bounce` 保留原类别，只在明确标出的该帧提供 Bounce=0。缺少负标记的帧保持 unknown。
- `ignore`、`uncertain_contact` 屏蔽该帧全部事件监督；不会屏蔽原有球位置标签。
- 同帧同时标 Bounce 与挥拍/落地时保留冲突记录并屏蔽 Bounce；同帧重复同标签合并。XML、ZIP、JSON 多份导出必须内容一致，否则导入失败。

逐帧 `labels/*.jsonl` 在原记录基础上增加 `event_annotation`：

```json
{
  "kinds": ["bounce"],
  "raw_labels": ["table_contact"],
  "class_order": ["bounce", "net"],
  "status": ["positive", "unknown"],
  "target": [1.0, null],
  "supervised": [true, false],
  "quality": "coarse_manual",
  "issues": []
}
```

未知类别的 target 为 null。后续训练读取时必须先用 `supervised` 屏蔽，再填充数值；不能把 null 变成确定负例。`quality=coarse_manual` 保留本批标注的粗粒度属性。

`events/*.jsonl` 只保存有明确标签的帧，同时给出原时间戳、原始标签、统一类别、监督 mask、问题标记、`contact_xy` 与 `ball_center_reference`。本批没有人工接触坐标，所以 `contact_xy` 全部为 null；球定位标签只放在独立参考字段。训练集 13 帧、验证集 14 帧触台标记连原球心参考也缺失，事件证据仍保留。

## 产物与复现

```text
data/datasets/pp_ball_events_v1/
  splits/train.jsonl、val.jsonl       # 与原划分逐字节相同
  labels/<视频>.jsonl、train/val.jsonl # 原位置字段＋逐帧事件
  events/<视频>.jsonl、train/val.jsonl # 显式事件帧
  windows/                           # 原时序窗口
  frame_store.json                   # 引用既有数组库
  metadata/event_policy.json         # 类别映射与 mask 规则
  metadata/event_sources.json        # 全部来源及 SHA256
  metadata/event_merge.json          # 数量、空标注、重复与冲突审计
  metadata/merge_verification.json    # 位置/划分/窗口不变性验证
  README.md
```

首次合并及重新验证：

```bash
.venv/bin/python -m ttnet.data_tools.merge_pp_events \
  --source event_annotations \
  --ball-dataset data/datasets/pp_ball_v2 \
  --output data/datasets/pp_ball_events_v1

.venv/bin/python -m ttnet.data_tools.merge_pp_events --verify-only \
  --output data/datasets/pp_ball_events_v1
```

合并命令拒绝覆盖已存在的数据版本。新增或修订人工标注时使用新输出目录；数据身份覆盖原位置标签、冻结划分、全部事件来源、合并规则及实现身份。原视频内容、分辨率、帧数、frame step、JSON 与 CVAT tag 一致性都在写入前检查。

`configs/datasets/pp_ball_events_v1.json` 描述这个沿用球定位格式的新版本；原训练默认配置不变。现有 `BallDataset` 可通过显式 dataset 路径读取它，球定位张量与旧数据一致。这个版本是粗事件合并层，尚不符合 `events.continuous` 的完整核验区间格式。

## 已记录的数据缺口

1. `02_004` 有导出文件和正确的 101 帧元信息，但没有图像条目或事件标签；整段事件保持 unknown。
2. `22_002:287` 的重复 `table_bounce` 已去重。
3. `24_006:37` 同时为 ignore 与 racket_hit，保留两项记录，事件监督由 ignore 屏蔽。
4. 验证集中没有显式挥拍/落地负标签，大部分时刻未标注；不能据此做完整连续误报率、阈值或 checkpoint 选择。后续需补核验区间和负样本，再接四组公平训练。
5. 视角与人工接触坐标未推断，Net 标签保留但未启用监督。

本轮仅合并与检查，没有启动正式训练。检查覆盖全量 23,690 行原位置字段、划分和窗口一致性，并以各视频首/中/尾真实样本对齐定位输入；另有标签叠加预览供人工检查。


## 用户确认后的训练解释（2026-10-16）

用户随后确认：触台已标全，主要漏了挥拍／落地等负样本标签。因此在原始合并层保持不变的基础上，生成 `data/annotations/pp_bounce_confirmed_v1.json` 供本轮训练使用：除ignore和空导出视频外，未标出的Bounce为负类。Net仍屏蔽。空间评分使用事件帧已有球心GT参考，缺参考的27个事件只参与时间评分，不补造接触坐标。当前默认四组配置已指向该确认版。
