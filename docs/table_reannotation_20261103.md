# 台面标注修订 v1（2026-11-03）

[查看新旧框对照](../out/results/20261103_table_reannotated_v1/index.html) · [六帧对照总图](../out/results/20261103_table_reannotated_v1/before_after.jpg) · [修订后完整视频](../out/results/20261103_table_reannotated_v1/04_004_reannotated_labels.mp4)

已发布独立数据版本 `data/datasets/pp_table_reannotated_v1`，修正 **04_004 第115–120帧共六个桌框**。六帧均在原验证集，保持验证身份；本版不增加训练样本，训练集桌框标签未改变。

后续[训练集专项核查](train_table_audit_20261103.md)已发现14_002的台面边界错误以及13组的可见台面缺标。它们尚未包含在本v1中，不能将v1发布视为训练标签已清理完成。

## 标注口径与范围

重标目标为主比赛球台的**台面及薄桌沿的轴对齐外包框**，不以桌腿、底座、独立球网或网柱扩张边界。人物遮挡台面时，结合原图及同镜头邻近帧确认整体外缘。六帧的镜头和台面位置稳定，经逐帧查看后采用同一整数像素框 `[310, 286, 862, 325]`；不是对整段视频自动套中位框。

修订由 **Codex 根据图像目视完成**，尚未经过独立人工复审。遮挡区域的边界依赖相邻帧判断，不等同于精确桌面分割或桌面四角标定。

原始全量标注的几何扫描基础上，额外查看13个与本视频中位框偏差较大的代表案例（最小IoU<0.65）。08_004、13组、15_002等存在真实构图变化；其他案例未确认错选到地板等明确目标错误，保留原框，细微边界仍可继续审核。图册抽样不代表全部23,690帧已经人工精修。本版只修正已确认的六个错框。

## 数据组成

| 集合 | 视频 | 全部帧 | 有效桌框 | unknown | 本版修改 |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 114 | 17,398 | 17,281 | 117 | 0 |
| val | 40 | 6,292 | 5,101 | 1,191 | 6 |

新版本的 `splits/train.jsonl`、`splits/val.jsonl` 与基线逐字节相同，球、球员等非桌框字段保持一致；原始标注与图像缓存保留。缺标仍为unknown，不生成确定的无桌负例。

```text
data/datasets/pp_table_reannotated_v1/
  annotations/                154份原格式JSON，只改六个table bbox
  labels/train.jsonl          桌框训练逐帧记录及unknown状态
  labels/val.jsonl            桌框验证逐帧记录及修订来源
  splits/train.jsonl、val.jsonl
  frame_store.json            引用已有RGB帧库，不复制画面
  metadata/dataset.json       数据身份、统计、标签/划分哈希
  metadata/revisions.json     原框、新框、原因与标注来源
  metadata/source_annotations.json
```

数据ID：`b0e8814616abec0fb69523d4711a2872172e219b2434db49028a7bb12e1377ee`。

轻量修订清单：[pp_table_reannotated_v1.json](../experiments/data_revisions/pp_table_reannotated_v1.json)。数据目录不随Git传输，可用[构建工具](../scripts/build_table_reannotations.py)从原始数据和修订清单重新生成，工具拒绝覆盖已有版本。

## 训练读取

桌框监督读取支持显式版本；球定位和历史事件配方继续使用各自原数据入口：

```python
from pathlib import Path
from ttnet.scene.table import annotations
labels, audit = annotations('val', dataset=Path('data/datasets/pp_table_reannotated_v1'))
```

既有桌框训练入口支持 `--table-dataset`：

```bash
.venv/bin/python -m ttnet.scene.train \
  --table-dataset data/datasets/pp_table_reannotated_v1 \
  --checkpoint <Global权重> --run <新的实验目录>
```

该入口仍是旧冻结特征桌框头，不代表独立桌面视觉分支已实现。标签版本与哈希会写入训练配置、特征缓存身份和指标；更换版本时旧目标缓存会被拒绝。训练后的叠图读取同版标签，避免新标签成绩与旧标签叠图混用。

本次只生成修订数据和可视化，没有启动模型训练，也没有重算历史模型成绩。后续比较旧模型与新模型时须使用同一修订版验证标签。

## 验证

[校验结果](../out/results/20261103_table_reannotated_v1/validation.json)包括全量原格式字段对比、修改范围、原始文件哈希、训练/验证清单一致性、unknown保留、训练读取、完整视频帧数与PTS检查。相关读取和回归测试另随代码保留。
