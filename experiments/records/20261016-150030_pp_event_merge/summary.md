# PP 粗事件合并记录

状态：合并与验证完成，未启动训练。

数据版本：`pp_ball_events_v1`；dataset_id：`c96cb96db39b940b2b41f3e990e723a421b30ecdd24e262a1626a4e15f1af2d6`。

| 集合 | 视频 | 帧 | 触台正帧 | 显式负帧 | ignore | Net正标记（监督屏蔽） |
|---|---:|---:|---:|---:|---:|---:|
| train | 114 | 17398 | 613 | 150 | 41 | 30 |
| val | 40 | 6292 | 203 | 0 | 2 | 10 |

全部23,690行位置字段、原始帧号/PTS、冻结划分与时序窗口一致；共用原数组库。302份来源文件（含重复导出与manifest）核对后按154视频合并。
79项自动测试通过；154视频首/中/尾共462真实定位样本的输入、mask、时间差、位置目标逐值对齐。
未标注时刻为unknown；所有contact_xy为空，球心仅作独立参考。训练13帧、验证14帧Bounce缺少球心参考。

待核验：02_004导出为空；22_002:287重复Bounce已去重；24_006:37的ignore优先于racket_hit。验证没有显式负样本，不能视作完整连续事件验证集。

[数据与格式说明](../../../docs/pp_event_merge.md) · [合并数据](../../../data/datasets/pp_ball_events_v1/README.md) · [标签叠加预览](../../../out/results/20261016_pp_event_merge/label_merge_preview.jpg)
