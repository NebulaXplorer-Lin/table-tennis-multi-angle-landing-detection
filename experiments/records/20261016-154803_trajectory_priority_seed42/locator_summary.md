# 冻结定位器 A/B 轨迹关联对照

同一固定 PP 验证集、定位权重、原图像素口径；只改变轨迹组件。GT 仅评分。

| 指标 | A 独立解码 | B 多候选关联 |
|---|---:|---:|
| 球心 F1 (10px) | 85.08% | 87.79% |
| 球心 F1 (50px) | 86.06% | 89.41% |
| 任一 Local ROI 覆盖率 | 87.21% | 93.93% |
| 选中 Local ROI 覆盖率 | 87.21% | 90.87% |
| 错误跳转次数（GT 相对位移偏差 >50px） | 825 | 534 |
| 最长连续定位漏检（50px，帧） | 22 | 22 |
| 最长连续无有效观测（帧） | 4 | 3 |
| 触台 ±1 帧球心 F1 (50px) | 87.16% | 93.49% |

恢复与退化逐帧清单：`paired_changes.jsonl`。视角、分辨率和逐视频统计：`stratified.json`。
A/B 的 `index.html` 提供覆盖全部评分错误的完整叠加视频。完整视频 FPS 单列于 `runtime.json`，不采用含叠加编码的诊断耗时。

此结果仅判断定位轨迹组件的收益；事件收益仍需同协议训练 A/B 后评估。

## 完整视频速度

RTX 5070 Ti，PyTorch CUDA AMP；40个原视频共6292帧，包含原视频解码、定位、相同随机初始化TCN与JSON序列化，不含初始化、磁盘写入及画图。

| 方案 | 总FPS | 峰值显存 MiB | 额外轨迹未来等待 |
|---|---:|---:|---:|
| A | 97.78 | 48.3 | 0帧 |
| B | 88.29 | 48.3 | 0帧 |

## 分视角球心 F1（50px）

视角为粗视觉分组。

| 视角 | A | B |
|---|---:|---:|
| front | 87.95% | 90.53% |
| diagonal | 85.83% | 89.83% |
| side | 85.83% | 88.97% |

50px评分中B恢复299帧、退化114帧。最长连续定位漏检仍为22帧；漏检段数量404→270，但剩余漏检段P90为3→5帧，仍有持续错误需要查看。当前支持继续训练B验证事件收益，不能据此宣布最终模型替换。

已核对80个完整叠加视频帧数和最后一帧可解码，A的809个与B的658个10px评分错误均有有效视频索引。

[A错误视频](/home/radno/projects/pingpong/out/results/20261016-154803_trajectory_priority_seed42/A/index.html) · [B错误视频](/home/radno/projects/pingpong/out/results/20261016-154803_trajectory_priority_seed42/B/index.html) · [恢复/退化样例](/home/radno/projects/pingpong/out/results/20261016-154803_trajectory_priority_seed42/paired_examples.jpg) · [逐帧变化](/home/radno/projects/pingpong/out/results/20261016-154803_trajectory_priority_seed42/paired_changes.jsonl) · [机器可读对照表](/home/radno/projects/pingpong/out/results/20261016-154803_trajectory_priority_seed42/comparison.csv)

历史全量候选诊断为CPU FP32，本轮为CUDA AMP，两者不是数值对齐比较：6292帧中4个有效观测坐标差异超过1px，错误数808→809。本轮A/B使用相同CUDA AMP后端，历史A0逐值兼容另由同后端测试保证。
