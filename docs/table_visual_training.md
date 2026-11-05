# 球桌两路线训练入口

更新：2026-11-04。B共享CNN和独立CNN均已完成。早先取消的YOLO路线按用户最新要求重新尝试：[YOLO11s原生训练对照](yolo_table_native_20261104.md)也已完成，24轮早停、best第12轮。统一验证帧均/视频均/薄框IoU为54.73%/71.08%/70.65%；但组22漏检1,652/1,726帧，尚未切换事件前端，没有启动事件头训练。

后续：[冻结Global＋旧MLP配对对照](B_legacy_mlp_control_20261104.md)也已完成，固定初始化、采样与旧MLP配方。B＋MLP验证帧均/视频均IoU为46.37%/36.18%，配对旧Global为39.40%/30.59%；B＋MLP薄框仍仅6.36%，需要继续解决分场景退化。原CNN结果继续保留，以下为原两路线任务与运行方式。

最新：[B＋MLP薄框改进](B_mlp_thin_20261104.md)已完成，合并16×4池化、10%薄框配额及纵向中心/高度损失；B仍冻结。best第12轮验证帧均/视频均48.25%/37.19%，薄框23.53%，但正视降至50.85%、整体IoU≥0.5比例降至45.62%，薄框仍无IoU≥0.5。保留两版MLP与完整分桶结果，尚未启动事件头训练。

新增实验：[独立CNN/FPN＋全局MLP＋调色](independent_mlp_color_20261104.md)已完成，第11轮早停、best第1轮；验证帧均/视频均41.86%/42.89%，薄框43.24%，但总体IoU≥0.5仅18.11%、组22退化。保留权重及完整分桶结果，未替换事件前端。以下原中心热图/CNN路线仅保留作历史对照。

## 原CNN任务（已完成）

- [任务状态](../out/experiments/20261104-095944_table_visual_comparison/task.json)
- [调度日志](../out/launches/20261104-095944_table_visual_comparison/driver.log)
- [B共享分支日志](../out/experiments/20261104-095944_table_visual_comparison/shared_B.log)
- [独立CNN日志](../out/experiments/20261104-095944_table_visual_comparison/independent_cnn.log)（该阶段开始后生成）

单GPU顺序运行：先`shared_B`，完成训练和完整验证后自动启动`independent_cnn`。更换调度器时保留原B训练PID并接管等待，不重启B、不改其权重与训练源文件。YOLO路线在任务JSON中保留取消记录；各活动阶段保存best/last、优化器/scaler/RNG、逐轮采样、指标和错误图，断点从完整epoch恢复。

| 路线 | 配方 | 模型与视觉输出 |
| --- | --- | --- |
| B特征复用 | [shared_B.json](../configs/table/shared_B.json) | 冻结当前B Global；32通道、144×256空间图上预测中心热图和密集bbox；ROI512维+上下文32维。同次Global前向供球与球桌使用 |
| 独立球桌 | [independent_cnn.json](../configs/table/independent_cnn.json) | 单帧512×288→深度可分离卷积16/32/64/96→多尺度融合→32×72×128空间图→轻量中心热图/bbox头；ROI512维+上下文32维 |

独立CNN约5.2万参数，编码器与桌框头从头接受桌框监督，无预训练依赖。它和B路线使用同一种桌框损失与视觉接口；独立CNN学习自己的特征，B路线保持定位器冻结。编码器lr=3e-4，头lr=1e-3；B原配方保持不变。此处原CNN任务中取消的YOLO阶段保留历史记录；新的原生YOLO实验使用独立运行目录与入口。

两者使用`pp_table_reannotated_v1`的固定114/40视频，unknown不作为无桌负例。每轮4000张、最多80轮、patience12、seed42、micro4/effective16、AMP。每轮对全部6292验证帧推理，只在5101个已知框帧评分，缺预测计IoU=0，按视频均IoU选best。薄框、视角、源组与边缘误差分别输出。

## 代码与安全入口

- [数据与采样](../ttnet/src/data_tools/prepare_table.py)：只读标签和RGB存储，生成索引，不复制全量图像。
- [原生CNN](../ttnet/src/models/table_light.py)与[共享头/ROI采样](../ttnet/src/models/table_encoder.py)：保留空间图、确定性选框、4×4 ROI及全图上下文。
- [独立CNN训练入口](../ttnet/src/scene/train_light.py)与[B训练入口](../ttnet/src/scene/train_visual.py)：显式学习率分组、完整验证、epoch断点恢复；默认仅准备，必须传`--train`才更新参数。
- [任务调度](../scripts/run_table_visual.py)：持久化串行任务、子进程退出状态与日志；同一`--task`可继续未完成阶段。
- [历史准备记录](../out/preparation/20261103_table_visual_v1/preparation.json)：原YOLO/B方案的只读前向与数据身份快照，不能当作新CNN的测试或训练成绩。本次按用户要求不追加测试。

后续恢复当前任务：

```bash
.venv/bin/python -u scripts/run_table_visual.py \
  --task out/experiments/20261104-095944_table_visual_comparison --device cuda
```

调度器以文件锁避免重复调度，可接管仍存活且路径匹配的原训练进程。训练配置或依赖代码、标签、源权重改变时，训练恢复入口拒绝继续，应新建实验。

## 事件头准备范围

`configs/experiments/event_table_*.json`提供B定位器F0及B共享/独立CNN各自的geometry/concat/zero/gated配方，共9份。独立配方已改为`independent_cnn`、32通道和512×288输入。当前任务的`event_configs/`已指向实际B/CNN权重目录；待模型完成训练、完整验证与前端锁定后使用。

新增`TableFusionEventHead`保留球视觉256维，单独接收桌面ROI、上下文和mask；新缓存绑定两路定位阈值、权重SHA、桌面特征与几何版本。轨迹JSON只写预测元数据，视觉张量保存在NPZ，避免结果保存失败。

本轮不运行新的测试、过拟合预检或事件训练。新桌框模型精度、在线/缓存完整一致性及端到端FPS须以后续训练产物与验收为准；空间桌框的联合FT和辅助损失路径仍属后续阶段，当前入口明确拒绝该模式。
