# 桌框与自回归实验实现

> 本页保留已完成研究的实现细节；完整版本见[结果首页](../result/README.md)，日常复现见[训练指南](training_guide.md)。以下study命令会执行多组实验，不作为组员首次启动入口。

本轮固定既有PP训练114/验证40视频及事件标签，不引入player。实验根目录：`out/experiments/20261016_table_ar`。未完成的阶段不标记为完成；实时状态见根目录 `status.json` 和各运行 `run.json`。

## 数据与模型

- 桌框训练：Global球定位器冻结，额外返回32通道空间图；4×4空间池化后回归有序归一化角点，L1＋GIoU。缺框不作负类，按验证平均IoU选权重。输入512×288，训练60轮上限、patience10、AdamW 3e-4、batch128、seed42。
- 桌框接口：逐帧记录增加 `table={bbox_xyxy,valid,source}`。特征9维为归一化中心/宽高、球相对桌框位置、桌框/关系mask及观测年龄；当前每帧预测，年龄为0。
- 事件输入：旧头仍接受256视觉＋9运动；`context_head=true`使用256视觉＋18维运动/场景，统一前端视觉96、运动32、场景32，再投影128。`scene_mode=none/box/relative`，`table_source=oracle`仅为诊断。
- 自回归：GRU128，候选含坐标、检测分数、来源、Global局部RGB摘要、场景关系、真实Δt和128维Global视觉。Local候选没有缓存独立外观，统一将该RGB项置0；共享Global视觉仍可用。Top5＋最多2个Local观测＋空观测，合格候选集合概率损失；10px固定训练匹配，unknown屏蔽。
- GRU先teacher forcing5轮，再5轮降到50%真值历史，最多30轮；验证全程预测历史。训练使用固定规则候选集合，完整推理中ROI随自回归历史变化，二者的候选分布差异作为结果限制保留。
- 自回归搜索保留每条路径独立隐藏状态；Global提议评分与Local最终联合选择都使用模型条件概率，最终只累计一次当帧概率。恒速外推只提出ROI，不输出伪观测；每帧保留全局提议重捕获机会。

## 配置与兼容

新增配置：`table_checkpoint`、`table_source`、`context_head`、`scene_mode`、`head_dropout`、`time_encoding`、`event_window_mode`、`ar_checkpoint`、`ar_beam_size`、`ar_use_table`；新增 `tracker_type=autoregressive`。

旧checkpoint保持原参数名及forward输出。新事件checkpoint包含桌框和关联器权重，完整推理不依赖外部预训练文件仍然存在。新增权重/来源/代码哈希进入特征缓存身份。新旧缓存不原地覆盖；复用旧冻结球定位观测时记录来源哈希，并与真实推理进行坐标一致性检查。

`event_window_mode=seconds`按真实PTS取±0.24秒、13个位置，重复或超范围位置mask，输出/评分仍用原视频帧号。两种头共用时间编码；长物理窗口仍保持13个slot与TCN13-slot感受野。

## 实验与恢复

入口：

```bash
.venv/bin/python -m ttnet.scene.train --checkpoint ttnet/output/20261014-103011_local_fixed3_ball/weights/global_best.pt --run out/experiments/20261016_table_ar/table
.venv/bin/python -m ttnet.scene.study --root out/experiments/20261016_table_ar
```

桌框缓存准备由 `scene.cache` 提供；GRU数据准备由 `scene.prepare_ar` 提供。队列保存每步命令、日志及退出码，失败时停止；重新运行队列恢复未完成训练，未完成评估目录重命名保留。最终配置在外部评估前锁定。

矩阵：场景none/box/relative＋oracle；GRU有/无桌框训练；按场景对照选择的输入做beam1/4/8固定事件头诊断及重训；固定最佳轨迹后，两种事件头各做2个窗口×2学习率×2dropout；最终两头及原B完成42/43/44种子。

正式指标以完整视频为准；缓存选模成绩另列。外部仅对锁定seed42最终模型测TTGames test与PP public，阈值不改。后者属于历史已查看迁移集，不称盲测。

## 验证与限制

测试覆盖框坐标反变换、缺失关系、GIoU梯度、上下文mask和权重重载、候选padding、未知监督、beam状态重置及物理窗口。现有事件、候选、恢复训练等回归测试也执行。

第一版桌框不输出额外校准置信度，仅验证有限且有序的框；桌面超薄时相对纵坐标对误差敏感。桌框是图像坐标参考，不作桌面单应变换。GRU的候选选择正确率包含空观测，不等于球定位F1。默认候选有效观测门槛0.7沿用B，学习式评分没有改变这一评估边界。
