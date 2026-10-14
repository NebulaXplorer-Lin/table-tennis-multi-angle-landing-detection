# 乒乓球落点检测项目：整体技术方案 V1.0

技术主线：TTNet-style Global-to-Local Ball Localization + Trajectory Reconstruction + Temporal Event Spotting

结合你已有的数据、YOLO 首轮实验结果，以及最终比赛的输入输出要求，我建议采用 分阶段训练、模块化开发、后期联合优化 的方案。

这套方案借鉴 TTNet 的核心思想，但不直接照搬完整网络，因为原始 TTNet 使用 120 FPS 视频，而你的测试集只有 25 FPS，且尚无反弹事件标注。原始 TTNet 还包含你暂时不需要的触网识别和其他场景分割任务。

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv

+1



## 一、最终技术架构

Input Video

单目视频，25 FPS 测试，不同机位

Global Ball Localization

多帧图像 → 全局球位置 Heatmap

Local Ball Refinement

原图局部裁剪 → 精确球心 + 置信度

Trajectory Reconstruction

轨迹关联、短时缺失修复、速度及变化特征

Visual Features

球、球桌、球拍周边图像

Motion Features

位置、速度、轨迹突变

Temporal Event Spotting

TCN / Temporal Transformer

Table Bounce / Racket Hit / Floor Bounce / None

Geometry & Rule Filtering

球桌区域、目标半台、接触点、时序去重

Final Output

落点帧 + 位置 + 置信度

上图是最终结构，不是要求第一天就把所有组件做完。

我建议项目划分为以下阶段：

| 阶段 | 核心任务                        | 所需标注         | 训练方式     |
| -- | --------------------------- | ------------ | -------- |
| 0  | 数据检查与 YOLO 诊断               | 已有框          | 无需重新训练   |
| 1  | Global Ball Localization    | 球框           | 单独训练     |
| 2  | Local Refinement + Tracking | 球框、视频连续帧     | 分阶段训练    |
| 3  | Event Annotation            | 人工事件标注       | 暂不训练     |
| 4  | Bounce Event Spotting       | 反弹及误判事件帧     | 冻结定位模块训练 |
| 5  | 几何过滤与多任务微调                  | 球框、事件、少量台面角点 | 可选联合训练   |
| 6  | 25 FPS 推理与 Docker 交付        | 无新增标注        | 集成验证     |

核心建议是先获得一个优秀的球心定位器，再训练事件模型。

## 二、数据处理与整体训练策略

### 2.1 数据集应该怎么组织？

你已经拥有逐帧四类 Bounding Box：

- Ball
- Player
- Referee
- Table

但每类标注在项目中的价值并不相同。

| 标注      | 主要用途               |
| ------- | ------------------ |
| Ball    | Global/Local 球定位监督 |
| Table   | 场景区域约束、粗几何信息       |
| Player  | 辅助判断击球方与球拍事件       |
| Referee | 暂时不需要，可以保留供以后扩展    |

球定位模型不需要强制学习四类目标。我的建议是先只训练 Ball Localization，不再将四类检测作为整个系统的共同入口。

这可以避免大量显而易见的球员、裁判检测任务占用你对小球定位问题的实验精力。

### 2.2 如何从 Bounding Box 得到训练标签？

对于每帧球框：

\\[ b_t=(x_1,y_1,x_2,y_2) \\]

转换为球心：

\\[ p_t=\left(\frac{x_1+x_2}{2},\frac{y_1+y_2}{2}\right) \\]

以球心为中心生成 Gaussian Heatmap：

\\[ H_t(x,y)= \exp\left( -\frac{(x-x_t)^2+(y-y_t)^2}{2\sigma^2} \right) \\]

推荐使用 Heatmap Regression，而不是直接回归两个坐标。

但是要注意两种标注情况：

- 如果 Bounding Box 标的是球的实际轮廓，框中心是合理的球心近似。
- 如果 Bounding Box 包含了较长的 Motion Blur 拖影，框中心未必是真实球心，需要抽样检查。

对于标注错误、没有标注球或球确实不可见的帧，要明确区分 `negative`、`unlabeled` 和 `ignore`，不能把所有缺少球框的帧都当成没有球。

### 2.3 跨帧率处理

不要提前把所有视频永久转换成 25 FPS。

保留原始视频和原始帧编号，然后通过时间采样生成模型输入。

以三帧模型为例：

\\[ I_t=[F\_{t-\Delta},F_t,F\_{t+\Delta}] \\]

其中 \\(\Delta\\) 可以按实际时间跨度确定。

| 数据来源        | 推荐处理                         |
| ----------- | ---------------------------- |
| 30 FPS      | 主要训练数据，构造原生相邻帧与近似 25 FPS 的窗口 |
| 60 FPS      | 按时间间隔采样，辅助学习运动模糊及高速轨迹        |
| 25 FPS Test | 使用实际连续帧，保持原始帧编号              |

如果采用非整数时间间隔采样，应选择最近的真实帧并保留其时间戳，而不是简单复制或混合帧。

不建议只保留从 60 FPS 视频抽取的 25 FPS 版本，因为额外的高帧率数据对球定位和事件标注仍有价值。

### 2.4 数据划分

这是一个很容易影响最终成绩的问题。

训练集和验证集必须按视频划分，而不是随机按帧划分。

最好再按拍摄机位分组，保留至少一部分完整的未见机位视频作为验证集。

推荐建立两套验证：

- `val_seen_view`：熟悉机位，但视频内容不同。
- `val_unseen_view`：相机俯仰角或拍摄位置此前未出现在训练中。

最终所有模型选择都应参考 `val_unseen_view` 的性能，否则很可能在训练视频中表现不错，但比赛测试集大幅退化。

## 三、阶段 1：Global Ball Localization

这是当前的最高优先级。

你的 YOLO 目前 F1 只有个位数，因此不建议马上在 YOLO 上继续投入大量训练资源。

但在正式更换模型前，应确认错误不是来自标注、图像缩放或 IoU 评价方式。小目标即使球心预测正确，框偏移几像素也可能导致 IoU 不达标。

### 3.1 模型选择

建议采用 TrackNet-style Backbone + TTNet 的分阶段思想。

不必严格复现原始 TrackNet 或 TTNet。直接用 PyTorch 实现一个轻量 U-Net 风格的多帧 Heatmap 网络，会更方便实验。

3 Consecutive Frames

[B, 9, H, W]

CNN Encoder

多尺度空间特征 + 多帧运动信息

U-Net Decoder

高分辨率 Heatmap Prediction

Ball Heatmap + Presence Score

球心位置 + 是否有可靠可见球

推荐初始配置：

| 参数         | 建议                                        |
| ---------- | ----------------------------------------- |
| 输入帧数       | 3                                         |
| 输入分辨率      | 先实验 960×540，再试 1280×720                   |
| Encoder    | 轻量 CNN / U-Net Encoder                    |
| Decoder    | 多尺度上采样 + Skip Connections                 |
| 输出         | 2D 球心 Heatmap + Presence                  |
| Loss       | Weighted BCE / Heatmap MSE + Presence BCE |
| Optimizer  | AdamW                                     |
| 初始学习率      | \\(3\times10^{-4}\\)                      |
| Batch size | 从 2～4 开始，按显存调整                            |

这是一套实验起始配置，不是论文中直接验证过的最优超参数。

尤其要根据你的实际球框尺寸决定网络分辨率。对于在 960×540 输入中仅剩 1～2 像素的球，应优先尝试更高分辨率、局部裁剪，而不是增加网络深度。

### 3.2 必须做的消融实验

建议一开始就训练以下三个版本：

| 模型 | 输入         | 目的                 |
| -- | ---------- | ------------------ |
| M0 | YOLO 单帧    | 已有 Baseline        |
| M1 | 单帧 Heatmap | 测试 Heatmap 监督是否更适合 |
| M2 | 三帧 Heatmap | 测试时序信息带来的提升        |

M1 和 M2 应尽量保持相同的网络主干、训练数据与评价方法。

否则如果直接从 YOLO 换成 TrackNet，性能提高后就很难确定是多帧信息、Heatmap 监督，还是其他模型结构产生了作用。

### 3.3 第一阶段输出

每一帧输出：

```
{    "frame": 125,    "x": 823.5,    "y": 419.2,    "confidence": 0.93,    "visible": True}
```

这里的置信度需要在验证集上校准，不应直接把 Heatmap 峰值解释为严格的真实检测概率。

### 3.4 验证指标

除了 F1，还需要直接评估定位精度。

\\[ E\_{\mathrm{center}}= \sqrt{(\hat x-x)^2+(\hat y-y)^2} \\]

建议报告：

- Ball Detection Precision / Recall
- Center Recall @ 5 px、10 px
- Mean / Median Center Error
- 轨迹断裂率与错误跳点数量

所有像素误差都还原到原始视频分辨率后计算。

阶段 1 的验收条件：三帧 Heatmap 模型在未见机位上的定位性能优于现有 YOLO，且可以稳定输出连续视频的逐帧球位置。

如果三帧模型表现并没有明显提高，不要立刻继续堆叠更多帧，应先分析图像分辨率、球尺寸以及标签噪声。

## 四、阶段 2：Local Ball Refinement

Global Locator 的主要任务是从完整画面找到球。

Local Refinement 则尝试解决另一个问题：找到了球的大致区域，但位置还不够准确。

这正是 TTNet 的 Global-to-Local 思想。

### 4.1 具体流程

首先 Global 模型预测：

\\[ \hat p_t^{global}=(x_g,y_g) \\]

然后以该预测位置为中心，从原始高分辨率视频中裁剪局部区域。

例如裁剪 \\(192\times192\\) 或 \\(256\times256\\) 的图像块，交给局部模型：

\\[ \hat p_t^{local}=f\_{\mathrm{local}}(Crop_t) \\]

将局部预测映射回原始坐标：

\\[ \hat p_t=\hat p_t^{global}+\Delta p_t \\]

这里的 \\(\Delta p_t\\) 表示局部裁剪中心到精细球心的相对位移。

这不是完整三维几何问题，纯粹是二维图像中的精细定位。

### 4.2 训练方式

局部模型也使用你现有的球框数据。

但有一个需要特别注意的问题：不要只用 GT 球心作为裁剪中心训练。

因为测试时 Global 模型并不能总是准确定位。

推荐：

- 先使用 GT 附近的随机偏移生成训练 Crop。
- 然后混入 Global 模型的实际预测 Crop。
- 对严重偏离球位置的 Crop 设置明确的失败或无球监督。

这样能够减少 Train-Test Mismatch。

### 4.3 不要让 Local 成为整个系统的单点故障

如果 Global 完全找错位置，那么 Local 无法通过一个很小的 Crop 找到真实球。

因此推荐保留 Top-K Global Candidates，并使用轨迹预测辅助候选选择。

当然，首版可以先做 Top-1，等确认定位误差来自精度而不是漏检，再加入 Top-K。

是否保留 Local 分支取决于实验结果。 如果 Global Heatmap 已经具有足够高的球心精度，Local 可能不是当前最有性价比的投入。

## 五、阶段 3：二维轨迹重建

有了稳定球心位置后，接下来需要重建球在视频中的运动轨迹。

这里我不推荐直接做单目三维轨迹重建。

### 5.1 轨迹输入

\\[ P=\\{(x_t,y_t,c_t)\\}\_{t=1}^{T} \\]

利用 Kalman Filter、候选关联与局部运动预测，得到一条连续轨迹。

但必须保留原始观测，因为轨迹在球拍击球和桌面反弹时本身就会出现速度突变。

建议保存两个版本：

- `raw_track`：模型原始定位结果。
- `reconstructed_track`：修复短时缺失、移除明显异常点后的轨迹。

对每帧计算：

\\[ v_t=\frac{p_t-p\_{t-1}}{\Delta t} \\]

\\[ \Delta v_t=v\_{t+1}-v_t \\]

对于不同 FPS，所有运动特征必须使用实际时间差。

### 5.2 为什么不能直接用速度反转检测 Bounce？

真实三维反弹中的竖直速度变化，在不同相机视角下投影成的二维运动并不一致。

而且：

- 球拍触球也会出现速度突变。
- 地面反弹也会出现速度突变。
- 球定位错误会产生假的速度突变。
- 25 FPS 可能跳过真实接触时刻。

因此，运动学特征适合用于寻找碰撞候选，不适合单独判定事件类别。

### 5.3 这一阶段的输出

```
video_001/
├── ball_positions.csv
├── track_reconstructed.csv
├── candidate_events.csv
└── trajectory_visualization.mp4
```

其中 `candidate_events.csv` 只包含疑似接触帧及其候选分数，不是最终 Bounce 结果。

## 六、阶段 4：建立事件标注数据集

这是当前数据最明显的缺口。

原始 TTNet 使用具有事件标签的 OpenTTGames 数据集，识别包括 Bounce 和 Net Hit 在内的事件。你的项目没有这类标签，因此不可能仅利用现有 Ball Bounding Box 就直接进行完整事件分支的监督训练。

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv

+1



### 6.1 推荐标注内容

| 标注             | 说明            |
| -------------- | ------------- |
| `table_bounce` | 球在桌面反弹        |
| `racket_hit`   | 球拍击球          |
| `floor_bounce` | 球在地面反弹        |
| `ignore`       | 不确定、遮挡或无法准确判断 |
| `target_side`  | 当前反弹属于哪一侧半台   |

建议再给事件添加 `frame_uncertainty`，表示最接近接触的帧是否存在不确定性。

例如：

```
{
  "video_id": "video_001",
  "events": [
    {
      "frame": 125,
      "type": "table_bounce",
      "side": "far",
      "uncertainty_frames": 1
    },
    {
      "frame": 140,
      "type": "racket_hit"
    }
  ]
}
```

示例中的帧编号不代表真实数据。

### 6.2 如何降低人工标注成本？

推荐先使用已有轨迹的突变特征生成事件候选，然后自动裁剪候选帧前后的短视频，人工判断是什么事件。

但必须保留部分随机抽样的完整视频供人工检查。

否则如果运动学方法漏掉某些视角下的真实落台，你的训练集也会系统性缺少这些事件，导致后续 Event Model 重复同样的错误。

### 6.3 初期标注多少？

建议先建立一个 Pilot Dataset：

- 约 200～300 个经过复核的桌面反弹事件。
- 尽可能补充球拍击球、地面反弹和检测异常导致的 Hard Negatives。
- 涵盖所有主要机位。
- 单独保留完整视频作为验证集。

这个规模是用于评估方案是否有学习信号的起点，不保证训练出最终比赛所需的准确率。

## 七、阶段 5：Event Spotting

这是整个系统的第二个核心学习任务。

推荐使用 Motion + Visual 双分支事件检测，而不只是轨迹速度阈值。

### 7.1 轨迹分支

输入每帧：

\\[ f_t^{motion}= [x_t,y_t,v\_{x,t},v\_{y,t}, \Delta v_t,c_t,m_t,\ldots] \\]

其中 \\(m_t\\) 标记该位置是真实观测还是轨迹重建。

对一段固定时间窗口进行 TCN 编码。

建议先试以候选帧为中心的前后各 0.24 秒，在 25 FPS 下约为 13 帧的总窗口。

### 7.2 视觉分支

视觉分支提取事件发生位置附近的视频特征。

与纯轨迹模型不同，它可以学习：

- 球与台面的接近及接触关系。
- 球与球拍、球员动作的关系。
- 球桌与地面的场景差异。

不过你没有球拍标注，因此第一版可以从候选事件附近的局部视频 Crop 开始，不必立即增加独立的球拍检测网络。

### 7.3 时序融合

Trajectory Encoder

TCN

Visual Encoder

CNN + Temporal Pooling

Feature Concatenation + Fusion MLP

Per-frame Event Probability

None / Table / Racket / Floor

训练时使用事件中心采样，重点增加 Hard Negative 样本，避免大量普通飞行帧淹没少量反弹帧。

在已经得到可靠事件标签之后，推荐的实验顺序为：

| 实验 | 输入                             | 目的            |
| -- | ------------------------------ | ------------- |
| E1 | GT Trajectory                  | 验证轨迹是否有能力识别反弹 |
| E2 | Predicted Trajectory           | 评估实际定位误差的影响   |
| E3 | Trajectory + Visual            | 判断视觉特征对误判的改善  |
| E4 | Trajectory + Visual + Geometry | 验证轻量几何信息的价值   |

如果 E1 就明显无法区分桌面与地面反弹，而 E3 能改善结果，说明主要问题是缺少场景信息，而不是轨迹精度不够。

### 7.4 联合 Fine-tuning 是否值得？

借鉴 TTNet 的训练顺序，我建议：

1. 独立训练 Global。
2. 独立或半联合训练 Local。
3. 冻结定位模块，训练 Event Spotting。
4. 等各模块稳定后，再考虑联合 Fine-tuning。

联合损失可以定义为：

\\[ \mathcal L= \lambda_g\mathcal L\_{\mathrm{global}} +\lambda_l\mathcal L\_{\mathrm{local}} +\lambda_e\mathcal L\_{\mathrm{event}} \\]

但要注意，基于预测坐标做硬裁剪通常不是对裁剪位置可微的。因此联合微调并不保证 Event Loss 能通过 Crop 坐标直接优化 Global Locator。

建议把联合微调作为后期可选优化，而不是项目成功的前提。

## 九、我建议的代码架构

整个项目采用独立模块和统一接口，避免后续更换网络时牵动所有代码。

```
table-tennis-bounce/
├── configs/
│   ├── global.yaml
│   ├── local.yaml
│   ├── tracking.yaml
│   └── event.yaml
│
├── datasets/
│   ├── ball_dataset.py
│   ├── event_dataset.py
│   ├── transforms.py
│   └── annotations.py
│
├── models/
│   ├── global_locator.py
│   ├── local_refiner.py
│   ├── trajectory_tracker.py
│   └── event_spotter.py
│
├── geometry/
│   ├── table_geometry.py
│   └── landing_point.py
│
├── scripts/
│   ├── train_global.py
│   ├── train_local.py
│   ├── build_tracks.py
│   ├── propose_events.py
│   ├── train_event.py
│   ├── evaluate.py
│   └── predict.py
│
├── tests/
├── checkpoints/
├── outputs/
├── Dockerfile
└── requirements.txt
```

模块之间统一输出，例如：

```
BallObservation(    frame_idx=125,    x=823.5,    y=419.2,    confidence=0.93,    observed=True)
```

并在 `predict.py` 中完成：

```
def predict_video(video_path):    ball_positions = global_local_locator(video_path)    trajectory = reconstruct_trajectory(ball_positions)    events = event_spotter(video_path, trajectory)    bounces = filter_table_bounces(events)    return localize_landings(bounces, trajectory)
```

这只是接口设计示意，不是已经实现的可运行代码。

后期应以赛事的实际输入输出规范为准，封装最终 Docker 容器，并确保权重、配置与依赖在离线评测环境下可用。

## 十、推荐的团队任务分工

你之前提到过本地多人协作，我建议模块开发尽量独立。

| 工作方向                     | 主要任务                  | 交付物         |
| ------------------------ | --------------------- | ----------- |
| Ball Localization        | Global/Local 网络与训练    | 球心预测权重及推理接口 |
| Tracking & Data          | 数据预处理、轨迹重建、事件标注工具     | 轨迹及事件标注数据   |
| Event Spotting           | 时序事件网络与后处理            | 事件预测权重      |
| Integration & Evaluation | 几何、跨机位实验、完整推理与 Docker | 最终提交系统      |

如果人手不够，优先合并 Tracking 与 Evaluation，不要因为分工而过早拆出许多小模块。

## 十一、整个项目最值得监控的指标

| 指标                      | 评价对象   | 优先级 |
| ----------------------- | ------ | --- |
| Ball Recall             | 球定位    | P0  |
| Center Error            | 球心精度   | P0  |
| Track Continuity        | 轨迹质量   | P0  |
| Bounce Event F1         | 反弹识别   | P0  |
| Temporal Error          | 反弹帧偏差  | P0  |
| Landing Error           | 落点空间位置 | P0  |
| Unseen-view Performance | 跨机位泛化  | P0  |
| Inference FPS           | 推理效率   | P1  |

反弹 Event F1 建议报告不同时间容差下的结果，例如 ±1 帧、±2 帧，同时报告毫秒误差。

尤其要关注完整视频中的误报数量：即使预测出的反弹中大部分都准确，如果每段视频还额外出现多个假的反弹点，最终成绩可能仍然不理想。

## 十二、建议的实际执行路线

里程碑 A：证明多帧定位优于 YOLO

训练 Single-frame Heatmap 和 3-frame Heatmap，重点分析未见机位的 Ball Recall 和 Center Error。

这将决定后续是否以 TrackNet-style 为主要定位方案。

里程碑 B：建立可靠的 Ball Track

加入 Local Refinement（若确实有效）、轨迹关联和可视化，生成完整视频的球运动序列。

里程碑 C：建立事件 Ground Truth

使用自动候选与人工审核结合，补充桌面反弹、球拍接触、地面反弹标签，并评估自动候选的漏报情况。

里程碑 D：训练事件模型

从 Trajectory-only TCN 开始，再加入局部视觉分支。重点解决球拍和地面反弹误判。

里程碑 E：完整系统集成

加入目标半台判定、坐标输出、规则后处理、跨机位评估及 Docker 交付。

## 十三、相关论文与代码

TTNet: Real-time Temporal and Spatial Video Analysis of Table Tennis

整体系统、Global-to-Local 定位、多任务学习和事件识别的主要参考。

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv



[�\[�w�w�K�ő싅�T�\[�u�̋��؂�\���@�łp�����璅�n�\_��������uFuturePong�v�A���H�傪�J���FInnovative Tech - ITmedia NEWS](https://images.openai.com/static-rsc-4/42eI9mbK9lNM-CVhxQvEf1P_dwSfpKNwAs3GXHpp-Ir0wefj62mTEmP9S-fi1tXU0JasJ5Xv8xi2RLv-L1utjQakiD0TnL3VygMwlx3lk78PUCHA-2Klm6qQQmP4nswuGnx11IRGx26ufVmjBt6wdPZL161cSzEcETtrdb4uocM?purpose=inline)

TrackNet: A Deep Learning Network for Tracking High-speed and Tiny Objects in Sports Applications

多帧 Heatmap 定位的主要参考，尤其适合研究高速模糊小球的定位问题。

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv



[TrackNet: A Deep Learning Network for Tracking High-speed and Tiny   Objects in Sports Applications | DeepAI](https://images.openai.com/static-rsc-4/zuz1Fp7oHDXRVvGFSUCKz6BqmCyl2iX0_nbv25Yag3JwRFULBMNhR9AWaZnG-ysS3hiyL7edqbB49Vr0uRnINsevtab5Q2rDzz6z5mLFuY7dC6q0BwO5sIGmBiGWP5oeDoOKmz-PQmgGWvzD-dK5zGCxdHQJfOEDjEFGPHrA5uk?purpose=inline)

TTNet PyTorch Implementation

非官方 PyTorch 复现，包含三阶段训练、Global / Local / Event 分支。可以借鉴训练组织方式，但需要适配你的数据格式和帧率。

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



[New TTNet Table Tennis Model Accelerates DL in Sports Analysis | by Synced | SyncedReview | Medium](https://images.openai.com/static-rsc-4/DkK7i7dgdwjZk5nlkMST7Mgb6raB2kogZD_H-OA7uKYDZups8C4AH8Nzla8xg7nJMu3lpDmfk9WTXMPZQKboqRKtmID9hEqkV44NW5tbKWSsSyn61egfLN-tdE-W_jSOxRxdBFbHInT1xSQSuegyXy1xTCjvzlRKTKGeb01F6R0?purpose=inline)

## 最终决策

对于你目前的项目，我建议把资源投入按以下思路分配：

球定位与轨迹

45%

反弹事件识别

30%

数据标注与评估

15%

几何与最终部署

10%

建议的研发精力分配，仅作为项目管理起点。

当前最应该启动的是 M1 单帧 Heatmap 与 M2 三帧 Heatmap 的对照实验。这一步只依赖已有球框，不需要等待事件标注，也不需要解决三维几何问题。

如果它们确实显著改善定位，再建设 Global-to-Local 体系，并行开展事件标注。等球轨迹和事件标签都达到可用程度，再进入最终的 Bounce Spotting。

这条路线最重要的优势是：每个阶段都有独立的实验结果，可以随时发现瓶颈、替换模型，而不会因为某一模块失败就推倒整个系统。