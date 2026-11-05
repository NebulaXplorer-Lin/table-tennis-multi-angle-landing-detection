# 新定位器直接接原赛事启发式：public验证

完整执行原`pingpang/run.py`，15个视频、2109帧，92个有效落点；使用新增`participant.solution_locator_local:Solution`适配A/B Global＋Local。原`participant.solution.Solution`的`process_frame`与`_emit`原样继承，保留MIN_GAP=10、FALL_AMP=2px、RISE_AMP=1px、CONF_TH=0.05及原尾部规则。未加入关联、平滑、桌框或事件头。

| 定位器 | 0° F1 | 45° F1 | 90° F1 | 宏F1 | 微F1 | TP/FP/FN | 本机FPS |
|---|---:|---:|---:|---:|---:|---|---:|
| 当前YOLO入口 | 17.54% | 40.86% | 57.58% | 38.66% | 39.81% | 43/81/49 | 98.8 |
| A：原结构Global＋Local | 15.87% | 31.68% | 42.86% | 30.14% | 30.77% | 36/106/56 | 76.7 |
| B：高分辨率＋SE Global＋Local | 9.37% | 31.07% | 39.44% | 26.63% | 27.73% | 33/113/59 | 51.1 |

本轮仅将模型在PP验证选出的best接入，不使用public选权重或阈值。A Global/Local阈值0.9/0.9；B为0.7/0.9。Local拒绝时使用本组有效Global回退。三帧定位增加一帧前瞻，输出保留中心帧原始帧号；首尾缺帧带mask，最后中心帧在finish_video刷新后再调用原尾部规则。时间差来自原视频实际解码PTS，不使用标称fps代替。

评分沿用本地public协议：valid GT，忽略靠近valid=false事件的预测，前后1帧、50px内进行一对一匹配；宏F1是三视角平均。它是本地公开验证成绩，不是平台私测成绩。

GPU推理部分独占RTX 5070 Ti，ORT CUDA FP32＋原CPU解码；A/B关闭TF32。FPS包含完整解码、定位、原启发式与输出，不含初始化；YOLO/A/B初始化分别0.6/1.0/1.2秒。官方4090/TensorRT尚未验收。

定位器适配核对通过：实际CUDA日志与原PyTorch CPU FP32在三视角、首尾边界的球心最大差异约2.24e-6px，置信度最大差异约1.79e-7。所有15视频通过赛事输出schema校验，所有评分FP/FN均有原图叠加。

92个有效落点附近（前后1帧、50px）存在正确球点的事件数，A为85、B为88；这只是事件附近的覆盖诊断，不是球定位F1。B漏检59个，其中55个附近有正确球点；这55个中41个在此前不足10帧已输出其他落点，可能触发原最小间隔抑制。该结果说明单帧定位改善并未直接转化为原y极值启发式的落点改善，不能据PP球定位F1直接判断public落点表现。

本轮结论：固定现有启发式时，直接替换为A/B均未优于当前YOLO。未据此修改启发式或重新调参。原混合训练任务的后续评估已自动恢复。

- [完整结果和错误叠图入口](../out/results/20261103-232128_locator_public_heuristic/index.html)
- [对照CSV](../out/results/20261103-232128_locator_public_heuristic/comparison.csv)
- [配置、权重身份与轻量记录](../experiments/records/20261103-232128_locator_public_heuristic/)
- [逐事件诊断](../out/experiments/20261103-232128_locator_public_heuristic/landing_diagnosis.json)

复现：`.venv/bin/python scripts/evaluate_locator_public.py`，新建独立实验目录并冻结两组模型及public身份；已有训练权重不修改。部署适配位于`pingpang/participant/solution_locator_local.py`，不修改受保护的赛事框架。

## 原规则精确回放：漏检来源

已用保存的定位轨迹逐项重放原规则，A/B全部输出与赛事入口结果完全一致，未改参数或生成新的评分方案。此前“41个事件前不足10帧已有输出”仅为相关性统计；精确回放确认B有29个漏检直接被MIN_GAP拦截。

| 原因 | A漏检 | B漏检 |
|---|---:|---:|
| GT前后1帧、50px内没有有效球点 | 7 | 4 |
| 存在正确球点，但不满足原y极值/幅度条件 | 27 | 26 |
| 已满足位置与y极值条件，但被MIN_GAP=10抑制 | 22 | 29 |

具体例子：v000001第23帧GT为(1096.70,367.69)，B预测(1093.46,364.99)，误差4.22px；第22/23/24帧的y分别348.13/364.99/361.19，满足原2px下落、1px回升条件。但第17帧已误报一次落点，23−17=6<10，该正确候选被间隔门控跳过。

“正确球点却不满足y极值”的26例还可能包含邻帧跳点、Global/Local切换与透视下的轨迹形状，单靠事件附近覆盖不能将它们全部归为启发式本身错误，也不能证明public整段轨迹稳定性更好。

[逐例精确回放](../out/experiments/20261103-232128_locator_public_heuristic/rule_replay_diagnosis.json)
