# Frozen locators + unchanged competition landing heuristic

All 15 public videos / 2,109 source frames. Fixed MIN_GAP=10, FALL_AMP=2, RISE_AMP=1, CONF_TH=0.05. No event head, association or public tuning. Locator thresholds and weights selected on PP only.

| Model | 0° F1 | 45° F1 | 90° F1 | Macro F1 | Micro F1 | P | R | TP/FP/FN | Local FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| YOLO | 17.54% | 40.86% | 57.58% | 38.66% | 39.81% | 34.68% | 46.74% | 43/81/49 | 98.8 |
| A | 15.87% | 31.68% | 42.86% | 30.14% | 30.77% | 25.35% | 39.13% | 36/106/56 | 76.7 |
| B | 9.37% | 31.07% | 39.44% | 26.63% | 27.73% | 22.60% | 35.87% | 33/113/59 | 51.1 |

Local ORT CUDA / CPU decoder on RTX 5070 Ti; inference time includes decoder, locator, original heuristic and outputs; initialization separate. This is not an official 4090/TensorRT acceptance test. The fixed3 locator adds one source-frame lookahead while preserving original event frame IDs.

Public scores use the existing local protocol: valid GT, ignore predictions near invalid GT, one-to-one ±1 frame / 50px. Exact video-hash matches are listed in public_identity.json; resized/reencoded source overlap has not been excluded. Public is diagnostic validation, not a new claim of independent test generalization.

- [YOLO: all scored FP/FN overlays](../../experiments/20261103-232128_locator_public_heuristic/eval/YOLO/index.html)
- [A: all scored FP/FN overlays](../../experiments/20261103-232128_locator_public_heuristic/eval/A/index.html)
- [B: all scored FP/FN overlays](../../experiments/20261103-232128_locator_public_heuristic/eval/B/index.html)

## 部署与错误诊断

最终 PyTorch CPU FP32 与本次实际 ORT CUDA 定位日志逐项核对通过（18个 Global/Local 输出，覆盖三视角及首尾缺帧）。最大球心差异 2.23e-06px，最大置信度差异 1.79e-07；未发现中心帧错位或明显导出差异。

92个有效落点的诊断：GT前后1帧、50px内存在已接受球点的事件数，A为85，B为88。此处是事件附近的定位覆盖诊断，不是完整球定位F1。

B漏检59个落点，其中55个附近存在正确球点；这55个中有41个在此前不足10帧已输出过其他落点，可能受到原MIN_GAP抑制。A漏检56个，其中49个附近存在正确球点。定位可见并不保证原y极值启发式输出正确落点。

因此，按本轮固定启发式直接替换，A/B均未优于当前YOLO入口。没有据此修改规则或用public重新选阈值。

[部署数值核对](../../experiments/20261103-232128_locator_public_heuristic/deployment_parity.json) · [逐事件诊断](../../experiments/20261103-232128_locator_public_heuristic/landing_diagnosis.json)

## 原规则精确回放：漏检来源

已用保存的定位轨迹逐项重放原规则，A/B全部输出与赛事入口结果完全一致，未改参数或生成新的评分方案。此前“41个事件前不足10帧已有输出”仅为相关性统计；精确回放确认B有29个漏检直接被MIN_GAP拦截。

| 原因 | A漏检 | B漏检 |
|---|---:|---:|
| GT前后1帧、50px内没有有效球点 | 7 | 4 |
| 存在正确球点，但不满足原y极值/幅度条件 | 27 | 26 |
| 已满足位置与y极值条件，但被MIN_GAP=10抑制 | 22 | 29 |

具体例子：v000001第23帧GT为(1096.70,367.69)，B预测(1093.46,364.99)，误差4.22px；第22/23/24帧的y分别348.13/364.99/361.19，满足原2px下落、1px回升条件。但第17帧已误报一次落点，23−17=6<10，该正确候选被间隔门控跳过。

“正确球点却不满足y极值”的26例还可能包含邻帧跳点、Global/Local切换与透视下的轨迹形状，单靠事件附近覆盖不能将它们全部归为启发式本身错误，也不能证明public整段轨迹稳定性更好。

[逐例精确回放](../../experiments/20261103-232128_locator_public_heuristic/rule_replay_diagnosis.json)
