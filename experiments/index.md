# 方案与实验索引

保留完整版本与已评估对照；中断和预检记录已移出日常入口。指标按任务和评估协议分别展示，稀疏事件分类F1不能替代完整落点F1。

[Quick intro](../docs/quick_intro.md) · [当前推荐结果](../result/README.md) · [后续计划](../docs/next_steps.md) · [CSV数据源](workspace_comparison.csv)

## ball_fixed3

| 实验 | 阶段 | 状态 | 评估协议 | F1 | FPS |
|---|---|---|---|---:|---:|
| [20261014-095202_global_fixed3_ball](../ttnet/output/20261014-095202_global_fixed3_ball/summary.md) | global | completed | val_ball_center_10px | 0.8434 | 236.8987 |
| [20261014-095202_global_fixed3_ball](../ttnet/output/20261014-095202_global_fixed3_ball/summary.md) | global | completed | val_ball_center_50px | 0.8548 | 236.8987 |
| [20261014-103011_local_fixed3_ball](../ttnet/output/20261014-103011_local_fixed3_ball/summary.md) | local | completed | val_ball_center_10px | 0.8508 | 141.2065 |
| [20261014-103011_local_fixed3_ball](../ttnet/output/20261014-103011_local_fixed3_ball/summary.md) | local | completed | val_ball_center_50px | 0.8606 | 141.2065 |

## ball_single

| 实验 | 阶段 | 状态 | 评估协议 | F1 | FPS |
|---|---|---|---|---:|---:|
| [20261013-234947_global_ball](../ttnet/output/20261013-234947_global_ball/summary.md) | global | completed | val_ball_center_10px | 0.4483 | 319.9384 |
| [20261013-234947_global_ball](../ttnet/output/20261013-234947_global_ball/summary.md) | global | completed | val_ball_center_50px | 0.4617 | 319.9384 |

## event_comparison_A

| 实验 | 阶段 | 状态 | 评估协议 | F1 | FPS |
|---|---|---|---|---:|---:|
| [20261016-153213_event_comparison_b8_seed42_A](../out/experiments/20261016-153213_event_comparison_b8_seed42_A/evaluations/continuous/metrics.json) | event | evaluated | pp_val_landing_macro_f1 | 0.7244 | 100.5321 |

## event_comparison_B

| 实验 | 阶段 | 状态 | 评估协议 | F1 | FPS |
|---|---|---|---|---:|---:|
| [20261016-153213_event_comparison_b8_seed42_B](../out/experiments/20261016-153213_event_comparison_b8_seed42_B/evaluations/continuous/metrics.json) | event | evaluated | pp_val_landing_macro_f1 | 0.7796 | 86.5420 |

## event_comparison_C

| 实验 | 阶段 | 状态 | 评估协议 | F1 | FPS |
|---|---|---|---|---:|---:|
| [20261016-153213_event_comparison_b8_seed42_C](../out/experiments/20261016-153213_event_comparison_b8_seed42_C/evaluations/continuous/metrics.json) | event | evaluated | pp_val_landing_macro_f1 | 0.5585 | 99.3360 |

## event_comparison_D

| 实验 | 阶段 | 状态 | 评估协议 | F1 | FPS |
|---|---|---|---|---:|---:|
| [20261016-153213_event_comparison_b8_seed42_D](../out/experiments/20261016-153213_event_comparison_b8_seed42_D/evaluations/continuous/metrics.json) | event | evaluated | pp_val_landing_macro_f1 | 0.6563 | 85.0325 |

## event_tcn_frozen

| 实验 | 阶段 | 状态 | 评估协议 | F1 | FPS |
|---|---|---|---|---:|---:|
| [20261014-181231_event25_frozen](../ttnet/output/events/20261014-181231_event25_frozen/summary.md) | event | completed | val_sparse_event_classification | 0.9616 | — |
| [20261014-181231_event25_frozen](../ttnet/output/events/20261014-181231_event25_frozen/summary.md) | event | completed | test_sparse_event_classification | 0.8824 | — |
| [20261014-181231_event25_frozen](../ttnet/output/events/20261014-181231_event25_frozen/summary.md) | event | completed | pp_public_landing | 0.4320 | 97.1373 |

## legacy_YOLO11n

| 实验 | 阶段 | 状态 | 评估协议 | F1 | FPS |
|---|---|---|---|---:|---:|
| [20261013-163117_yolo11n_ball-640](../experiments/records/20261013-163117_yolo11n_ball-640/summary.md) | ball_localisation | completed | legacy_ball_v1 | 0.0000 | 316.7322 |
| [20261013-163117_yolo11n_ball-640](../experiments/records/20261013-163117_yolo11n_ball-640/summary.md) | ball_localisation | completed | legacy_ball_v1_posthoc_score001_ap101 | 0.0235 | — |

## legacy_official_HRNet-style_heatmap

| 实验 | 阶段 | 状态 | 评估协议 | F1 | FPS |
|---|---|---|---|---:|---:|
| [20261013-212318_baseline_ball-position](../experiments/records/20261013-212318_baseline_ball-position/summary.md) | ball_localisation | completed | legacy_ball_v1_center | 0.9776 | 111.2253 |

## retained_20261017-112200_tcn_fast_refine

| 实验 | 阶段 | 状态 | 评估协议 | F1 | FPS |
|---|---|---|---|---:|---:|
| [20261017-112200_tcn_fast_refine](../result/20261017-112200_tcn_fast_refine/README.md) | event | evaluated | pp_val_landing_macro_f1 | 0.7969 | 86.8840 |

## retained_baseline_B

| 实验 | 阶段 | 状态 | 评估协议 | F1 | FPS |
|---|---|---|---|---:|---:|
| [20261016_table_ar_baseline_B_seed42](../result/baseline_B/README.md) | event | evaluated | pp_val_landing_macro_f1 | 0.7796 | 79.7525 |

## retained_scene_box

| 实验 | 阶段 | 状态 | 评估协议 | F1 | FPS |
|---|---|---|---|---:|---:|
| [20261016_table_ar_scene_box](../result/scene_box/README.md) | event | evaluated | pp_val_landing_macro_f1 | 0.7873 | 85.3916 |

## retained_tcn_frames_lr0.0003_d0.1

| 实验 | 阶段 | 状态 | 评估协议 | F1 | FPS |
|---|---|---|---|---:|---:|
| [20261016_table_ar_tcn_frames_lr0.0003_d0.1](../result/tcn_frames_lr0.0003_d0.1/README.md) | event | evaluated | pp_val_landing_macro_f1 | 0.7956 | 77.5279 |

## retained_transformer_frames_lr0.0001_d0.1

| 实验 | 阶段 | 状态 | 评估协议 | F1 | FPS |
|---|---|---|---|---:|---:|
| [20261016_table_ar_transformer_frames_lr0.0001_d0.1](../result/transformer_frames_lr0.0001_d0.1/README.md) | event | evaluated | pp_val_landing_macro_f1 | 0.7503 | 79.8307 |
