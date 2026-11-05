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

## ball_mixed_baseline

| 实验 | 阶段 | 状态 | 评估协议 | F1 | FPS |
|---|---|---|---|---:|---:|
| [20261103-142106_mixed_ball_A_global](../out/results/20261103-142106_mixed_ball_A_global/summary.md) | global | completed | val_ball_center_10px | 0.8706 | 239.0574 |
| [20261103-142106_mixed_ball_A_global](../out/results/20261103-142106_mixed_ball_A_global/summary.md) | global | completed | val_ball_center_50px | 0.8757 | 239.0574 |
| [20261103-142106_mixed_ball_A_local](../out/results/20261103-142106_mixed_ball_A_local/summary.md) | local | completed | val_ball_center_10px | 0.8738 | 141.4763 |
| [20261103-142106_mixed_ball_A_local](../out/results/20261103-142106_mixed_ball_A_local/summary.md) | local | completed | val_ball_center_50px | 0.8779 | 141.4763 |

## ball_mixed_hr_se

| 实验 | 阶段 | 状态 | 评估协议 | F1 | FPS |
|---|---|---|---|---:|---:|
| [20261103-142106_mixed_ball_B_global](../out/results/20261103-142106_mixed_ball_B_global/summary.md) | global | completed | val_ball_center_10px | 0.9021 | 94.1812 |
| [20261103-142106_mixed_ball_B_global](../out/results/20261103-142106_mixed_ball_B_global/summary.md) | global | completed | val_ball_center_50px | 0.9098 | 94.1812 |
| [20261103-142106_mixed_ball_B_local](../out/results/20261103-142106_mixed_ball_B_local/summary.md) | local | completed | val_ball_center_10px | 0.9046 | 73.6401 |
| [20261103-142106_mixed_ball_B_local](../out/results/20261103-142106_mixed_ball_B_local/summary.md) | local | completed | val_ball_center_50px | 0.9114 | 73.6401 |

## ball_single

| 实验 | 阶段 | 状态 | 评估协议 | F1 | FPS |
|---|---|---|---|---:|---:|
| [20261013-234947_global_ball](../ttnet/output/20261013-234947_global_ball/summary.md) | global | completed | val_ball_center_10px | 0.4483 | 319.9384 |
| [20261013-234947_global_ball](../ttnet/output/20261013-234947_global_ball/summary.md) | global | completed | val_ball_center_50px | 0.4617 | 319.9384 |
| [20261104-145747_B_mlp_thin](../out/experiments/20261104-145747_B_mlp_thin/run.json) | pipeline | completed | unscored | — | — |

## event_B_yolo_F0

| 实验 | 阶段 | 状态 | 评估协议 | F1 | FPS |
|---|---|---|---|---:|---:|
| [20261104-182418_B_event_F0_seed42](../out/experiments/20261104-182418_B_event_F0_seed42/evaluations/continuous/metrics.json) | event | evaluated | pp_val_landing_macro_f1 | 0.8133 | 53.1128 |

## event_B_yolo_F0_seed43

| 实验 | 阶段 | 状态 | 评估协议 | F1 | FPS |
|---|---|---|---|---:|---:|
| [20261104-194046_B_event_F0_seed43](../out/experiments/20261104-194046_B_event_F0_seed43/evaluations/continuous/metrics.json) | event | evaluated | pp_val_landing_macro_f1 | 0.7689 | 52.9502 |

## event_B_yolo_F0_seed44

| 实验 | 阶段 | 状态 | 评估协议 | F1 | FPS |
|---|---|---|---|---:|---:|
| [20261104-194046_B_event_F0_seed44](../out/experiments/20261104-194046_B_event_F0_seed44/evaluations/continuous/metrics.json) | event | evaluated | pp_val_landing_macro_f1 | 0.7957 | 53.5345 |

## event_B_yolo_concat

| 实验 | 阶段 | 状态 | 评估协议 | F1 | FPS |
|---|---|---|---|---:|---:|
| [20261104-182418_B_event_concat_seed42](../out/experiments/20261104-182418_B_event_concat_seed42/evaluations/continuous/metrics.json) | event | evaluated | pp_val_landing_macro_f1 | 0.7898 | 43.2819 |

## event_B_yolo_concat_contiguous

| 实验 | 阶段 | 状态 | 评估协议 | F1 | FPS |
|---|---|---|---|---:|---:|
| [20261104-195314_B_event_concat_seed42_contiguous](../out/experiments/20261104-195314_B_event_concat_seed42_contiguous/evaluations/continuous/metrics.json) | event | evaluated | pp_val_landing_macro_f1 | 0.8391 | 43.1744 |

## event_B_yolo_concat_seed43

| 实验 | 阶段 | 状态 | 评估协议 | F1 | FPS |
|---|---|---|---|---:|---:|
| [20261104-194046_B_event_concat_seed43](../out/experiments/20261104-194046_B_event_concat_seed43/evaluations/continuous/metrics.json) | event | evaluated | pp_val_landing_macro_f1 | 0.8202 | 43.2392 |

## event_B_yolo_concat_seed44

| 实验 | 阶段 | 状态 | 评估协议 | F1 | FPS |
|---|---|---|---|---:|---:|
| [20261104-194046_B_event_concat_seed44](../out/experiments/20261104-194046_B_event_concat_seed44/evaluations/continuous/metrics.json) | event | evaluated | pp_val_landing_macro_f1 | 0.8238 | 43.2600 |

## event_B_yolo_geometry

| 实验 | 阶段 | 状态 | 评估协议 | F1 | FPS |
|---|---|---|---|---:|---:|
| [20261104-182418_B_event_geometry_seed42](../out/experiments/20261104-182418_B_event_geometry_seed42/evaluations/continuous/metrics.json) | event | evaluated | pp_val_landing_macro_f1 | 0.8216 | 43.4532 |

## event_B_yolo_geometry_contiguous

| 实验 | 阶段 | 状态 | 评估协议 | F1 | FPS |
|---|---|---|---|---:|---:|
| [20261104-195314_B_event_geometry_seed42_contiguous](../out/experiments/20261104-195314_B_event_geometry_seed42_contiguous/evaluations/continuous/metrics.json) | event | evaluated | pp_val_landing_macro_f1 | 0.8216 | 42.8413 |

## event_B_yolo_geometry_seed43

| 实验 | 阶段 | 状态 | 评估协议 | F1 | FPS |
|---|---|---|---|---:|---:|
| [20261104-194046_B_event_geometry_seed43](../out/experiments/20261104-194046_B_event_geometry_seed43/evaluations/continuous/metrics.json) | event | evaluated | pp_val_landing_macro_f1 | 0.7514 | 43.0387 |

## event_B_yolo_geometry_seed44

| 实验 | 阶段 | 状态 | 评估协议 | F1 | FPS |
|---|---|---|---|---:|---:|
| [20261104-194046_B_event_geometry_seed44](../out/experiments/20261104-194046_B_event_geometry_seed44/evaluations/continuous/metrics.json) | event | evaluated | pp_val_landing_macro_f1 | 0.8295 | 43.9452 |

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

## event_public_B_F0_seed42

| 实验 | 阶段 | 状态 | 评估协议 | F1 | FPS |
|---|---|---|---|---:|---:|
| [20261105-092319_public_B_F0_seed42](../experiments/records/20261105-092319_public_B_F0_seed42/summary.md) | event | completed | pp_public_validation_landing_macro_f1 | 0.6647 | 53.6191 |

## event_public_B_F0_seed43

| 实验 | 阶段 | 状态 | 评估协议 | F1 | FPS |
|---|---|---|---|---:|---:|
| [20261105-092319_public_B_F0_seed43](../experiments/records/20261105-092319_public_B_F0_seed43/summary.md) | event | completed | pp_public_validation_landing_macro_f1 | 0.6616 | 53.7440 |

## event_public_B_F0_seed44

| 实验 | 阶段 | 状态 | 评估协议 | F1 | FPS |
|---|---|---|---|---:|---:|
| [20261105-092319_public_B_F0_seed44](../experiments/records/20261105-092319_public_B_F0_seed44/summary.md) | event | completed | pp_public_validation_landing_macro_f1 | 0.7357 | 53.9894 |

## event_public_B_concat_seed42

| 实验 | 阶段 | 状态 | 评估协议 | F1 | FPS |
|---|---|---|---|---:|---:|
| [20261105-092319_public_B_concat_seed42](../experiments/records/20261105-092319_public_B_concat_seed42/summary.md) | event | completed | pp_public_validation_landing_macro_f1 | 0.7566 | 43.1795 |

## event_public_B_concat_seed43

| 实验 | 阶段 | 状态 | 评估协议 | F1 | FPS |
|---|---|---|---|---:|---:|
| [20261105-092319_public_B_concat_seed43](../experiments/records/20261105-092319_public_B_concat_seed43/summary.md) | event | completed | pp_public_validation_landing_macro_f1 | 0.7395 | 43.1352 |

## event_public_B_concat_seed44

| 实验 | 阶段 | 状态 | 评估协议 | F1 | FPS |
|---|---|---|---|---:|---:|
| [20261105-092319_public_B_concat_seed44](../experiments/records/20261105-092319_public_B_concat_seed44/summary.md) | event | completed | pp_public_validation_landing_macro_f1 | 0.7329 | 43.5433 |

## event_public_B_gated_seed42

| 实验 | 阶段 | 状态 | 评估协议 | F1 | FPS |
|---|---|---|---|---:|---:|
| [20261105-092319_public_B_gated_seed42](../experiments/records/20261105-092319_public_B_gated_seed42/summary.md) | event | completed | pp_public_validation_landing_macro_f1 | 0.7442 | 43.3016 |

## event_public_B_gated_seed43

| 实验 | 阶段 | 状态 | 评估协议 | F1 | FPS |
|---|---|---|---|---:|---:|
| [20261105-092319_public_B_gated_seed43](../experiments/records/20261105-092319_public_B_gated_seed43/summary.md) | event | completed | pp_public_validation_landing_macro_f1 | 0.7759 | 43.1847 |

## event_public_B_gated_seed44

| 实验 | 阶段 | 状态 | 评估协议 | F1 | FPS |
|---|---|---|---|---:|---:|
| [20261105-092319_public_B_gated_seed44](../experiments/records/20261105-092319_public_B_gated_seed44/summary.md) | event | completed | pp_public_validation_landing_macro_f1 | 0.7226 | 43.2588 |

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
