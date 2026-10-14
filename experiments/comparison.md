# Experiment comparison

Ball localisation and landing detection are different tasks. touch_f1 is angle macro F1 under the recorded local public protocol.

| experiment_id | eval_id | task | status | center_f1 | touch_f1 | e2e_fps |
|---|---|---|---|---|---|---|
| 20261013-163117_yolo11n_ball-640 | best_fp32_640 | ball_localisation | completed | 0.0 | — | 316.73222827605866 |
| 20261013-163117_yolo11n_ball-640 | score001_ap101_diagnostic | ball_localisation | completed | 0.023459318562651276 | — | — |
| 20261013-212251_baseline_ball-position | onnx_fp32_ball | ball_localisation | failed | — | — | — |
| 20261013-212318_baseline_ball-position | onnx_fp32_ball | ball_localisation | completed | 0.977598307855014 | — | 111.22533523477739 |
| 20261013-215749_yolo11n_full1280_public | epoch032_public_paused | landing_detection | evaluated_training_paused | — | 0.38659944092881277 | 66.2 |
