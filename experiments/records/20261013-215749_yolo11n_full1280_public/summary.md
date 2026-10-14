# YOLO11n paused evaluation

Training paused after 32 completed epochs; 80 were planned. All 154 training videos / 20,846 labelled positive frames, 1280 input. Checkpoint: `out/experiments/20261013-215749_yolo11n_full1280_public/eval/epoch032_public_paused/checkpoint.pt`.

Public landing validation: micro F1=0.398148, macro F1=0.386599.

- 0度: {'tp': 5, 'fp': 31, 'fn': 16, 'precision': 0.1388888888888889, 'recall': 0.23809523809523808, 'f1': 0.17543859649122806}
- 45度: {'tp': 19, 'fp': 36, 'fn': 19, 'precision': 0.34545454545454546, 'recall': 0.5, 'f1': 0.40860215053763443}
- 90度: {'tp': 19, 'fp': 14, 'fn': 14, 'precision': 0.5757575757575758, 'recall': 0.5757575757575758, 'f1': 0.5757575757575758}

Local approximate scorer: ±1 frame / 50px; predictions near ignored events removed before valid-GT matching. All 15 complete overlay videos: `out/experiments/20261013-215749_yolo11n_full1280_public/eval/epoch032_public_paused/overlays`. Training remains paused.
