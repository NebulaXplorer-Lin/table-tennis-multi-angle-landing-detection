# Local TensorRT validation

Tested entry: `participant.solution_public_trt_local:Solution`.
Comparison entry: `participant.solution_public_trt_fast:Solution`.

Local is the residual-FPN ball refinement network on three 256x256 crops.
It is separate from YOLO11s, which detects the table and provides spatial features.
This entry inherits the existing Global/table TensorRT engines and exact
normalization lookup, and replaces only the Local forward.

## Measured results

Full official-runner comparison on 15 videos / 2,109 frames:

| Entry | FPS | Macro F1 | TP/FP/FN |
|---|---:|---:|---:|
| Existing fast entry | 35.3 | 78.121430% | 75/24/17 |
| With Local TensorRT | 43.1 | 78.121430% | 75/24/17 |

All 102 prediction frame IDs match. 100 coordinates change slightly; the maximum
absolute coordinate-component difference is 0.003690 pixel. This is not bit-exact
inference and is only a public-set regression, not an independent accuracy test.
Single sequential pair, not repeated randomized trials. Still below 50 FPS.

80 actual Local inputs from the tracking pipeline cover all 15 videos and normal,
start-padding and end-padding masks. All heatmap argmax positions match.
Standalone median Local forward: PyTorch AMP 4.160 ms; TensorRT 1.094 ms.
Those isolated timings exclude data preparation and do not represent whole FPS.

## Run

```bash
cd /home/lirong/pingpong-public-ee754ef
export PYTHONUTF8=1
export PYTHONDONTWRITEBYTECODE=1
export YOLO_CONFIG_DIR=$PWD/out/yolo-settings
export TTNET_CHECKPOINT=/home/lirong/pingpong-tcn/out/new-public-weight-20261105/20261105-145517_public_event_best/best.pt
export TTNET_TRT_DIR=$PWD/out/public-trt-001/engines
export TTNET_LOCAL_TRT_DIR=$PWD/out/public-local-trt-001/engines
/opt/conda/envs/conda_env/bin/python pingpang/run.py --input /workspace/pingpong/pingpang/public_data --output out/NEW_OUTPUT --solution participant.solution_public_trt_local:Solution --decoder gpu </dev/null
```

Use a new output directory. Original weights, thresholds, crop strategy, resolution,
Local call frequency, tracking and NMS remain unchanged. TCN remains PyTorch AMP.
Default solution.py and the previous fast entry remain available.

## Rebuild and verify

Retain the environment above; use a new experiment directory:

```bash
/opt/conda/envs/conda_env/bin/python pingpang/participant/capture_local_trt_inputs.py out/NEW_LOCAL_EXPERIMENT
/opt/conda/envs/conda_env/bin/python pingpang/participant/build_local_trt.py out/NEW_LOCAL_EXPERIMENT
/opt/conda/envs/conda_env/bin/python pingpang/participant/evaluate_local_trt.py --variant baseline --input /workspace/pingpong/pingpang/public_data --output out/NEW_LOCAL_EXPERIMENT
/opt/conda/envs/conda_env/bin/python pingpang/participant/evaluate_local_trt.py --variant local --input /workspace/pingpong/pingpang/public_data --output out/NEW_LOCAL_EXPERIMENT
```

The capture pass is for numeric inputs and must not be used as the FPS baseline.
Its temporary `local_inputs.pt` can be removed after validation. Numeric summaries
and logs remain in the experiment directory. Engines are built for RTX 4090 /
TensorRT 8.6.1, using FP16 with FP32 normalization/reduction constraints. No INT8
quantization or dependency changes. Existing environment: PyTorch 2.1.2+cu118.
