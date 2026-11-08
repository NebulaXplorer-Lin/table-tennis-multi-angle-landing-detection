# Public checkpoint: TensorRT + exact normalization

Recommended entry: `participant.solution_public_trt_fast:Solution`.
The original cached PyTorch entry remains `participant.solution_public_cached:Solution`.

Validated on RTX 4090, TensorRT 8.6.1, PyTorch 2.1.2+cu118, ONNX 1.16.2,
Python 3.10.20 and the unchanged ee754ef public checkpoint.
No package installation or dependency upgrade is required on this server.

## Run the prepared engines

```bash
cd /home/lirong/pingpong-public-ee754ef
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUTF8=1
export YOLO_CONFIG_DIR=$PWD/out/yolo-settings
export TTNET_CHECKPOINT=/home/lirong/pingpong-tcn/out/new-public-weight-20261105/20261105-145517_public_event_best/best.pt
export TTNET_TRT_DIR=$PWD/out/public-trt-001/engines
/opt/conda/envs/conda_env/bin/python pingpang/run.py --input /workspace/pingpong/pingpang/public_data --output out/NEW_OUTPUT --solution participant.solution_public_trt_fast:Solution --decoder gpu </dev/null
```

Choose a new output directory. The default `participant.solution:Solution` has
not been replaced. Engine building is not performed during video processing.
Use the original cached entry if the prepared engines are unavailable.

## Measured results

One sequential full-public-set comparison, 15 videos / 2,109 frames:

| Entry | FPS | Macro F1 |
|---|---:|---:|
| solution_public_cached | 20.0 | 78.121430% |
| solution_public_trt_global | 21.4 | 78.121430% |
| solution_public_trt_both | 28.3 | 78.121430% |
| solution_public_trt_fast | 36.6 | 78.121430% |

All have 102 predictions with the same frame IDs. TensorRT changes two ball
coordinates by at most 0.002731 pixel per coordinate. Normalization lookup is
exact: fast and both have identical final predictions. TP/FP/FN stay 75/24/17.
These are public-set regression results, not an independent accuracy test or
a guaranteed speed on other machines. The current implementation is below 50 FPS.

## Implementation

Global keeps the same three-frame input, masks and offsets, and returns the
heatmap and 128-D descriptor. FP16 convolutions are enabled while exported
normalization/reduction arithmetic is protected in FP32. YOLO exports both
detections and P3/8 spatial features. All tensor bindings use their names.
Local, TCN, thresholds, crop logic, frame windows and NMS remain unchanged.

The per-channel uint8 normalization table is generated with the original
float32 formula. It preserves exact pixel values, returns contiguous CHW,
and retains the three-frame Global cache. Non-uint8 inputs use the old function.

## Rebuild and validate on the target GPU

Use a new experiment directory and the environment variables above. Build Global,
run the baseline and Global-only comparison, then build the table engine and run
the two combined variants. Do not run inference benchmarks during engine builds.

```bash
/opt/conda/envs/conda_env/bin/python pingpang/participant/build_public_trt.py --stage global --input /workspace/pingpong/pingpang/public_data --output out/NEW_EXPERIMENT/engines
/opt/conda/envs/conda_env/bin/python pingpang/participant/evaluate_public_trt.py --variant baseline --input /workspace/pingpong/pingpang/public_data --output out/NEW_EXPERIMENT
/opt/conda/envs/conda_env/bin/python pingpang/participant/evaluate_public_trt.py --variant global --input /workspace/pingpong/pingpang/public_data --output out/NEW_EXPERIMENT
/opt/conda/envs/conda_env/bin/python pingpang/participant/build_public_trt.py --stage table --input /workspace/pingpong/pingpang/public_data --output out/NEW_EXPERIMENT/engines
/opt/conda/envs/conda_env/bin/python pingpang/participant/evaluate_public_trt.py --variant both --input /workspace/pingpong/pingpang/public_data --output out/NEW_EXPERIMENT
/opt/conda/envs/conda_env/bin/python pingpang/participant/test_public_fast_normalize.py out/NEW_EXPERIMENT/normalize-tests.json
/opt/conda/envs/conda_env/bin/python pingpang/participant/evaluate_public_trt.py --variant fast --input /workspace/pingpong/pingpang/public_data --output out/NEW_EXPERIMENT
```

Inspect numeric differences, FPS and accuracy before adopting newly built engines.
`--resume-engine` continues validation of an existing engine after a validation
failure; it does not rebuild or change its precision. Only use it with the same
checkpoint and build settings. Building can take several minutes. Engine files
are specific to the runtime and target GPU; changing those requires rebuilding.

Detailed evidence is in `out/public-trt-001`; source and experiment records are
retained without modifying the original checkpoint or training code.
