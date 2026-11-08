# Public gated checkpoint participant

Uses the exact ee754ef model implementation and strict full checkpoint loading.
Only the structured table-descriptor batching differs from the previous streaming adapter.

From /home/lirong/pingpong-public-ee754ef:

```bash
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUTF8=1
export YOLO_CONFIG_DIR="$PWD/out/yolo-settings"
export TTNET_CHECKPOINT=/home/lirong/pingpong-tcn/out/new-public-weight-20261105/20261105-145517_public_event_best/best.pt
/opt/conda/envs/conda_env/bin/python pingpang/run.py --input /workspace/pingpong/pingpang/public_data --output out/NEW_OUTPUT_DIRECTORY --solution participant.solution_public_trt_fast:Solution --decoder gpu </dev/null
```

Requires isolated out/yolo-deps/ultralytics 8.3.225. Installed with --no-deps; fixed torch/numpy/ORT unchanged. Network installation is not needed at inference time. Keep output directory unique. The default participant.solution:Solution is unchanged.

Three raw frames and bounded feature/table records; uses seven future images plus peak/NMS delay. NMS connected chains have no universal bounded delay. Official timestamps are frame_id/fps, so equivalence for arbitrary VFR is not claimed. The original PyTorch verification used no ONNX/TensorRT conversion. See README_public_trt.md for the new accelerated path. No 50 FPS or independent-test claim.

Verified: all public15/2109 frames; 102 predictions identical to native offline inference; Macro F1 78.121430%; official run 17.2 FPS. Full overlays/errors are in out/public-seed43-retest-001. Public GT is used only for scoring/overlays.

## Global preprocessing reuse (2026-11-06)

The preprocessing-only entry is participant.solution_public_cached:Solution. The unmodified participant.solution_public:Solution remains the comparison reference. No cache switch is needed.

Each source frame is letterboxed and normalized once for Global; adjacent windows reuse the same arrays. Retains at most three entries, evicts older frames, clears on reset and finish, and keeps separate state per Solution instance. Local crops, boundary padding, YOLO table preprocessing, neural windows, thresholds and NMS retain their original behavior.

Per-video cache counters are recorded in solution_info.global_cache_video_stats inside run_status.json. Current acceptance results are in out/public-seed43-cache-001; use its summary.json for measured speed, parity.json for exact output comparison, and cache_audit.json for hit counts and memory.

## TensorRT and exact normalization (2026-11-06)

Current recommended entry: participant.solution_public_trt_fast:Solution. See [README_public_trt.md](README_public_trt.md) for setup, rebuilding and measured results. Full public-set paired run: 20.0 to 36.6 FPS; macro F1 unchanged at 78.121430%. Original cached entry remains available.
