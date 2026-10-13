#!/usr/bin/env bash
# 单任务入口：固化环境 → 完整性校验 → exec run.py。
# 联合评测由 /participant/run_all.sh（或平台锚点 start.sh）调用；单任务可直接执行：
#   run.sh [<input_dir> <output_dir>]    # 缺省 /participant/input 与 /participant/output
# 退出码（与 run.py 一致）：0 成功 / 1 预测结果校验不通过 / 2 任务运行失败 / 3 输出超1GB / 4 完整性校验不通过 / 124 外层超时强杀
set -euo pipefail

# conda_env 的 python3（base 环境无依赖）
PY=/opt/conda/envs/conda_env/bin/python3
[ -x "$PY" ] || PY=python3

# 固化运行环境：cuDNN8/cu11 由 torch wheel 自带（喂活 TRT/ORT CUDA EP）；覆写 ENTRYPOINT 时兜底
export LD_LIBRARY_PATH="/opt/conda/envs/conda_env/lib/python3.10/site-packages/torch/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="/participant/basketball${PYTHONPATH:+:${PYTHONPATH}}"

# 完整性自校验（防框架被改或传输损坏；被改则无成绩）
"$PY" /participant/basketball/integrity.py --verify

# 输入/输出目录为位置参数，缺省平台约定路径；其余默认值可在参赛者 Dockerfile 用 ENV 覆写
INPUT="${1:-/participant/input}"
OUTPUT="${2:-/participant/output}"
SOLUTION="${SOLUTION:-participant.solution:Solution}"
DECODER="${DECODER:-gpu}"
TIMEOUT_SEC="${TIMEOUT_SEC:-7200}"

# 透传执行 run.py；"$@" 追加在末尾（argparse 后写覆盖先写）
exec "$PY" /participant/basketball/run.py \
    --input "$INPUT" --output "$OUTPUT" \
    --solution "$SOLUTION" --decoder "$DECODER" \
    --timeout-sec "$TIMEOUT_SEC" \
    "$@"
