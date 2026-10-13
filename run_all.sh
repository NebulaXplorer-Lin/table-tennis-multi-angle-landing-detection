#!/usr/bin/env bash
# 联合入口：一次容器依次执行两个任务（乒乓球 → 篮球）。
# 自测用法：run_all.sh [<input_dir> <output_dir>]    # 缺省 /participant/input 与 /participant/output
# 输入含 {pingpang,basketball}/ 两子目录；输出落 <output_dir>/{pingpang,basketball}/。
# 退出码 = 第一个非零任务（两任务都跑完；integrity 失败直接 exit 4）：
#   0 成功 / 1 预测结果校验不通过 / 2 任务运行失败 / 3 输出超1GB / 4 完整性校验不通过 / 124 外层超时强杀
# 注意：TIMEOUT_SEC 对两任务各自生效，平台外层硬超时须 ≥ 2×TIMEOUT_SEC。
# 评测时平台挂载自己的锚点 start.sh 并以 --entrypoint 覆写，本脚本不执行（行为与之一致）。
set -uo pipefail

# conda_env 的 python3（base 环境无依赖）
PY=/opt/conda/envs/conda_env/bin/python3
[ -x "$PY" ] || PY=python3

# 固化运行环境：cuDNN8/cu11 由 torch wheel 自带（喂活 TRT/ORT CUDA EP）
export LD_LIBRARY_PATH="/opt/conda/envs/conda_env/lib/python3.10/site-packages/torch/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

ROOT=/participant
IN="${1:-/participant/input}"
OUT="${2:-/participant/output}"
SOLUTION="${SOLUTION:-participant.solution:Solution}"
DECODER="${DECODER:-gpu}"
TIMEOUT_SEC="${TIMEOUT_SEC:-7200}"

# 完整性自校验（防框架被改或传输损坏；被改则无成绩）
for t in pingpang basketball; do
    PYTHONPATH="$ROOT/$t" "$PY" "$ROOT/$t/integrity.py" --verify || exit 4
done

# 双任务依次执行
run_task () {  # $1 = 任务名
    local task="$1"
    PYTHONPATH="$ROOT/$task" "$PY" "$ROOT/$task/run.py" \
        --input "$IN/$task" --output "$OUT/$task" \
        --solution "$SOLUTION" --decoder "$DECODER" \
        --timeout-sec "$TIMEOUT_SEC"
}

mkdir -p "$OUT/pingpang" "$OUT/basketball"

rc1=0
echo "[run_all] 任务 1/2：pingpang 开始 $(date '+%H:%M:%S')"
run_task pingpang || rc1=$?
echo "[run_all] 任务 1/2：pingpang 退出码 $rc1"

rc2=0
echo "[run_all] 任务 2/2：basketball 开始 $(date '+%H:%M:%S')"
run_task basketball || rc2=$?
echo "[run_all] 任务 2/2：basketball 退出码 $rc2"

# 输出两任务合计 ≤1GB（超出按退出码 3）
TOTAL_MB=$(du -sm "$OUT" 2>/dev/null | cut -f1)
if [ "${TOTAL_MB:-0}" -gt 1024 ]; then
    echo "[run_all] $OUT 合计 ${TOTAL_MB}MB 超 1GB 限额"
    [ "$rc1" -ne 0 ] && exit "$rc1"
    exit 3
fi
[ "$rc1" -ne 0 ] && exit "$rc1"
exit "$rc2"
