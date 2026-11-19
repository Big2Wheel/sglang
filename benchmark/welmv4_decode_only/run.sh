#!/usr/bin/env bash
# Run inside a four-card npu-run lease. Never assign devices or change CANN here.
set -eo pipefail

mode=${1:?native or megakernel}
case "$mode" in native|megakernel) ;; *) echo "Unknown mode: $mode" >&2; exit 2;; esac
out=${2:?output directory}
mkdir -p "$out"
out=$(cd "$out" && pwd)
[[ ! -e "$out/bench.jsonl" && ! -e "$out/serve.log" ]]
: "${NPU_POOL_USER:?run within a card-pool lease}"
: "${NPU_POOL_LEASE:?run within a card-pool lease}"
assigned=${ASCEND_RT_VISIBLE_DEVICES:?visible devices must come from the lease}
[[ "$assigned" =~ ^[0-9]+(,[0-9]+){3}$ ]]
[[ "$(tr ',' '\n' <<< "$assigned" | sort -u | wc -l)" -eq 4 ]]
: "${WELM_MODEL:?48-layer BF16 compact-OE synthetic checkpoint required}"
: "${DEEPEP_PACKAGE_DIR:?installed Ascend allgather-deepep package directory required}"
[[ -f "$WELM_MODEL/config.json" && -d "$DEEPEP_PACKAGE_DIR/vendors/hwcomputing/op_api/lib" ]]

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
sglang=$(cd "$here/../.." && pwd)
cann=${CANN_DIR:-/usr/local/Ascend/cann-9.2.1}
python=${WELM_PYTHON:-python3}
source "$cann/set_env.sh"
[[ "$ASCEND_RT_VISIBLE_DEVICES" == "$assigned" ]]
export PYTHONPATH="$sglang/python:$DEEPEP_PACKAGE_DIR:${PYTHONPATH:-}:$cann/python/site-packages"
export LD_LIBRARY_PATH="$DEEPEP_PACKAGE_DIR/vendors/hwcomputing/op_api/lib:${LD_LIBRARY_PATH:-}"
unset LD_PRELOAD WELM_NPU_USE_MK_TARGET_DECODE WELM_NPU_USE_MK_TARGET_VERIFY WELM_NPU_USE_MK_FLASH_DECODE
unset WELM_MK_TRITON_CHILDREN
export TMPDIR=/tmp OMP_NUM_THREADS=4 PYTHONFAULTHANDLER=1
export HCCL_CONNECT_TIMEOUT=120 HCCL_EXEC_TIMEOUT=180 HCCL_OP_EXPANSION_MODE=CCU_SCHED
export SGLANG_NPU_ALLREDUCE_OUTPLACE=1 SGLANG_WELMV4_DECODE_RS_AG=1
export SGLANG_SCHEDULER_SKIP_ALL_GATHER=0 SGLANG_SET_CPU_AFFINITY=1 STREAMS_PER_DEVICE=32
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True SGLANG_NPU_USE_MULTI_STREAM=0
export WELM_NPU_USE_FLASH_ATTN=1 ASCEND_USE_FIA=1 WELM_NPU_USE_MEGAMOE=0
export SGLANG_DEEPEP_NORMAL_USE_ALLGATHER=1 SGLANG_DEEPEP_NORMAL_USE_ALLTOALL=0
export SGLANG_NPU_USE_SCATTER_PA_KV_CACHE=1 SGLANG_NPU_WELMV4_FUSED_QKV=0
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=8 DEEPEP_HCCL_BUFFSIZE=2000 DEEP_USE_MODE=default
export SGLANG_FORCE_COARSE_WAR_BARRIER=1 SGLANG_WELMV4_MK_DECODE_FAST_PATH=0
if [[ "$mode" == megakernel ]]; then
    : "${WELM_MEGAKERNEL_SRC:?MegaKernel checkout containing the WeLM replay changes required}"
    : "${SHMEM_HOME_PATH:?SHMEM install root required}"
    export MEGA_KERNEL_HOME=${MEGA_KERNEL_HOME:-"$WELM_MEGAKERNEL_SRC/output"}
    test -s "$MEGA_KERNEL_HOME/lib/libmk_capture.so"
    export PYTHONPATH="$WELM_MEGAKERNEL_SRC/python:$PYTHONPATH"
    export LD_LIBRARY_PATH="$MEGA_KERNEL_HOME/lib:$SHMEM_HOME_PATH/shmem/lib:$LD_LIBRARY_PATH"
    export LD_PRELOAD="$MEGA_KERNEL_HOME/lib/libmk_capture.so"
    export WELM_NPU_USE_MK_TARGET_DECODE=1 SGLANG_WELMV4_MK_DECODE_FAST_PATH=1
fi

port=${WELM_TEST_PORT:-16698}
endpoint="http://127.0.0.1:$port"
env -u LD_PRELOAD "$python" - "$port" <<'PY'
import socket
import sys

with socket.socket() as sock:
    sock.bind(("127.0.0.1", int(sys.argv[1])))
PY
cleanup() {
    if [[ -n ${server:-} ]]; then
        kill -TERM "$server" 2>/dev/null || true
        wait "$server" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

if [[ "$mode" == megakernel ]]; then
    timeout -k 10 120 env -u LD_PRELOAD "$python" "$here/preflight.py" > "$out/preflight.log" 2>&1
fi

timeout -k 10 750 "$python" -u -m sglang.launch_server \
    --model-path "$WELM_MODEL" --skip-tokenizer-init --skip-server-warmup \
    --mem-fraction-static 0.80 --tp-size 4 --dp-size 4 --enable-dp-attention --enable-dp-lm-head \
    --moe-dense-tp-size 1 --port "$port" --host 127.0.0.1 --served-model-name welmv45 \
    --nnodes 1 --node-rank 0 --cuda-graph-backend-decode full --cuda-graph-max-bs-decode 1 \
    --attention-backend ascend --context-length 18432 --json-model-override-args '{"max_position_embeddings":262144}' \
    --enable-kv-mirror --chunked-prefill-size 16384 --enable-over-encoding \
    --max-running-requests 32 --swa-full-tokens-ratio 0.8 --disable-swa-chunk-cap --page-size 128 \
    --moe-a2a-backend deepep --deepep-mode auto --enable-cache-report --sampling-backend ascend \
    --disaggregation-mode decode --disaggregation-transfer-backend fake > "$out/serve.log" 2>&1 &
server=$!
for _ in {1..140}; do
    if ! kill -0 "$server" 2>/dev/null; then
        tail -40 "$out/serve.log"
        exit 11
    fi
    if grep -q "Uvicorn running" "$out/serve.log" && curl -fsS --max-time 2 "$endpoint/health" >/dev/null; then
        curl -fsS --max-time 5 "$endpoint/get_server_info" > "$out/server-info.json" || true
        env -u LD_PRELOAD "$python" "$here/bench.py" --url "$endpoint" --input 16500 --output 150 --rounds 5 \
            | tee "$out/bench.jsonl"
        grep -q 'npu graph: True' "$out/serve.log"
        if [[ "$mode" == megakernel ]]; then
            env -u LD_PRELOAD "$python" "$here/audit.py" "$out/serve.log" > "$out/replay-audit.json"
        fi
        exit 0
    fi
    sleep 3
done
tail -40 "$out/serve.log"
exit 12
