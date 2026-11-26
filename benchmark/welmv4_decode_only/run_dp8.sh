#!/usr/bin/env bash
# Foreground WeLMV4 DP8EP8 Decode service. Device selection belongs to the caller.
set -eo pipefail

mode=${1:?usage: run_dp8.sh megakernel-native|native [--dry-run]}
dry_run=${2:-}
case "$mode" in megakernel-native|native) ;; *) echo "Unknown backend: $mode" >&2; exit 2;; esac
[[ ${#} -le 2 && ( -z $dry_run || $dry_run == --dry-run ) ]]
router_fusion=${WELM_DP8_ROUTER_FUSION-0}
if [[ $router_fusion != 0 && ( $router_fusion != 1 || $mode != megakernel-native ) ]]; then
    echo "WELM_DP8_ROUTER_FUSION must be 0 (off) or 1 with megakernel-native" >&2
    exit 2
fi
postmoe_fusion=${WELM_DP8_POSTMOE_FUSION-0}
if [[ $postmoe_fusion != 0 && ( $postmoe_fusion != 1 || $mode != megakernel-native ) ]]; then
    echo "WELM_DP8_POSTMOE_FUSION must be 0 (off) or 1 with megakernel-native" >&2
    exit 2
fi
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
sglang=$(cd "$here/../.." && pwd)
python=${WELM_PYTHON:-python3}
model=${WELM_MODEL:?set WELM_MODEL to the WeLMV4 checkpoint}
transfer=${WELM_DECODE_TRANSFER_BACKEND:-fake}
port=${WELM_TEST_PORT:-16698}
pool=${WELM_BENCH_KV_TOKENS:-196608}
context=${WELM_CONTEXT_LENGTH:-18432}
for value in "$port" "$pool" "$context"; do [[ $value =~ ^[1-9][0-9]*$ ]]; done

args=(--model-path "$model" --served-model-name welmv4 --device npu
    --host 127.0.0.1 --port "$port" --skip-server-warmup
    --nnodes 1 --node-rank 0 --tp-size 8 --dp-size 8 --ep-size 8
    --enable-dp-attention --enable-dp-lm-head --moe-dense-tp-size 1
    --moe-a2a-backend deepep --deepep-mode auto --attention-backend ascend
    --sampling-backend ascend --cuda-graph-backend-decode full
    --cuda-graph-max-bs-decode 8 --cuda-graph-bs-decode 1 2 4 8
    --model-execution-backend "$mode" --max-running-requests 64
    --enable-kv-mirror --enable-over-encoding --enable-cache-report
    --context-length "$context" --max-total-tokens "$pool" --mem-fraction-static 0.80
    --chunked-prefill-size 16384 --page-size 128 --swa-full-tokens-ratio 0.8
    --disable-swa-chunk-cap --disaggregation-mode decode
    --disaggregation-transfer-backend "$transfer"
    --json-model-override-args '{"max_position_embeddings":262144}')
if [[ $transfer == fake ]]; then args+=(--skip-tokenizer-init); fi
if [[ $dry_run == --dry-run ]]; then
    if [[ $router_fusion == 1 ]]; then printf 'WELM_DP8_ROUTER_FUSION=1 '; fi
    if [[ $postmoe_fusion == 1 ]]; then printf 'WELM_DP8_POSTMOE_FUSION=1 '; fi
    printf '%q ' "$python" -u -m sglang.launch_server "${args[@]}"
    printf '\n'
    exit 0
fi

assigned=${ASCEND_RT_VISIBLE_DEVICES:?obtain eight authorized devices before launching}
[[ $assigned =~ ^[0-9]+(,[0-9]+){7}$ ]]
[[ $(tr ',' '\n' <<< "$assigned" | sort -nu | wc -l) -eq 8 ]]
deepep=${DEEPEP_PACKAGE_DIR:?set to the compatible stock Ascend deep_ep package directory}
cann=${CANN_DIR:-${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit}}
test -f "$model/config.json"
test -d "$deepep/vendors/hwcomputing/op_api/lib"
# The caller owns the writable, private SDK static registry. Do not install or
# mount operators into a shared toolkit on their behalf.
opp=${ASCEND_OPP_PATH:-}
if [[ $mode == megakernel-native ]]; then
    : "${opp:?set ASCEND_OPP_PATH to your private CANN operator tree}"
    opp=$(cd "$opp" && pwd -P)
    test -d "$opp/static_kernel/ai_core"
    test -w "$opp/static_kernel/ai_core"
    [[ ! "$opp/static_kernel/ai_core" -ef "$cann/opp/static_kernel/ai_core" ]]
fi
source "$cann/set_env.sh"
[[ $ASCEND_RT_VISIBLE_DEVICES == "$assigned" ]]
if [[ -n $opp ]]; then export ASCEND_OPP_PATH="$opp"; fi

export PYTHONPATH="$sglang/python:$deepep:${deepep%/*}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$deepep/vendors/hwcomputing/op_api/lib:${LD_LIBRARY_PATH:-}"
export ASCEND_CUSTOM_OPP_PATH="$deepep/vendors/hwcomputing"
unset LD_PRELOAD WELM_NPU_USE_MK_TARGET_VERIFY WELM_NPU_USE_MK_FLASH_DECODE WELM_MK_TRITON_CHILDREN
export ASCEND_LAUNCH_BLOCKING=0 TASK_QUEUE_ENABLE=1 OMP_NUM_THREADS=4
export HCCL_OP_EXPANSION_MODE=CCU_SCHED HCCL_CONNECT_TIMEOUT=120 HCCL_EXEC_TIMEOUT=180
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-auto}
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export SGLANG_NPU_USE_MULTI_STREAM=0 SGLANG_SET_CPU_AFFINITY=1 STREAMS_PER_DEVICE=32
export SGLANG_LOG_DECODE_GRAPH_KEY=${SGLANG_LOG_DECODE_GRAPH_KEY:-1}
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0
export WELM_NPU_USE_FLASH_ATTN=1 SGLANG_WELMV4_DECODE_RS_AG=1
export SGLANG_FORCE_COARSE_WAR_BARRIER=1 SGLANG_NPU_ALLREDUCE_OUTPLACE=1
export SGLANG_NPU_USE_SCATTER_PA_KV_CACHE=1 SGLANG_NPU_WELMV4_FUSED_QKV=0
export SGLANG_SCHEDULER_SKIP_ALL_GATHER=0 SGLANG_WELMV4_MK_EP_ALLGATHER=0
export SGLANG_DEEPEP_NORMAL_USE_ALLGATHER=1 SGLANG_DEEPEP_NORMAL_USE_ALLTOALL=0
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=8 DEEPEP_HCCL_BUFFSIZE=2000
export DEEP_USE_MODE=default MOE_ENABLE_CCU=0 MOE_ENABLE_TOPK_NEG_ONE=0
export WELM_NPU_MOE_USE_ND=0 WELM_NPU_USE_MEGAMOE=0 ASCEND_USE_FIA=1
export SGLANG_WELMV4_MK_DECODE_TOPK_MODE3=0 SGLANG_WELMV4_MK_DECODE_FLASH_KV_BOUND=0
export SGLANG_WELMV4_MK_DECODE_PRE_MOE_MASK_NORM=0 SGLANG_WELMV4_MK_DECODE_KV_PROLOGUE=0
export SGLANG_WELMV4_MK_DECODE_FLASH_FULL_UNMASKED=0
export SGLANG_NPU_MOE_GATING_TOPK_SIGMOID_NO_RENORM=0
export WELM_NPU_USE_MK_TARGET_DECODE=0 SGLANG_WELMV4_MK_DECODE_FAST_PATH=0
if [[ $mode == megakernel-native ]]; then
    mk=${WELM_MEGAKERNEL_SRC:?set to the built MegaKernel checkout}
    : "${SHMEM_HOME_PATH:?source the SHMEM installation environment first}"
    export MEGA_KERNEL_HOME=${MEGA_KERNEL_HOME:-"$mk/output"}
    test -s "$MEGA_KERNEL_HOME/lib/libmk_capture.so"
    MEGA_KERNEL_HOME=$(cd "$MEGA_KERNEL_HOME" && pwd -P)
    export PYTHONPATH="$mk/python:$PYTHONPATH"
    export LD_LIBRARY_PATH="$MEGA_KERNEL_HOME/lib:$SHMEM_HOME_PATH/shmem/lib:$SHMEM_HOME_PATH/lib:$LD_LIBRARY_PATH"
    export LD_PRELOAD="$MEGA_KERNEL_HOME/lib/libmk_capture.so"
    export WELM_NPU_USE_MK_TARGET_DECODE=1 SGLANG_WELMV4_MK_DECODE_FAST_PATH=1
fi
exec "$python" -u -m sglang.launch_server "${args[@]}"
