# WeLMV4 Decode replay comparison

The benchmark uses a 48-layer BF16 synthetic checkpoint on an Ascend 950
DP4EP4 deployment. It sends 16,500 input tokens and generates 150 tokens at
concurrency 1 with fake KV transfer. The five measured requests exclude warmup
and count TPOT and decode throughput from the first to the last output token.

Run inside a four-card lease. Set `WELM_MODEL`, `DEEPEP_PACKAGE_DIR`,
`WELM_MEGAKERNEL_SRC`, and `SHMEM_HOME_PATH` for your installation. If using
an extracted DeepEP wheel, select its vendor operator path for the test process.
Use an isolated CANN operator registry: MegaKernel may install generated
static-kernel packages during the first capture. Use the same checkpoint, card
assignment, pool size, and operator options in both arms.

```bash
bash run.sh megakernel /path/to/results/acl
bash run.sh megakernel-native /path/to/results/mk-direct
```

For a fixed KV pool request, set the same `WELM_BENCH_KV_TOKENS` in both
arms and verify the realized pool sizes in the startup logs.

- `megakernel`: existing MegaKernel execution with an outer ACL Graph.
- `megakernel-native`: MegaKernel submits the prepared Decode sequence
  directly. The script passes `--model-execution-backend megakernel-native`.
- `native`: regular SGLang, without the MegaKernel interceptor.

For a manual launch, build a compatible MegaKernel checkout and preload its
`libmk_capture.so`. Use the full SGLang Decode runner
(`--cuda-graph-backend-decode full`) and BS1
(`--cuda-graph-max-bs-decode 1`). The full runner still provides fixed buffers
and shape selection; it does not create the outer ACL Graph in direct mode.
The direct mode requires the existing WeLM settings
`WELM_NPU_USE_MK_TARGET_DECODE=1`, `WELM_NPU_USE_FLASH_ATTN=1`,
`SGLANG_WELMV4_MK_DECODE_FAST_PATH=1`, `SGLANG_WELMV4_DECODE_RS_AG=1`, and
`SGLANG_FORCE_COARSE_WAR_BARRIER=1`. The last setting keeps scheduler
writes ordered after the forward's reads of shared KV metadata. Use a single
stream; MTP and the fixed EP AllGather experiment are not supported in direct
mode. This backend does not change Prefill.

The replay audit checks all four ranks and 48 layers, records the observed
MegaKernel scope and step counts, and rejects Eager fallback. With
`megakernel-native`, it also rejects the outer ACL Graph marker. Use the
token IDs in `bench.jsonl` to compare numerical outputs across the arms.

## Single-node Ascend 950 DP8EP8

`run_dp8.sh` starts a foreground Decode worker with TP8/DP8/EP8, stock
Ascend DeepEP, one stream, no MTP, and captured rows 1/2/4/8 **per DP rank**.
It inherits the eight-device scope assigned by your environment; it does
not reserve or select cards. The global request cap is 64 and DeepEP's
per-rank Dispatch cap is 8. Optional DP4-only fusions and the experimental
fixed AllGather transport are off.

Set `WELM_DP8_ROUTER_FUSION=1` on the `megakernel-native` launch to group
`MatMulV3`, `MoeGatingTopK`, and `MaskedFill` into one Router segment per
layer for every supported Decode bucket (B1/B2/B4/B8). The setting is read
when building the graphs. Leave it unset or set it to `0` to keep the
original SDK packet policy. Prefill and the native backend are unchanged.

Use a compatible stock `deep_ep` package (`DEEPEP_PACKAGE_DIR`, containing
`vendors/hwcomputing/op_api/lib`) and source the SHMEM install environment.
Set `CANN_DIR` if it is not already selected, and `WELM_MODEL` to the model
directory. For `megakernel-native`, set `WELM_MEGAKERNEL_SRC` to the built
matching checkout and `ASCEND_OPP_PATH` to a **private** CANN operator tree
with writable `static_kernel/ai_core`. Do not point it at the shared toolkit's
operator registry. `WELM_BENCH_KV_TOKENS` (default 196608) and
`WELM_CONTEXT_LENGTH` (default 18432) can be adjusted to your available HBM.
The script defaults to fake transfer for isolated Decode testing; for real
P/D deployment set `WELM_DECODE_TRANSFER_BACKEND` to match the Prefill peer.

```bash
WELM_DP8_ROUTER_FUSION=1 bash run_dp8.sh megakernel-native > dp8-mk.log 2>&1
# After the service is ready, in a separate terminal:
python3 smoke_dp8.py --serve-log dp8-mk.log | tee dp8-smoke.jsonl
```

The smoke test is for fake transfer only. It sends rank-pinned, synchronized
waves for global C1/4/8/16/32/64, plus two skewed C8 waves with idle ranks:
`skew` (maximum 3 real rows, padded to B4) and `skew8` (maximum 5 real rows,
padded to B8). It
requires valid SSE outputs and at least three consecutive matching per-rank
row vectors and buckets per case, plus all-rank capture/replay evidence for
MegaKernel. It uses per-rank log order within an isolated workload, not a
device timestamp or shared step ID. Offered C64 does not count as a pass
unless all eight ranks actually run eight rows together. Keep the server
dedicated to this test and graph telemetry enabled (the script's default).
For a shorter check use `--cases 1,4,8`; for the corresponding native graph
smoke use `run_dp8.sh native` and `smoke_dp8.py --backend native` with a fresh
log. Fake TTFT is not a Prefill measurement, and per-step telemetry makes
these timings diagnostic rather than comparable performance results.
Set `SGLANG_LOG_DECODE_GRAPH_KEY=0` for formal performance measurements.
