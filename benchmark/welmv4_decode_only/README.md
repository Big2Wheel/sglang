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
