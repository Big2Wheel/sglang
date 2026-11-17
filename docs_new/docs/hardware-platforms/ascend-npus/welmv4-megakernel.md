# Experimental WeLMV4 MegaKernel execution on Ascend 950

This integration was validated on Ascend 950DT with CANN 9.2.1, synthetic
BF16 weights, and DP8EP8. It requires a matching MegaKernel runtime with
ACL HostArgs capture and recorded Triton replay support. Build it for the
installed SDK and preload its capture library before starting SGLang.

- Use `WELM_NPU_USE_FLASH_ATTN=1` for the Ascend FlashAttn path.
- Set `WELM_NPU_USE_MK_TARGET_DECODE=1` to opt in to model-level replay.
  Keep `WELM_NPU_USE_MK_FLASH_DECODE` disabled when using that executor.
- With stock Ascend Triton, set `MK_FIXED_REPLAY_TRITON_GLOBALS=1`.
  Leave `WELM_MK_TRITON_CHILDREN` unset unless using a compatible
  child-kernel compiler.
- To use model-level replay during NextN, also set
  `WELM_NPU_USE_MK_TARGET_VERIFY=1`. Use TopK 1 and
  `num_draft_tokens = num_steps + 1`. Draft and Draft Extend stay native.
- Set `WELM_STAGE7_FUSED_VERIFY_METADATA=1` to fuse linear Verify metadata
  updates. It works with native Verify as well as with MegaKernel Verify.
- For routed-expert export on the tested DP8EP8 layout, enable
  `WELM_LOCAL_EP_ROUTE_AUDIT=1` so each DP rank exports its own MoE rows.

The NextN validation used Host Embedding, `ASCEND_LAUNCH_BLOCKING=0`,
`TASK_QUEUE_ENABLE=1`, and `SGLANG_SET_CPU_AFFINITY=1`.

Do not disable scheduler DP/MLP synchronization. Prefill, KV allocation,
and speculative acceptance remain under SGLang control. This path does not
opt in to MegaKernel by default. The synthetic workload did not demonstrate
an end-to-end speedup over the native Verify graph, nor establish real-model
quality or acceptance rates.

The NPU metadata regression is located at
`test/registered/unit/npu/attention/test_welm_verify_metadata.py`.
