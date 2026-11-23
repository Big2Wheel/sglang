"""Model-level WeLM Decode graph; reuse SGLang's fixed buffers and DP shape protocol."""

import json
import logging
import os
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path

import torch
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.graph_runner.npu_cudagraph_backend import (
    NPUCudaGraphBackend,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.welmv4_mk_triton import target_native_capture
from sglang.srt.utils.nvtx_utils import profile_range

logger = logging.getLogger(__name__)


class WelmMKDecodeBackend(NPUCudaGraphBackend):
    """Substitute the executor, not the collective shape selection or KV pool."""

    def __init__(self, runner, *, direct_replay=False):
        super().__init__(runner)
        self._direct_replay = direct_replay
        args = runner.model_runner.server_args
        architectures = runner.model_runner.model_config.hf_config.architectures or []
        if (
            "WeLMV4MoeForCausalLM" not in architectures
            or runner.model_runner.is_draft_worker
        ):
            raise ValueError("Model-level MegaKernel Decode requires the WeLMV4 target")
        if args.enable_memory_saver or self._enable_torch_compile:
            raise ValueError(
                "Model-level MegaKernel does not support memory saver or torch compile"
            )
        if runner.enable_pdmux:
            raise ValueError(
                "Model-level MegaKernel does not support multiple decode streams"
            )
        is_verify = not runner.model_runner.spec_algorithm.is_none()
        if direct_replay and is_verify:
            raise ValueError("Direct WeLM MegaKernel replay requires Decode without MTP")
        if is_verify and not (
            os.environ.get("WELM_NPU_USE_MK_TARGET_VERIFY") == "1"
            and runner.capture_forward_mode.is_target_verify()
            and getattr(
                runner.model_runner.model_config.hf_config,
                "num_nextn_predict_layers",
                0,
            )
            == 1
            and (args.speculative_algorithm or "").upper() in {"EAGLE", "NEXTN"}
            and args.speculative_eagle_topk == 1
            and args.speculative_num_draft_tokens == args.speculative_num_steps + 1
        ):
            raise ValueError("MegaKernel MTP requires opt-in linear WeLM Target Verify")
        self._phase = "TARGET_VERIFY" if is_verify else "DECODE"
        if os.environ.get("WELM_NPU_USE_MK_FLASH_DECODE") == "1":
            raise ValueError(
                "Do not nest per-layer MegaKernel attention in the target graph"
            )
        if os.environ.get("WELM_NPU_USE_FLASH_ATTN") != "1":
            raise ValueError(
                "Model-level MegaKernel requires device-side FlashAttn graph metadata"
            )
        self._layer_count = runner.model_runner.model_config.hf_config.num_hidden_layers
        self._decoder_layers = tuple(runner.model_runner.model.model.layers)
        if len(self._decoder_layers) != self._layer_count:
            raise ValueError(
                "Model-level MegaKernel requires the complete local decoder stack"
            )
        self._directory = tempfile.TemporaryDirectory(prefix="welm-mk-target-")
        # NpuGraphReadDone's flag handshake belongs to torch.npu.NPUGraph.
        # Keep the scheduler WAR barrier, with its whole-forward wait fallback.
        self.uses_fixed_decode_prologue = envs.SGLANG_WELMV4_MK_DECODE_FAST_PATH.get()
        if direct_replay and not self.uses_fixed_decode_prologue:
            raise ValueError("Direct WeLM MegaKernel replay requires fixed Decode metadata")
        if self.uses_fixed_decode_prologue and (
            is_verify or not envs.SGLANG_WELMV4_DECODE_RS_AG.get()
        ):
            raise ValueError("Fixed WeLM MegaKernel Decode requires Decode RS/AG without MTP")
        self._native_router_sigmoid = envs.SGLANG_WELMV4_MK_DECODE_TOPK_MODE3.get()
        if self._native_router_sigmoid or envs.SGLANG_WELMV4_MK_DECODE_FLASH_KV_BOUND.get():
            if (
                not self.uses_fixed_decode_prologue
                or not args.enable_dp_attention
                or args.tp_size != 4
                or args.dp_size != 4
                or "950" not in self._device_module.get_device_name(self._device_id)
            ):
                raise ValueError("WeLM Decode operator A/B requires fixed Ascend 950 DP4 Decode")
        self._fuse_pre_moe_mask_norm = (
            envs.SGLANG_WELMV4_MK_DECODE_PRE_MOE_MASK_NORM.get()
        )
        self._mask_norm_layers = (
            tuple(i for i, layer in enumerate(self._decoder_layers) if layer.self_attn.use_o_norm)
            if self._fuse_pre_moe_mask_norm else ()
        )
        if self._fuse_pre_moe_mask_norm and (
            not self.uses_fixed_decode_prologue
            or not args.enable_dp_attention
            or args.tp_size != 4
            or args.dp_size != 4
            or runner.model_runner.ps.moe_ep_size != 4
            or "950" not in self._device_module.get_device_name(self._device_id)
            or not self._mask_norm_layers
        ):
            raise ValueError("Pre-MoE O-Norm mask requires fixed Ascend 950 DP4 Decode")
        self._fuse_decode_kv_prologue = envs.SGLANG_WELMV4_MK_DECODE_KV_PROLOGUE.get()
        if self._fuse_decode_kv_prologue and (
            not self.uses_fixed_decode_prologue
            or not args.enable_dp_attention
            or args.tp_size != 4
            or args.dp_size != 4
            or runner.model_runner.ps.moe_ep_size != 4
            or "950" not in self._device_module.get_device_name(self._device_id)
            or args.disaggregation_mode != "decode"
            or args.disaggregation_transfer_backend != "fake"
        ):
            raise ValueError("WeLM fused KV prologue requires fixed Ascend 950 DP4EP4 fake Decode")
        self._ep_allgather_group = None
        if envs.SGLANG_WELMV4_MK_EP_ALLGATHER.get():
            if direct_replay:
                raise ValueError("WeLM EP AllGather requires the ACL Graph replay backend")
            if (
                not self._fuse_decode_kv_prologue
                or not self._native_router_sigmoid
                or len(self._mask_norm_layers) != 47
                or self._layer_count != 48
                or args.moe_a2a_backend != "deepep"
                or args.enable_eplb
                or args.ep_num_redundant_experts
                or args.enable_two_batch_overlap
                or args.enable_single_batch_overlap
                or args.enable_lora
                or not envs.SGLANG_WELMV4_MK_DECODE_FLASH_KV_BOUND.get()
                or not envs.SGLANG_WELMV4_MK_DECODE_FLASH_FULL_UNMASKED.get()
            ):
                raise ValueError(
                    "WeLM EP AllGather requires the fixed 48-layer DP4EP4 "
                    "MegaKernel BF16 Decode configuration with fused KV prologue"
                )
            from sglang.srt.distributed.parallel_state import get_moe_ep_group

            self._ep_allgather_group = get_moe_ep_group().device_group
        runner.model_runner.welm_graph_read_done_enabled = False
        self._attn_backend = runner.attn_backend
        self._metadata = {}
        self._page_prologue_batches = {}
        self._replays = {}
        self._ineligible_shapes = set()

    @contextmanager
    def capture_session(self, stream):
        if not self._direct_replay:
            with super().capture_session(stream):
                yield
            return
        # SGLang still owns the capture stream and shape buckets. MK owns
        # its allocations; there is no Torch graph pool in direct mode.
        self._capture_stream = stream
        try:
            yield
        finally:
            self._capture_stream = None

    # Keep reusable schedule buffers mutable from the scheduler's replay path.
    @torch.no_grad()
    @target_native_capture()
    def capture_one(
        self, shape_key, forward_fn, capture_inputs=None, post_warmup_hook=None
    ):
        if self._ep_allgather_group is None:
            return self._capture_one(
                shape_key, forward_fn, capture_inputs, post_warmup_hook, None
            )
        if (
            shape_key.size != 1
            or shape_key.stream_idx is not None
            or shape_key.variant_label is not None
        ):
            raise ValueError("WeLM EP AllGather requires a fixed single-row capture")
        from sglang.srt.layers.moe.token_dispatcher.welm_ep_allgather import (
            welm_ep_allgather_scope,
        )

        with welm_ep_allgather_scope(
            True, group=self._ep_allgather_group, expected_layers=self._layer_count
        ) as ep_scope:
            return self._capture_one(
                shape_key, forward_fn, capture_inputs, post_warmup_hook, ep_scope
            )

    def _capture_one(
        self, shape_key, forward_fn, capture_inputs, post_warmup_hook, ep_scope
    ):
        from megakernel._capture_access import _ACTIVE
        from megakernel._fixed import capture_fixed, prepare_static
        from megakernel.ops.tensor_concat import LowerConcats

        scope_options = (
            {"welm_mask_norm_layers": self._mask_norm_layers} if self._mask_norm_layers else {}
        )
        # The runner owns and initializes the fixed inputs, attention metadata,
        # scratch KV slots, and DeepEP mode. Capture is entered in the same
        # order on all ranks, including those without local decode requests.
        output_fields = tuple(field.name for field in fields(LogitsProcessorOutput))

        def capture_ep_forward():
            with ep_scope.forward_pass():
                return forward_fn()

        capture_forward = forward_fn if ep_scope is None else capture_ep_forward

        def capture_outputs():
            output = capture_forward()
            if not isinstance(output, LogitsProcessorOutput):
                raise TypeError(
                    "Expected WeLM Decode logits at the model graph boundary"
                )
            # The MK recorder walks standard pytrees. Hand it a tuple rather
            # than an opaque output dataclass so tensor bindings are retained.
            return tuple(getattr(output, name) for name in output_fields)

        path = Path(self._directory.name) / str(len(self._graphs))
        self._device_module.synchronize()
        self._tp_group.barrier()
        metadata = self._attn_backend.forward_metadata
        page_prologue = self.uses_fixed_decode_prologue
        if page_prologue and (
            not metadata.welm_decode_metadata_in_graph
            or capture_inputs is None
            or not capture_inputs.forward_mode.is_decode_or_idle()
            or capture_inputs.spec_info is not None
            or capture_inputs.seq_lens.shape != (1,)
            or capture_inputs.out_cache_loc is None
            or capture_inputs.out_cache_loc.shape != (1,)
            or metadata.seq_lens is not capture_inputs.seq_lens
        ):
            raise RuntimeError("Device page prologue requires an ordinary, fixed single-row Decode capture")
        metadata.welm_mk_schedule_recipes = {}
        metadata.welm_mk_schedule_bootstrap = True
        try:
            # Discover the full/SWA scheduling geometries with the runner's
            # dummy batch, with no active MK recorder.
            capture_forward()
        finally:
            metadata.welm_mk_schedule_bootstrap = False
        if not metadata.welm_mk_schedule_recipes:
            raise RuntimeError("Did not discover any WeLM Decode FlashAttn schedules")
        if post_warmup_hook is not None:
            post_warmup_hook()
        self._device_module.synchronize()
        self._tp_group.barrier()
        logger.info(
            "welm-mk-target prepare shape=%s layers=%s phase=%s",
            shape_key,
            self._layer_count,
            self._phase,
        )
        with LowerConcats():
            prepare_static(capture_forward, path)
        if post_warmup_hook is not None:
            post_warmup_hook()
        self._device_module.synchronize()
        self._tp_group.barrier()
        logger.info(
            "welm-mk-target record shape=%s layers=%s", shape_key, self._layer_count
        )
        ranges = {}
        hooks = []
        for layer_id, layer in enumerate(self._decoder_layers):

            def enter(_layer, _inputs, *, layer_id=layer_id):
                capture = _ACTIVE.get()
                if capture is not None:
                    ranges[layer_id] = [capture.launches.checkpoint()]

            def leave(_layer, _inputs, _outputs, *, layer_id=layer_id):
                capture = _ACTIVE.get()
                if capture is not None:
                    ranges[layer_id].append(capture.launches.checkpoint())

            hooks.append(layer.register_forward_pre_hook(enter))
            hooks.append(layer.register_forward_hook(leave))
        try:
            try:
                # The native dummy warmup above populated the stable metadata
                # buffers. Exclude only the producers from the MK recorder:
                # replay enqueues them before the model, via ACL or directly.
                metadata.welm_mk_defer_page_prologue = page_prologue
                with LowerConcats():
                    graph = capture_fixed(
                        capture_outputs,
                        path,
                        prepared=True,
                        welm_decode_layers=self._layer_count if page_prologue else 0,
                        **(
                            {"welm_native_router_sigmoid": True}
                            if page_prologue and self._native_router_sigmoid
                            else {}
                        ),
                        **(
                            {"welm_ep_allgather_group": self._ep_allgather_group}
                            if ep_scope is not None
                            else {}
                        ),
                        **scope_options,
                    )
            finally:
                metadata.welm_mk_defer_page_prologue = False
                for hook in hooks:
                    hook.remove()
        except Exception:
            if root := os.environ.get("WELM_MK_CAPTURE_DIAGNOSTICS"):
                dest = Path(root) / f"{len(self._graphs)}-{os.getpid()}"
                dest.mkdir(parents=True, exist_ok=True)
                for artifact in path.glob("capture_failure_*.json"):
                    shutil.copy2(artifact, dest / artifact.name)
            raise
        if self._direct_replay and not callable(getattr(graph, "replay_direct", None)):
            graph.reset()
            raise RuntimeError("Rebuild MegaKernel with WeLM direct replay support")
        summary = graph.execution_summary
        fused_indices = {index for plan in graph.plans for index in plan.task_indices}
        layer_launches = {
            layer: (
                end - start,
                sum(index in fused_indices for index in range(start, end)),
            )
            for layer, (start, end) in sorted(ranges.items())
        }
        logger.info(
            "welm-mk-target recorded shape=%s layer_launches(total,fused)=%s coverage=%s",
            shape_key,
            layer_launches,
            summary,
        )
        if (
            len(layer_launches) != self._layer_count
            or not all(total and fused for total, fused in layer_launches.values())
            or not graph.fused_operator_count
            or graph.updates
        ):
            graph.reset()
            raise RuntimeError(
                f"WeLM target graph is missing native coverage: {summary}"
            )
        if graph.host_boundary_count:
            graph.reset()
            raise RuntimeError(
                f"WeLM target graph contains unsupported host work: {summary}"
            )
        if ep_scope is not None:
            try:
                ep_scope.assert_complete()
                proof = graph.audit_welm_fixed_allgather(ranges)
                # Hold the fixed-address workspaces through every ACL replay.
                graph._welm_ep_allgather_owner = ep_scope
            except Exception:
                graph.reset()
                raise
            logger.info(
                "welm-mk fixed allgather audited: %s",
                json.dumps(proof, separators=(",", ":")),
            )
        elif self._fuse_decode_kv_prologue:
            try:
                fused, scopes, steps = graph.audit_welm_fixed_kv_prologue(ranges)
            except Exception:
                graph.reset()
                raise
            logger.info(
                "welm-mk fixed KV prologue audited: layers=%d fused=%d scopes=%d steps=%d",
                self._layer_count, fused, scopes, steps,
            )

        self._graphs[shape_key] = graph
        self._outputs[shape_key] = LogitsProcessorOutput(
            **dict(zip(output_fields, graph.outputs, strict=True))
        )
        self._metadata[shape_key] = metadata
        if page_prologue:
            self._page_prologue_batches[shape_key] = capture_inputs
        self._replays[shape_key] = 0

    def replay(self, shape_key, static_forward_batch, **kwargs):
        graph = self._graphs[shape_key]
        metadata = self._metadata[shape_key]
        if self._attn_backend.forward_metadata is not metadata:
            raise RuntimeError("WeLM MK Decode metadata identity changed")
        captured_inputs = self._page_prologue_batches.get(shape_key)

        def update_metadata():
            if captured_inputs is not None:
                # Captured tensors are the buffer registry views used by the
                # model; load_batch writes current values there before replay.
                self._attn_backend._prepare_welm_decode_metadata_device(captured_inputs)
            with profile_range("welm.mk.attn_metadata"):
                for key, (dims, common) in metadata.welm_mk_schedule_recipes.items():
                    updated = self._attn_backend._welm_flash_attn_metadata(*dims, **common)
                    metadata.welm_flash_schedules[key].copy_(updated, non_blocking=True)

        if captured_inputs is None and not self._direct_replay:
            update_metadata()
        with profile_range("welm.mk.replay"):
            if self._direct_replay:
                # The MK replay lock and stream ordering cover both metadata
                # producers and submission of the prepared device sequence.
                graph.replay_direct(before_launch=update_metadata)
            elif captured_inputs is None:
                graph.replay()
            else:
                graph.replay(before_acl_capture=update_metadata)
        count = self._replays[shape_key] + 1
        self._replays[shape_key] = count
        if count <= 3 or count % 128 == 0:
            logger.info(
                "welm-mk-target replay shape=%s count=%s backend=%s",
                shape_key,
                count,
                "mk-direct" if self._direct_replay else "acl",
            )
        return self._outputs[shape_key]

    def replay_with_input_update(self, *args, **kwargs):
        raise RuntimeError("WeLM target graph must use device-side attention metadata")

    def report_ineligible(self, forward_batch, shape_key):
        """Make graph misses visible without altering collective admission."""
        disposition = (
            shape_key,
            forward_batch.forward_mode.name,
            bool(forward_batch.can_run_dp_cuda_graph),
        )
        if disposition not in self._ineligible_shapes:
            self._ineligible_shapes.add(disposition)
            logger.warning(
                "welm-mk-target eager fallback shape=%s mode=%s dp_graph_allowed=%s",
                *disposition,
            )

    def cleanup(self):
        for graph in self._graphs.values():
            graph.reset()
        super().cleanup()
        self._replays.clear()
        self._metadata.clear()
        self._ineligible_shapes.clear()
        self._directory.cleanup()
