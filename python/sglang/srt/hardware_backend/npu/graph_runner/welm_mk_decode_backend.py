"""Model-level WeLM Decode graph; reuse SGLang's fixed buffers and DP shape protocol."""

import logging
import os
from dataclasses import fields
from pathlib import Path
import shutil
import tempfile

import torch

from sglang.srt.hardware_backend.npu.graph_runner.npu_cudagraph_backend import (
    NPUCudaGraphBackend,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.welmv4_mk_triton import target_native_capture
from sglang.srt.utils.nvtx_utils import profile_range

logger = logging.getLogger(__name__)


class WelmMKDecodeBackend(NPUCudaGraphBackend):
    """Substitute the executor, not the collective shape selection or KV pool."""

    def __init__(self, runner):
        super().__init__(runner)
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
        runner.model_runner.welm_graph_read_done_enabled = False
        self._attn_backend = runner.attn_backend
        self._metadata = {}
        self._replays = {}
        self._ineligible_shapes = set()

    # Keep reusable schedule buffers mutable from the scheduler's replay path.
    @torch.no_grad()
    @target_native_capture()
    def capture_one(
        self, shape_key, forward_fn, capture_inputs=None, post_warmup_hook=None
    ):
        from megakernel._fixed import capture_fixed, prepare_static
        from megakernel._capture_access import _ACTIVE
        from megakernel.ops.tensor_concat import LowerConcats

        # The runner owns and initializes the fixed inputs, attention metadata,
        # scratch KV slots, and DeepEP mode. Capture is entered in the same
        # order on all ranks, including those without local decode requests.
        output_fields = tuple(field.name for field in fields(LogitsProcessorOutput))

        def capture_outputs():
            output = forward_fn()
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
        metadata.welm_mk_schedule_recipes = {}
        metadata.welm_mk_schedule_bootstrap = True
        try:
            # Discover the full/SWA scheduling geometries with the runner's
            # dummy batch, with no active MK recorder.
            forward_fn()
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
            prepare_static(forward_fn, path)
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
                with LowerConcats():
                    graph = capture_fixed(capture_outputs, path, prepared=True)
            finally:
                for hook in hooks:
                    hook.remove()
        except Exception:
            if root := os.environ.get("WELM_MK_CAPTURE_DIAGNOSTICS"):
                dest = Path(root) / f"{len(self._graphs)}-{os.getpid()}"
                dest.mkdir(parents=True, exist_ok=True)
                for artifact in path.glob("capture_failure_*.json"):
                    shutil.copy2(artifact, dest / artifact.name)
            raise
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

        self._graphs[shape_key] = graph
        self._outputs[shape_key] = LogitsProcessorOutput(
            **dict(zip(output_fields, graph.outputs, strict=True))
        )
        self._metadata[shape_key] = metadata
        self._replays[shape_key] = 0

    def replay(self, shape_key, static_forward_batch, **kwargs):
        graph = self._graphs[shape_key]
        metadata = self._metadata[shape_key]
        if self._attn_backend.forward_metadata is not metadata:
            raise RuntimeError("WeLM MK Decode metadata identity changed")
        with profile_range("welm.mk.attn_metadata"):
            for key, (dims, common) in metadata.welm_mk_schedule_recipes.items():
                updated = self._attn_backend._welm_flash_attn_metadata(*dims, **common)
                metadata.welm_flash_schedules[key].copy_(updated, non_blocking=True)
        with profile_range("welm.mk.replay"):
            graph.replay()
        count = self._replays[shape_key] + 1
        self._replays[shape_key] = count
        if count <= 3 or count % 128 == 0:
            logger.info("welm-mk-target replay shape=%s count=%s", shape_key, count)
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
