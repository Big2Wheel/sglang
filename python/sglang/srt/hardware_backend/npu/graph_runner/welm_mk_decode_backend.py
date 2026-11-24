"""WeLMV4 Decode execution through a prepared native MegaKernel sequence."""

import logging
import tempfile
from contextlib import nullcontext
from dataclasses import fields

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.runner_backend.base_cuda_graph_backend import (
    BaseCudaGraphBackend,
)

logger = logging.getLogger(__name__)


class WelmMKDecodeBackend(BaseCudaGraphBackend):
    """Keep the runner's fixed inputs, KV buffers and distributed admission."""

    def __init__(self, runner):
        model_runner = runner.model_runner
        if model_runner.is_draft_worker or not model_runner.spec_algorithm.is_none():
            raise ValueError("MegaKernel requires ordinary WeLMV4 target Decode")
        self._layers = tuple(model_runner.model.model.layers)
        layer_count = model_runner.model_config.hf_config.num_hidden_layers
        if not self._layers or len(self._layers) != layer_count:
            raise ValueError("MegaKernel requires the entire WeLMV4 decoder stack")
        self._attn_backend = runner.attn_backend
        if not self._attn_backend.use_welm_flash_attn:
            raise ValueError("MegaKernel requires native WeLMV4 FlashAttn")
        self._device_module = runner.device_module
        self._tp_group = model_runner.tp_group
        # Native replay has no ACL Graph read-done ticket. The scheduler's
        # existing whole-forward dependency protects writes to shared buffers.
        model_runner.welm_graph_read_done_enabled = False
        self._directory = tempfile.TemporaryDirectory(prefix="welm-mk-decode-")
        self._graph = self._shape = self._metadata = self._output = None
        self._replays = 0

    def capture_session(self, stream):
        # The runner has already selected the stream. No ACL Graph pool.
        return nullcontext()

    @torch.no_grad()
    def capture_one(
        self, shape_key, forward_fn, capture_inputs=None, post_warmup_hook=None
    ):
        if (
            shape_key.size != 1
            or shape_key.stream_idx is not None
            or shape_key.variant_label is not None
            or self._graph is not None
        ):
            raise ValueError("MegaKernel requires one single-row Decode shape")
        metadata = self._attn_backend.forward_metadata
        if metadata is None or metadata.welm_decode_metadata_in_graph:
            raise RuntimeError("MegaKernel requires the runner's out-of-graph metadata")

        from megakernel._capture_access import _ACTIVE
        from megakernel._fixed import capture_fixed, prepare_static
        from megakernel.ops.tensor_concat import LowerConcats

        output_fields = tuple(field.name for field in fields(LogitsProcessorOutput))

        def capture_outputs():
            output = forward_fn()
            if not isinstance(output, LogitsProcessorOutput):
                raise TypeError("Expected WeLMV4 logits at the MegaKernel boundary")
            return tuple(getattr(output, field_name) for field_name in output_fields)

        def sync_ranks():
            self._device_module.synchronize()
            self._tp_group.barrier()

        sync_ranks()
        # The recorder sees only the schedule consumers; the same schedule
        # tensors will be refreshed in place on each real Decode step.
        metadata.welm_mk_schedule_recipes = {}
        metadata.welm_mk_schedule_bootstrap = True
        try:
            forward_fn()
        finally:
            metadata.welm_mk_schedule_bootstrap = False
        if not metadata.welm_mk_schedule_recipes:
            raise RuntimeError("No WeLMV4 Decode FlashAttn schedules were found")
        if post_warmup_hook is not None:
            post_warmup_hook()
        sync_ranks()
        with LowerConcats():
            prepare_static(forward_fn, self._directory.name)
        if post_warmup_hook is not None:
            post_warmup_hook()
        sync_ranks()

        launches = {layer_id: [] for layer_id in range(len(self._layers))}

        def record_boundary(layer_id):
            def hook(*_):
                capture = _ACTIVE.get()
                if capture is not None:
                    launches[layer_id].append(capture.launches.checkpoint())

            return hook

        hooks = []
        try:
            for layer_id, layer in enumerate(self._layers):
                hook = record_boundary(layer_id)
                hooks.append(layer.register_forward_pre_hook(hook))
                hooks.append(layer.register_forward_hook(hook))
            with LowerConcats():
                graph = capture_fixed(
                    capture_outputs, self._directory.name, prepared=True
                )
        finally:
            for hook in hooks:
                hook.remove()

        try:
            if not callable(getattr(graph, "replay_direct", None)):
                raise TypeError("Build MegaKernel with native direct replay support")
            summary = graph.execution_summary
            if graph.host_boundary_count or graph.updates:
                raise RuntimeError(
                    f"WeLMV4 needs a fixed device-only MegaKernel sequence: {summary}"
                )
            fused = {index for plan in graph.plans for index in plan.task_indices}
            if any(len(bounds) != 2 for bounds in launches.values()):
                raise RuntimeError(
                    "MegaKernel did not record all WeLMV4 decoder layers"
                )
            layer_launches = {
                layer: (end - start, sum(i in fused for i in range(start, end)))
                for layer, (start, end) in launches.items()
            }
            if not all(total and count for total, count in layer_launches.values()):
                raise RuntimeError(
                    f"WeLMV4 layers lack native MegaKernel coverage: {layer_launches}"
                )
            output = LogitsProcessorOutput(
                **dict(zip(output_fields, graph.outputs, strict=True))
            )
        except Exception:
            graph.reset()
            raise
        self._graph, self._shape, self._metadata, self._output = (
            graph,
            shape_key,
            metadata,
            output,
        )
        logger.info(
            "welm-mk-target recorded shape=%s layer_launches(total,fused)=%s coverage=%s",
            shape_key,
            layer_launches,
            summary,
        )

    def can_run(self, forward_batch, shape_key):
        return (
            self._graph is not None
            and shape_key == self._shape
            and forward_batch.forward_mode.is_decode_or_idle()
            and forward_batch.spec_info is None
        )

    def replay_session(self):
        return nullcontext()

    def replay(self, shape_key, static_forward_batch, **kwargs):
        if not self.can_run(static_forward_batch, shape_key):
            raise RuntimeError("No prepared MegaKernel sequence for this Decode batch")
        metadata = self._metadata
        if self._attn_backend.forward_metadata is not metadata:
            raise RuntimeError("WeLMV4 Decode metadata bindings changed")

        def refresh_schedules():
            for key, (dims, attrs) in metadata.welm_mk_schedule_recipes.items():
                updated = self._attn_backend._welm_flash_attn_metadata(*dims, **attrs)
                schedule = metadata.welm_flash_schedules[key]
                if (
                    updated.shape != schedule.shape
                    or updated.dtype != schedule.dtype
                    or updated.device != schedule.device
                ):
                    raise RuntimeError("WeLMV4 FlashAttn schedule layout changed")
                schedule.copy_(updated, non_blocking=True)

        self._graph.replay_direct(before_launch=refresh_schedules)
        self._replays += 1
        if self._replays <= 3 or self._replays % 128 == 0:
            logger.info(
                "welm-mk-target replay shape=%s count=%s backend=mk-direct",
                shape_key,
                self._replays,
            )
        return self._output

    def replay_with_input_update(self, *args, **kwargs):
        raise RuntimeError("MegaKernel Decode requires device-side FlashAttn metadata")

    def cleanup(self):
        if self._graph is not None:
            self._graph.reset()
            self._metadata.welm_mk_schedule_recipes = None
            self._metadata.welm_flash_schedules.clear()
        self._graph = self._shape = self._metadata = self._output = None
        self._directory.cleanup()
