"""Opt-in fixed-buffer FlashAttn Decode; SGLang still owns and writes KV."""

from collections import OrderedDict
import logging
import os
from pathlib import Path
import tempfile

import torch

logger = logging.getLogger(__name__)


class WelmMKFlash:
    def __init__(self, *, max_plans=None):
        self._plans = OrderedDict()
        if max_plans is None:
            max_plans = int(os.environ.get("WELM_NPU_MK_FLASH_MAX_PLANS", "32"))
        if not 1 <= max_plans <= 256:
            raise ValueError(
                "WeLM fixed FlashAttn plan cache limit must be between 1 and 256"
            )
        self._max_plans = max_plans
        self._directory = tempfile.TemporaryDirectory(prefix="welm-mk-decode-")
        self._next_plan = 0

    def forward(
        self, flash, query, key, value, *, layer_id, common, table_width=None, **kwargs
    ):
        from megakernel._fixed import capture_fixed

        # Work on the same execution stream as the KV write. A different stream
        # (or a replaced KV storage) must never borrow this snapshot's addresses.
        stream = torch.npu.current_stream(query.device).npu_stream
        dynamic = {
            "query": query,
            **{name: kwargs[name] for name in ("block_table", "metadata")},
            **{
                name: common[name]
                for name in ("cu_seqlens_q", "seqused_q", "seqused_kv")
            },
        }
        table = dynamic["block_table"]
        if table_width is None:
            table_width = table.shape[1]
        if table.shape[1] > table_width:
            raise RuntimeError(
                "WeLM FlashAttn paged KV exceeded the configured table width"
            )
        table_shape = (table.shape[0], table_width)
        static_common = {
            name: value for name, value in common.items() if name not in dynamic
        }
        static_kwargs = {
            name: value for name, value in kwargs.items() if name not in dynamic
        }
        bindings = (key, value, static_kwargs["sinks"], static_kwargs["attn_mask"])
        ident = (
            layer_id,
            query.device,
            stream,
            tuple(static_common.items()),
            static_kwargs["softmax_scale"],
            tuple(
                None
                if t is None
                else (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype)
                for t in bindings
            ),
            tuple(
                (n, table_shape if n == "block_table" else tuple(t.shape), t.dtype)
                for n, t in dynamic.items()
            ),
        )

        def copy_table(target):
            target[:, : table.shape[1]].copy_(table, non_blocking=True)
            target[:, table.shape[1] :].fill_(-1)

        entry = self._plans.get(ident)
        if entry is None:
            if len(self._plans) >= self._max_plans:
                _old_key, (old_plan, _) = self._plans.popitem(last=False)
                old_plan.reset()
            buffers = {
                name: tensor.contiguous().clone()
                for name, tensor in dynamic.items()
                if name != "block_table"
            }
            buffers["block_table"] = torch.empty(
                table_shape, device=table.device, dtype=table.dtype
            )
            copy_table(buffers["block_table"])
            launch_kwargs = dict(
                static_kwargs,
                block_table=buffers["block_table"],
                metadata=buffers["metadata"],
            )
            launch_common = dict(
                static_common,
                **{
                    name: buffers[name]
                    for name in ("cu_seqlens_q", "seqused_q", "seqused_kv")
                },
            )

            def run():
                return flash(
                    buffers["query"], key, value, **launch_kwargs, **launch_common
                )[0]

            self._next_plan += 1
            path = Path(self._directory.name) / str(self._next_plan)
            plan = capture_fixed(run, path)
            if (
                not plan.fused_operator_count
                or plan.host_boundary_count
                or plan.execution_summary["native_globals"]
            ):
                summary = plan.execution_summary
                plan.reset()
                raise RuntimeError(
                    f"WeLM FlashAttn Decode must use a fused MK scope: {summary}"
                )
            if plan.updates:
                plan.reset()
                raise RuntimeError(
                    "WeLM FlashAttn Decode unexpectedly requires host argument updates"
                )
            plan._warmup_static()
            self._plans[ident] = (plan, buffers)
            logger.info(
                "welm-mk capture layer=%d win_left=%d q=%s table=%s kv=%s fused=%d scopes=%d",
                layer_id,
                static_common["win_left"],
                tuple(query.shape),
                table_shape,
                key.data_ptr(),
                plan.fused_operator_count,
                plan.scope_count,
            )
        else:
            plan, buffers = entry
            self._plans.move_to_end(ident)
            for name, tensor in dynamic.items():
                if name == "block_table":
                    copy_table(buffers[name])
                else:
                    buffers[name].copy_(tensor, non_blocking=True)

        result = plan.replay()
        if plan.replayed <= 3 or plan.replayed % 128 == 0:
            logger.info(
                "welm-mk replay layer=%d win_left=%d batch=%d replay=%d",
                layer_id,
                static_common["win_left"],
                query.shape[0],
                plan.replayed,
            )
        return result
