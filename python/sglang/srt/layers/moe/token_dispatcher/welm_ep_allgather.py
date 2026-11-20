"""Lifetime and pairing for the opt-in WeLM fixed Decode EP transport."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class _Transfer:
    owner: int
    workspace: object
    topk_ids: object
    topk_weights: object


class _WelmEpAllGatherScope:
    def __init__(self, group, expected_layers):
        if group is None or not isinstance(expected_layers, int) or expected_layers <= 0:
            raise ValueError("WeLM AllGather needs the model's EP group and layer count")
        self._group = group
        self._expected_layers = expected_layers
        self._workspaces = {}
        self._pending = {}
        self._completed = set()
        self._forward_active = False

    @contextmanager
    def forward_pass(self):
        if active_welm_ep_allgather() is not self or self._forward_active or self._pending:
            raise RuntimeError("Invalid or overlapping WeLM AllGather forward pass")
        self._completed.clear()
        self._forward_active = True
        try:
            yield
            self.assert_complete()
        finally:
            self._forward_active = False

    def _create_workspace(self, device):
        # No import of Triton or torch_npu on the disabled path.
        from sglang.srt.layers.moe.token_dispatcher.welm_ep_allgather_npu import (
            WelmEpAllGather,
        )

        return WelmEpAllGather(self._group, device)

    def dispatch(self, dispatcher, hidden_states, topk_ids, topk_weights):
        if not self._forward_active:
            raise RuntimeError("WeLM AllGather dispatch requires the fixed model forward scope")
        if (
            dispatcher.group is not self._group
            or dispatcher.router_topk != 10
            or dispatcher.num_experts != 512
            or dispatcher.num_local_experts != 128
            or dispatcher.hidden_size != 2048
            or dispatcher.use_fp8
            or dispatcher.use_nvfp4
            or dispatcher.overlap_args is not None
            or dispatcher.meta_overlap_args is not None
            or dispatcher.quant_config.get("input_global_scale") is not None
            or tuple(hidden_states.shape) != (1, 2048)
            or dispatcher._resolve_num_max_dispatch_tokens_per_rank(1) != 8
        ):
            raise RuntimeError("Incompatible fixed WeLM BF16 low-latency EP4 dispatcher")
        key = id(dispatcher)
        if key in self._pending or key in self._completed:
            raise RuntimeError("Repeated WeLM AllGather dispatch in one fixed model forward")
        workspace = self._workspaces.get(key)
        if workspace is None:
            if len(self._workspaces) == self._expected_layers:
                raise RuntimeError("Unexpected additional WeLM AllGather layer")
            workspace = self._workspaces[key] = self._create_workspace(hidden_states.device)
        packed, counts = workspace.dispatch(hidden_states, topk_ids, topk_weights)
        ticket = _Transfer(key, workspace, topk_ids, topk_weights)
        self._pending[key] = ticket
        return packed, counts, ticket

    def engine_combine(self, dispatcher, ticket, down, topk_ids, topk_weights):
        key = id(dispatcher)
        if (
            not self._forward_active
            or ticket is None
            or self._pending.get(key) is not ticket
            or ticket.owner != key
            or ticket.workspace is not self._workspaces[key]
            or ticket.topk_ids is not topk_ids
            or ticket.topk_weights is not topk_weights
        ):
            raise RuntimeError("WeLM AllGather Combine does not match its live dispatch")
        out = ticket.workspace.combine(down, topk_ids, topk_weights)
        del self._pending[key]
        self._completed.add(key)
        return out

    def assert_complete(self):
        if self._pending or len(self._completed) != self._expected_layers:
            raise RuntimeError(
                f"WeLM AllGather captured {len(self._completed)}/{self._expected_layers} "
                f"matched layers and has {len(self._pending)} unfinished dispatches"
            )


_ACTIVE: ContextVar[_WelmEpAllGatherScope | None] = ContextVar(
    "welm_fixed_decode_ep_allgather", default=None
)


def active_welm_ep_allgather():
    return _ACTIVE.get()


@contextmanager
def welm_ep_allgather_scope(enabled, *, group=None, expected_layers=48):
    if _ACTIVE.get() is not None:
        raise RuntimeError("WeLM AllGather capture scopes cannot be nested")
    if not enabled:
        yield None
        return
    scope = _WelmEpAllGatherScope(group, expected_layers)
    token = _ACTIVE.set(scope)
    try:
        yield scope
    finally:
        _ACTIVE.reset(token)
