"""Fixed WeLM EP4 BF16 transport: exact routing, GMM ABI and return arithmetic."""

import torch
import torch.distributed as dist
import torch_npu
import triton
import triton.language as tl
from torch.utils._python_dispatch import _disable_current_modes


EP, TOPK, LOCAL_EXPERTS, H, CAPACITY = 4, 10, 128, 2048, 320
ROUTES = EP * TOPK
H_I32 = H // 2  # Reinterpret, never convert, the BF16 payload bits.
MESSAGE_I32 = H_I32 + 32


@triton.jit
def _pack_ints(hidden_words, ids, wire, HWORDS: tl.constexpr, COUNT: tl.constexpr):
    i = tl.arange(0, 256) + tl.program_id(0) * 256
    x = tl.load(hidden_words + i, mask=i < HWORDS, other=0)
    route = tl.load(ids + i - HWORDS, mask=(i >= HWORDS) & (i < HWORDS + COUNT), other=0)
    out = tl.where(i < HWORDS, x, route)
    tl.store(wire + i, out, mask=i < HWORDS + 32)


@triton.jit
def _make_plan(all_wires, flat_to_packed, counts,
               EXPERT_FIRST: tl.constexpr, EXPERTS: tl.constexpr,
               R: tl.constexpr, K: tl.constexpr, HWORDS: tl.constexpr,
               STRIDE: tl.constexpr, MAP_BLOCKS: tl.constexpr, BLOCK: tl.constexpr):
    j = tl.arange(0, 64)
    peer_id = tl.load(all_wires + (j // K) * STRIDE + HWORDS + j % K,
                      mask=j < R, other=-1)
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    if pid < MAP_BLOCKS:
        route = pid * BLOCK + lane
        this_id = tl.load(all_wires + (route // K) * STRIDE + HWORDS + route % K,
                          mask=route < R, other=-1)
        precedes = ((peer_id[None, :] < this_id[:, None]) |
                    ((peer_id[None, :] == this_id[:, None]) & (j[None, :] < route[:, None])))
        peer_local = ((j < R) & (peer_id >= EXPERT_FIRST) &
                      (peer_id < EXPERT_FIRST + EXPERTS))
        order = tl.sum((peer_local[None, :] & precedes).to(tl.int32), axis=1)
        local = (this_id >= EXPERT_FIRST) & (this_id < EXPERT_FIRST + EXPERTS)
        tl.store(flat_to_packed + route, tl.where(local, order, -1), mask=route < R)
    else:
        index = (pid - MAP_BLOCKS) * BLOCK + lane
        matches = ((index[:, None] + EXPERT_FIRST == peer_id[None, :]) & (j[None, :] < R))
        sizes = tl.sum(matches.to(tl.int32), axis=1)
        tl.store(counts + index, sizes.to(tl.int64), mask=index < EXPERTS)


@triton.jit
def _unpack(all_wires, flat_to_packed, output_words, K: tl.constexpr,
            HWORDS: tl.constexpr, STRIDE: tl.constexpr):
    route = tl.program_id(0)
    row = tl.load(flat_to_packed + route)
    if row >= 0:
        i = tl.arange(0, HWORDS)
        x = tl.load(all_wires + route // K * STRIDE + i)
        tl.store(output_words + row * HWORDS + i, x)


@triton.jit
def _pack_owned_rows(down, mapping, send, H: tl.constexpr, N: tl.constexpr,
                     META_WIDTH: tl.constexpr, INVERSE: tl.constexpr):
    flat = tl.program_id(0)
    if INVERSE:
        packed = tl.load(mapping + flat)
        found = (packed >= 0) & (packed < N)
    else:
        r = tl.arange(0, META_WIDTH)
        f = tl.load(mapping + r, r < N, other=-1)
        match = (r < N) & (f == flat)
        packed = tl.max(tl.where(match, r, -1), axis=0)
        found = tl.sum(match.to(tl.int32), axis=0) == 1
    h = tl.arange(0, H)
    if found:
        y = tl.load(down + packed * H + h)
    else:
        y = tl.full((H,), 0, tl.bfloat16)
    tl.store(send + flat * H + h, y)


@triton.jit
def _combine_one_destination(gathered, expert_ids, weights, out,
                             RANK: tl.constexpr, H: tl.constexpr, TOPK: tl.constexpr,
                             WORLD: tl.constexpr, LOCAL_EXPERTS_: tl.constexpr,
                             BLOCK: tl.constexpr):
    h = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.full((BLOCK,), 0.0, tl.float32)
    # Preserve vendor slot order, separate FP32 multiply/add, and final RNE.
    for k in tl.static_range(TOPK):
        selected = tl.load(expert_ids + k)
        valid = (selected >= 0) & (selected < WORLD * LOCAL_EXPERTS_)
        owner = tl.where(valid, selected // LOCAL_EXPERTS_, 0)
        row = ((owner * WORLD + RANK) * TOPK + k) * H
        x = tl.load(gathered + row + h, mask=valid, other=0).to(tl.float32)
        w = tl.load(weights + k)
        p = x * w
        new = acc + p
        acc = tl.where(valid, new, acc)
    tl.store(out + h, acc.to(tl.bfloat16, fp_downcast_rounding="rtne"))


def _raw_nd(tensor, shape, dtype, device):
    # MK canonicalizes operator inputs, not the physical allocation. As in
    # the KV guard, probe only the allocation format with modes disabled.
    with _disable_current_modes():
        fmt = int(torch_npu.get_npu_format(tensor))
    if (tuple(tensor.shape) != tuple(shape) or tensor.dtype != dtype
            or tensor.device != device or not tensor.is_contiguous() or fmt != 2):
        raise ValueError(
            "WeLM AllGather raw access needs physical ND with fixed ABI: "
            f"expected {shape}/{dtype}/{device}; got {tuple(tensor.shape)}/"
            f"{tensor.dtype}/{tensor.device} format={fmt}"
        )
    storage = tensor.untyped_storage()
    offset = tensor.data_ptr() - storage.data_ptr()
    if offset < 0 or offset + tensor.numel() * tensor.element_size() > storage.nbytes():
        raise ValueError("WeLM AllGather tensor span is outside owner storage")


def _dtype_alias(original, alias, shape, dtype):
    if (tuple(alias.shape) != tuple(shape) or alias.dtype != dtype or not alias.is_contiguous()
            or alias.device != original.device or alias.data_ptr() != original.data_ptr()
            or alias.untyped_storage().data_ptr() != original.untyped_storage().data_ptr()
            or alias.numel() * alias.element_size() != original.numel() * original.element_size()
            or alias.data_ptr() % alias.element_size()):
        raise ValueError("WeLM AllGather view must alias exactly the same aligned bytes")
    return alias


def _allocation(tensor):
    storage = tensor.untyped_storage()
    return storage.data_ptr(), storage.data_ptr() + storage.nbytes()


def _verify_no_alias(writers, readers):
    for idx, (begin, end) in enumerate(writers):
        if any(begin < other_end and other_begin < end
               for other_begin, other_end in (*writers[:idx], *readers)):
            raise ValueError("WeLM AllGather transport tensors unexpectedly share storage")


class WelmEpAllGather:
    """Own fixed non-overlapping addresses for one recorded MoE layer."""

    def __init__(self, group, device):
        self.group, self.device = group, torch.device(device)
        self.rank = dist.get_rank(group)
        if (self.device.type != "npu" or self.device.index is None
                or self.device.index != torch.npu.current_device()
                or "950" not in torch.npu.get_device_name(self.device.index)
                or str(dist.get_backend(group)).lower() != "hccl"
                or dist.get_world_size(group) != EP or not 0 <= self.rank < EP
                or tuple(dist.get_process_group_ranks(group)) != tuple(range(EP))):
            raise ValueError("WeLM AllGather requires the original ordered single-node A5 EP4 group")

        def alloc(shape, dtype):
            value = torch.empty(shape, dtype=dtype, device=self.device)
            _raw_nd(value, shape, dtype, self.device)
            return value

        self.send = alloc((MESSAGE_I32,), torch.int32)
        self.gathered = alloc((EP * MESSAGE_I32,), torch.int32)
        self.flat_to_packed = alloc((ROUTES,), torch.int32)
        self.counts = alloc((LOCAL_EXPERTS,), torch.int64)
        self.packed = alloc((CAPACITY, H), torch.bfloat16)
        self.packed_words = _dtype_alias(self.packed, self.packed.view(torch.int32),
                                         (CAPACITY, H_I32), torch.int32)

        # Do not directly allocate rank-four storage: its physical format
        # can differ from ND even when the logical strides look contiguous.
        self.return_send_owner = alloc((EP * TOPK * H,), torch.bfloat16)
        self.return_recv_owner = alloc((EP * EP * TOPK * H,), torch.bfloat16)
        self.return_send = self.return_send_owner.view(EP, TOPK, H)
        self.return_recv = self.return_recv_owner.view(EP, EP, TOPK, H)
        self.return_recv_cat = self.return_recv_owner.view(EP * EP, TOPK, H)
        self.return_out = alloc((1, H), torch.bfloat16)
        for owner, alias in ((self.return_send_owner, self.return_send),
                             (self.return_recv_owner, self.return_recv),
                             (self.return_recv_owner, self.return_recv_cat)):
            _raw_nd(alias, alias.shape, torch.bfloat16, self.device)
            _dtype_alias(owner, alias, alias.shape, torch.bfloat16)
            if owner.storage_offset() or alias.storage_offset():
                raise ValueError("WeLM AllGather view must start at its owner base")
        self._writers = tuple(map(_allocation, (
            self.send, self.gathered, self.flat_to_packed, self.counts, self.packed,
            self.return_send_owner, self.return_recv_owner, self.return_out,
        )))
        _verify_no_alias(self._writers, ())

    def _drain(self):
        torch_npu._C._npu_getCurrentRawStream(self.device.index)

    def _allgather(self, out, send):
        # Keep Torch's producer/join and raw Triton device work ordered.
        # MK may lower the collective to its device-only packet during replay.
        self._drain()
        work = dist.all_gather_into_tensor(out, send, group=self.group, async_op=True)
        if work is None:
            raise RuntimeError("WeLM AllGather did not return an ordered HCCL Work")
        work.wait()
        self._drain()

    def _topk(self, ids, weights):
        _raw_nd(ids, (1, TOPK), torch.int32, self.device)
        _raw_nd(weights, (1, TOPK), torch.float32, self.device)

    def dispatch(self, hidden, ids, weights):
        _raw_nd(hidden, (1, H), torch.bfloat16, self.device)
        self._topk(ids, weights)
        _verify_no_alias(self._writers, tuple(map(_allocation, (hidden, ids, weights))))
        hidden_words = _dtype_alias(hidden, hidden.view(torch.int32), (1, H_I32), torch.int32)
        self._drain()
        _pack_ints[(triton.cdiv(MESSAGE_I32, 256),)](hidden_words, ids, self.send, H_I32, TOPK)
        self._allgather(self.gathered, self.send)
        block = 8
        map_blocks = triton.cdiv(ROUTES, block)
        _make_plan[(map_blocks + triton.cdiv(LOCAL_EXPERTS, block),)](
            self.gathered, self.flat_to_packed, self.counts,
            self.rank * LOCAL_EXPERTS, LOCAL_EXPERTS, ROUTES, TOPK,
            H_I32, MESSAGE_I32, map_blocks, block,
        )
        _unpack[(ROUTES,)](self.gathered, self.flat_to_packed, self.packed_words,
                           TOPK, H_I32, MESSAGE_I32)
        return self.packed, self.counts

    def combine(self, down, ids, weights):
        _raw_nd(down, (CAPACITY, H), torch.bfloat16, self.device)
        self._topk(ids, weights)
        _verify_no_alias(self._writers, tuple(map(_allocation, (down, ids, weights))))
        self._drain()
        _pack_owned_rows[(ROUTES,)](down, self.flat_to_packed, self.return_send, H, ROUTES, 64, True)
        self._allgather(self.return_recv_cat, self.return_send)
        block = 64
        _combine_one_destination[(H // block,)](
            self.return_recv, ids, weights, self.return_out, self.rank, H, TOPK, EP,
            LOCAL_EXPERTS, block, enable_fp_fusion=False,
        )
        return self.return_out
