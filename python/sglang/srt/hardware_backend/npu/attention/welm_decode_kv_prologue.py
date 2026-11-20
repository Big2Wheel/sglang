"""Fixed-row BF16 WeLM K-norm/RoPE/paged store for static Decode."""

import torch
import triton
import triton.language as tl

from sglang.srt.layers.welmv4_npu_op import _rope_npu

Q_HEADS, KV_HEADS, HEAD_DIM, ROPE_DIM, PAGE = 24, 2, 256, 64, 128


@triton.jit
def _welmv4_fixed_decode_kv_prologue_kernel(
    Q, K, V, Gamma, Positions, CosSin, Slot, KCache, VCache, NormDebug,
    eps,
    Q_HEADS: tl.constexpr, Q_BLOCK: tl.constexpr, KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr, ROPE_DIM: tl.constexpr, CACHE_ROWS: tl.constexpr,
    DEBUG_NORM: tl.constexpr,
):
    # Keep the same two independent [head, dimension] reductions as
    # mmq_style_k_rms_norm. A store/reload would convert to BF16 here.
    heads = tl.arange(0, KV_HEADS)
    dims = tl.arange(0, HEAD_DIM)
    k_off = heads[:, None] * HEAD_DIM + dims[None, :]
    raw_k = tl.load(K + k_off).to(tl.float32)
    gamma = tl.load(Gamma + dims)
    inv_rms = tl.math.rsqrt(tl.sum(raw_k * raw_k, axis=-1) / HEAD_DIM + eps)
    k_norm = raw_k * inv_rms[:, None]
    k_norm *= gamma[None, :]
    k_norm_bf16 = k_norm.to(tl.bfloat16)
    if DEBUG_NORM:
        tl.store(NormDebug + k_off, k_norm_bf16)

    slot = tl.load(Slot).to(tl.int64)
    position = tl.load(Positions).to(tl.int32)
    half: tl.constexpr = ROPE_DIM // 2
    tail: tl.constexpr = HEAD_DIM - ROPE_DIM
    lane = tl.arange(0, half)
    cos = tl.load(CosSin + position * ROPE_DIM + lane)
    sin = tl.load(CosSin + position * ROPE_DIM + half + lane)
    # Same kernel helper and same in-place packed Q storage. Even for negative
    # slot sentinels the baseline still rotates Q.
    _rope_npu(Q + tail, cos, sin, Q_HEADS, Q_BLOCK, HEAD_DIM, ROPE_DIM)
    # Every legal slot is written, including slot 0 on padded graph rows.
    # Quarantined negative slots (and out-of-range slots) are never addressed.
    if slot >= 0 and slot < CACHE_ROWS:
        start = slot * (KV_HEADS * HEAD_DIM)
        tl.store(KCache + start + k_off, k_norm_bf16, mask=dims[None, :] < tail)
        raw_v = tl.load(V + k_off)
        tl.store(VCache + start + k_off, raw_v)

        # Reload only the rotary half lanes of the raw K so that each write is
        # contiguous. The reduction and operation order stay unchanged,
        # including the BF16 rounding BEFORE using the FP32 cos/sin table.
        t1 = tail + lane
        t2 = tail + half + lane
        head_base = heads[:, None] * HEAD_DIM
        k1 = tl.load(K + head_base + t1[None, :]).to(tl.float32)
        k2 = tl.load(K + head_base + t2[None, :]).to(tl.float32)
        k1 *= inv_rms[:, None]
        k2 *= inv_rms[:, None]
        k1 *= tl.load(Gamma + t1)[None, :]
        k2 *= tl.load(Gamma + t2)[None, :]
        x1, x2 = k1.to(tl.bfloat16), k2.to(tl.bfloat16)
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        tl.store(KCache + start + head_base + t1[None, :], out1)
        tl.store(KCache + start + head_base + t2[None, :], out2)


def _overlap(lhs, rhs):
    def interval(t):
        start = t.data_ptr()
        return start, start + t.numel() * t.element_size()

    l0, l1 = interval(lhs)
    r0, r1 = interval(rhs)
    return l0 < r1 and r0 < l1


def fused_fixed_decode_kv_prologue(
    q, k, v, gamma, positions, cos_sin, slot, key_cache, value_cache, eps,
    *, norm_debug=None,
):
    """Use the real cache slots without mutating potentially mirrored raw K/V."""
    tensors = (q, k, v, gamma, positions, cos_sin, slot, key_cache, value_cache)
    if norm_debug is not None:
        tensors = (*tensors, norm_debug)
    if (
        q.device.type != "npu"
        or any(t.device != q.device or not t.is_contiguous() for t in tensors)
        or q.shape != (1, Q_HEADS, HEAD_DIM)
        or k.shape != (1, KV_HEADS, HEAD_DIM)
        or v.shape != k.shape
        or any(t.dtype != torch.bfloat16 for t in (q, k, v, key_cache, value_cache))
        or gamma.shape != (HEAD_DIM,)
        or gamma.dtype not in (torch.bfloat16, torch.float32)
        or positions.shape != (1,)
        or positions.dtype not in (torch.int32, torch.int64)
        or cos_sin.ndim != 2
        or cos_sin.shape[1] != ROPE_DIM
        or cos_sin.dtype != torch.float32
        or slot.shape != (1,)
        or slot.dtype not in (torch.int32, torch.int64)
        or key_cache.ndim != 4
        or key_cache.shape[1:] != (PAGE, KV_HEADS, HEAD_DIM)
        or key_cache.shape != value_cache.shape
        or (norm_debug is not None and (norm_debug.shape != k.shape or norm_debug.dtype != k.dtype))
    ):
        raise ValueError("Prototype requires contiguous fixed-row BF16 WeLM attention on a single NPU")
    destinations = (q, key_cache, value_cache)
    if norm_debug is not None:
        destinations = (*destinations, norm_debug)
    readonly = (k, v, gamma, positions, cos_sin, slot)
    for index, dest in enumerate(destinations):
        if any(_overlap(dest, peer) for peer in (*readonly, *destinations[:index])):
            raise ValueError("Prototype does not support overlapping read/write intervals")

    _welmv4_fixed_decode_kv_prologue_kernel[(1,)](
        q, k, v, gamma, positions, cos_sin, slot, key_cache, value_cache,
        norm_debug if norm_debug is not None else k,
        eps, Q_HEADS, triton.next_power_of_2(Q_HEADS), KV_HEADS, HEAD_DIM,
        ROPE_DIM, key_cache.shape[0] * PAGE, norm_debug is not None,
    )
    return q
