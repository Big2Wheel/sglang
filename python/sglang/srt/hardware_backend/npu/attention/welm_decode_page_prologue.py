"""Fixed-size device metadata prologue for WeLMV4 MegaKernel Decode."""

import triton
import triton.language as tl

from .welm_verify_metadata import _floor_page


@triton.jit
def _welm_decode_page_prologue(
    Req, ReqID, SeqLen, FullWrite, SWAMap, OutFull, OutSWA, OutWrite, QUsed, KVUsed,
    Width: tl.constexpr, ReqStride: tl.constexpr, Context: tl.constexpr,
    MapSize: tl.constexpr, Page: tl.constexpr, Block: tl.constexpr,
):
    part = tl.program_id(0)
    col = part * Block + tl.arange(0, Block)
    token = col * Page
    length = tl.load(SeqLen)
    active = length > 0
    valid = (col < Width) & (token < Context) & (token < length) & active
    req = tl.load(ReqID).to(tl.int64)
    full = tl.load(Req + req * ReqStride + token, valid, 0)
    swa_loc = tl.load(SWAMap + tl.where(full < 0, full + MapSize, full), valid, 0)
    tl.store(OutFull + col, _floor_page(full, Page), col < Width)
    tl.store(OutSWA + col, _floor_page(swa_loc, Page), col < Width)
    first = part == 0
    full_write_loc = tl.load(FullWrite, mask=first & active, other=0)
    idx = tl.where(full_write_loc < 0, full_write_loc + MapSize, full_write_loc)
    slot = tl.load(SWAMap + idx, mask=first & active, other=0)
    tl.store(OutWrite, slot, mask=first)
    tl.store(QUsed, active.to(tl.int32), mask=first)
    tl.store(KVUsed, tl.where(active, length, 0), mask=first)


def prepare_page_prologue(req, ids, seq, slot, mapping, full, swa, out_slot, q_used, kv_used, page):
    if (
        full.ndim != 2 or full.shape[0] != 1 or full.shape != swa.shape
        or req.ndim != 2 or req.stride(1) != 1
        or any(x.numel() != 1 for x in (ids, seq, slot, out_slot, q_used, kv_used))
        or any(not x.is_contiguous() for x in (mapping, full, swa, out_slot, q_used, kv_used))
    ):
        raise RuntimeError("Fused WeLM Decode page prologue expects the fixed single-row layout")
    width = full.shape[1]
    block = 128
    _welm_decode_page_prologue[(triton.cdiv(width, block),)](
        req, ids, seq, slot, mapping, full, swa, out_slot, q_used, kv_used,
        width, req.stride(0), req.shape[1], mapping.numel(), page, block,
    )
