"""Fixed-address FlashAttn metadata for WeLM target VERIFY capture buckets."""

import triton
import triton.language as tl


@triton.jit
def _floor_page(loc, page_tokens: tl.constexpr):
    return tl.where(
        loc < 0, -((-loc + page_tokens - 1) // page_tokens), loc // page_tokens
    )


@triton.jit
def _prepare_verify_metadata(
    req_table,
    req_ids,
    seq_lens,
    swa_mapping,
    full_pages,
    swa_pages,
    attention_lens,
    q_used,
    kv_used,
    live_pages,
    req_stride: tl.constexpr,
    full_stride: tl.constexpr,
    swa_stride: tl.constexpr,
    width: tl.constexpr,
    page_tokens: tl.constexpr,
    draft_width: tl.constexpr,
    tile: tl.constexpr,
    mapping_size: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, tile)
    req = tl.load(req_ids + row).to(tl.int64)
    length = tl.load(seq_lens + row)
    full_loc = tl.load(
        req_table + req * req_stride + cols * page_tokens,
        mask=(cols < width) & (cols < live_pages),
        other=0,
    )
    mapping_index = tl.where(full_loc < 0, full_loc + mapping_size, full_loc)
    swa_loc = tl.load(swa_mapping + mapping_index, mask=cols < live_pages, other=0)
    tl.store(
        full_pages + row * full_stride + cols,
        _floor_page(full_loc, page_tokens),
        mask=cols < width,
    )
    tl.store(
        swa_pages + row * swa_stride + cols,
        _floor_page(swa_loc, page_tokens),
        mask=cols < width,
    )
    tl.store(attention_lens + row, length + draft_width)
    tl.store(q_used + row, tl.where(length > 0, draft_width, 0))
    tl.store(kv_used + row, tl.where(length > 0, length + draft_width, 0))


def prepare_verify_metadata(
    metadata,
    req_table,
    req_ids,
    seq_lens,
    swa_mapping,
    batch_size,
    live_pages,
    page_tokens,
    draft_width,
):
    full, swa = metadata.block_tables, metadata.block_tables_swa
    width = full.shape[1]
    assert full.shape == swa.shape and full.shape[0] == batch_size
    assert 0 <= live_pages <= width
    assert req_table.stride(1) == full.stride(1) == swa.stride(1) == 1
    _prepare_verify_metadata[(batch_size,)](
        req_table,
        req_ids,
        seq_lens,
        swa_mapping,
        full,
        swa,
        metadata.seq_lens,
        metadata.welm_flash_seqused_q,
        metadata.welm_flash_seqused_kv,
        live_pages,
        req_table.stride(0),
        full.stride(0),
        swa.stride(0),
        width,
        page_tokens,
        draft_width,
        triton.next_power_of_2(width),
        swa_mapping.numel(),
    )
