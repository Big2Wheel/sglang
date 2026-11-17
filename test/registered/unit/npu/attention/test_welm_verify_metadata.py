from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from sglang.srt.hardware_backend.npu.attention.welm_verify_metadata import (
    prepare_verify_metadata,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=10, suite="stage-a-unit-test-npu")


@pytest.mark.parametrize(
    "request_ids,lengths", [([2], [510]), ([1, 0], [384, 0]), ([1, 2], [4100, 256])]
)
def test_verify_metadata_matches_eager(request_ids, lengths):
    device, dtype = "npu:0", torch.int32
    page, width, draft = 128, 33, 4
    bs = len(request_ids)
    table = torch.zeros((3, width * page), dtype=dtype, device=device)
    table[:, ::page] = (
        torch.arange(3 * width, device=device, dtype=dtype).reshape(3, width) * page
    )
    table[1, 2 * page] = -1
    mapping = torch.arange(4 * width * page, dtype=torch.int64, device=device)
    mapping = (2 * mapping + page) % mapping.numel()
    mapping[-1] = -1
    ids = torch.tensor(request_ids, device=device, dtype=torch.int64)
    seq = torch.tensor(lengths, device=device, dtype=dtype)
    pages = (max(lengths) + draft + page - 1) // page
    metadata = SimpleNamespace(
        block_tables=torch.full((bs, width), -2, dtype=dtype, device=device),
        block_tables_swa=torch.full((bs, width), -2, dtype=dtype, device=device),
        seq_lens=torch.empty(bs, device=device, dtype=dtype),
        welm_flash_seqused_q=torch.empty(bs, device=device, dtype=dtype),
        welm_flash_seqused_kv=torch.empty(bs, device=device, dtype=dtype),
    )
    prepare_verify_metadata(metadata, table, ids, seq, mapping, bs, pages, page, draft)
    expected = torch.zeros_like(metadata.block_tables)
    locations = table[ids, : pages * page : page]
    expected[:, :pages] = locations // page
    assert torch.equal(metadata.block_tables, expected)
    expected[:, :pages] = (mapping[locations.to(torch.int64)] // page).to(dtype)
    assert torch.equal(metadata.block_tables_swa, expected)
    assert torch.equal(metadata.seq_lens, seq + draft)
    assert torch.equal(metadata.welm_flash_seqused_q, (seq > 0).to(dtype) * draft)
    assert torch.equal(
        metadata.welm_flash_seqused_kv, torch.where(seq > 0, seq + draft, 0)
    )
