"""Validate WeLMV4's fixed Decode Post-MoE add and mask on NPU."""

import pytest
import torch


@pytest.mark.parametrize("rows,batched", [(1, False), (2, True), (4, True), (8, True)])
@pytest.mark.parametrize("hidden", [2048, 2304])
def test_post_moe_add_mask_matches_original(rows, batched, hidden):
    pytest.importorskip("torch_npu")
    from sglang.srt.models.welmv4_post_moe import merge_and_mask

    if not torch.npu.is_available():
        pytest.skip("requires NPU")

    # An aligned but partial final tile also exercises the per-row bounds.
    gen = torch.Generator().manual_seed(389)
    expert = torch.randn((rows, hidden), generator=gen).to(torch.bfloat16)
    shared = torch.randn((rows, hidden), generator=gen).to(torch.bfloat16).npu()
    invalid = torch.empty(rows, dtype=torch.bool, device="npu")
    for invalid_cpu in (
        torch.zeros(rows, dtype=torch.bool),
        torch.ones(rows, dtype=torch.bool),
        torch.arange(rows) % 2 == 1,
    ):
        payload = expert.clone()
        payload[invalid_cpu, 0] = float("inf")
        payload[invalid_cpu, 1] = float("nan")
        actual = payload.npu()
        invalid.copy_(invalid_cpu)
        expected = actual.clone().add_(shared).masked_fill_(invalid[:, None], 0)
        merge_and_mask(actual, shared, invalid, batched=batched)
        assert torch.equal(actual.cpu().view(torch.int16), expected.cpu().view(torch.int16))
