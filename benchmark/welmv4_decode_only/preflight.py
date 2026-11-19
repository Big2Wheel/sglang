"""Check the two fused operators against the existing NPU PyTorch path."""

import torch
import torch_npu

from sglang.srt.hardware_backend.npu.attention.welm_decode_page_prologue import (
    prepare_page_prologue,
)
from sglang.srt.models.welmv4_post_moe import merge_and_mask


def check_pages():
    page, context, width = 128, 128 * 129 - 5, 131
    req = torch.randint(0, 4096, (3, context), dtype=torch.int64).npu()
    req[1, page] = -1
    mapping = torch.randint(0, 4096, (4097,), dtype=torch.int64).npu()
    mapping[-1] = -1
    opts = dict(dtype=torch.int32, device="npu")
    ids = torch.tensor([1], **opts)
    starts = torch.arange(width, dtype=torch.int64, device="npu")[None, :] * page
    for length in (-1, 0, 1, 127, 128, 129, 513, context):
        for slot in (0, -1, 4095):
            seq, write = torch.tensor([length], **opts), torch.tensor([slot], **opts)
            full, swa = torch.empty(1, width, **opts), torch.empty(1, width, **opts)
            out_slot, q, kv = (torch.empty(1, **opts) for _ in range(3))
            prepare_page_prologue(
                req, ids, seq, write, mapping, full, swa, out_slot, q, kv, page
            )
            valid = (starts < context) & (starts < length) & (length > 0)
            loc = torch.where(valid, req[ids[:, None], torch.where(valid, starts, 0)], 0)
            ref = (
                loc // page,
                torch.where(valid, mapping[loc], 0) // page,
                mapping[write] if length > 0 else torch.zeros_like(out_slot),
                (seq > 0).int(),
                seq.clamp_min(0),
            )
            for actual, expected in zip((full, swa, out_slot, q, kv), ref):
                torch.testing.assert_close(actual, expected.int(), rtol=0, atol=0)


def check_post_moe():
    for width in (512, 1536, 2048, 2560, 4096):
        for masked in (False, True):
            expert = torch.randn(1, width).bfloat16().npu()
            shared = torch.randn(1, width).bfloat16().npu()
            invalid = torch.tensor([masked], device="npu")
            if masked:
                expert[0, :3] = torch.tensor(
                    [float("nan"), float("inf"), -float("inf")], device="npu"
                )
            expected = expert.clone().add_(shared).masked_fill_(invalid[:, None], 0)
            merge_and_mask(expert, shared, invalid)
            assert torch.equal(expert.cpu().view(torch.int16), expected.cpu().view(torch.int16))


if __name__ == "__main__":
    torch_npu.npu.set_device(0)  # Logical device inside the npu-run lease.
    torch.manual_seed(914)
    check_pages()
    check_post_moe()
    print("WeLMV4 Decode preflight passed")
