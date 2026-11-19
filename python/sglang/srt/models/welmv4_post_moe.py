"""Merge and mask one BF16 row in the fixed WeLMV4 Decode tail."""

import torch
import triton
import triton.language as tl

@triton.jit
def _welm_postmoe_add_mask(Expert, Shared, Invalid, Hidden: tl.constexpr, Block: tl.constexpr):
    cols = tl.program_id(0) * Block + tl.arange(0, Block)
    invalid = tl.load(Invalid)
    expert = tl.load(Expert + cols, cols < Hidden, 0).to(tl.float32)
    shared = tl.load(Shared + cols, cols < Hidden, 0).to(tl.float32)
    # The store rounds one BF16 add, just like add_. The exact zero in
    # invalid rows has the masked_fill_ semantics even for Inf or NaN.
    result = tl.where(invalid, 0.0, expert + shared)
    tl.store(Expert + cols, result, cols < Hidden)


def merge_and_mask(expert, shared, invalid):
    if (
        expert.dim() != 2 or expert.shape[0] != 1
        or expert.shape != shared.shape or invalid.shape != (1,)
        or expert.dtype != shared.dtype or expert.dtype != torch.bfloat16
        or invalid.dtype != torch.bool
        or expert.device != shared.device or expert.device != invalid.device
        or not (expert.is_contiguous() and shared.is_contiguous() and invalid.is_contiguous())
    ):
        raise RuntimeError("WeLMV4 Post-MoE fusion requires one contiguous BF16 row")
    hidden = expert.shape[-1]
    block = 512
    _welm_postmoe_add_mask[(triton.cdiv(hidden, block),)](
        expert, shared, invalid, hidden, block
    )
