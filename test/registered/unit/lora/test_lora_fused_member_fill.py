"""CPU-only regression tests: an adapter may target any subset of a fused q/k/v or MLA q_a/kv_a
projection; the absent members are served as zero adapters of the base layer's widths."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang.srt.lora.lora import LoRAAdapter
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_RANK = 2
_HIDDEN = 8
_STEM = "base_model.model.model.layers.0.self_attn."
# q: 4 heads x 2, k/v: 2 heads x 2 -> fused qkv output 16
_CONFIG = SimpleNamespace(
    hidden_size=_HIDDEN,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=2,
    q_lora_rank=6,
    kv_lora_rank=4,
    qk_rope_head_dim=2,
)


def _adapter() -> LoRAAdapter:
    adapter = LoRAAdapter.__new__(LoRAAdapter)
    adapter.base_hf_config = _CONFIG
    object.__setattr__(adapter, "base_model", None)
    return adapter


def _weights(**rows: int) -> dict[str, torch.Tensor]:
    weights = {}
    for member, member_rows in rows.items():
        weights[f"{_STEM}{member}.lora_A.weight"] = torch.ones(_RANK, _HIDDEN)
        weights[f"{_STEM}{member}.lora_B.weight"] = torch.ones(member_rows, _RANK)
    return weights


def _normalized(weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    adapter = _adapter()
    adapter._fill_absent_fused_members(0, weights)
    adapter.normalize_qkv_proj(list(weights), weights)
    adapter.normalize_fused_qkv_a_proj(list(weights), weights)
    return weights


def test_a_q_only_adapter_stacks_into_the_full_qkv_buffer():
    """A q-only adapter raised KeyError on the missing v_proj."""
    weights = _normalized(_weights(q_proj=8))

    qkv_b = weights[f"{_STEM}qkv_proj.lora_B.weight"]
    assert weights[f"{_STEM}qkv_proj.lora_A.weight"].shape == (3 * _RANK, _HIDDEN)
    assert qkv_b.shape == (16, _RANK)
    assert qkv_b[:8].eq(1).all() and qkv_b[8:].eq(0).all()


def test_an_adapter_without_q_is_stacked_not_dropped():
    """k/v adapters with no q_proj were left unstacked and never reached the qkv buffer."""
    weights = _normalized(_weights(k_proj=4, v_proj=4))

    qkv_b = weights[f"{_STEM}qkv_proj.lora_B.weight"]
    assert qkv_b.shape == (16, _RANK)
    assert qkv_b[:8].eq(0).all() and qkv_b[8:].eq(1).all()
    assert (
        f"{_STEM}k_proj.lora_B.weight" not in weights
        and f"{_STEM}v_proj.lora_B.weight" not in weights
    )


def test_an_already_fused_qkv_adapter_is_not_mistaken_for_a_partial_one():
    """``qkv_proj`` ends in ``v_proj``; a suffix match would zero-fill a phantom q and k."""
    weights = {
        f"{_STEM}qkv_proj.lora_A.weight": torch.ones(_RANK, _HIDDEN),
        f"{_STEM}qkv_proj.lora_B.weight": torch.ones(16, _RANK),
    }
    _adapter()._fill_absent_fused_members(0, weights)

    assert sorted(weights) == [
        f"{_STEM}qkv_proj.lora_A.weight",
        f"{_STEM}qkv_proj.lora_B.weight",
    ]


def test_a_q_a_only_mla_adapter_sizes_kv_a_from_the_fused_layer():
    """The missing kv_a_proj_with_mqa was copied from q_a's shape, not kv_lora_rank + rope."""
    weights = _normalized(_weights(q_a_proj=6))

    fused_b = weights[f"{_STEM}fused_qkv_a_proj_with_mqa.lora_B.weight"]
    assert fused_b.shape == (6 + 4 + 2, _RANK)
    assert fused_b[:6].eq(1).all() and fused_b[6:].eq(0).all()
