"""Host contracts for the multi-row WeLM MK Decode attention adapter."""

import ast
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, Mock, patch


ROOT = Path(__file__).resolve().parents[5]
SOURCE = ROOT / "python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py"


class Vec:
    def __init__(self, data, dtype="i32"):
        self.data = list(data)
        self.dtype = dtype
        self.device = "npu"

    @property
    def shape(self):
        return (len(self.data),)

    def __getitem__(self, index):
        return Vec(self.data[index], self.dtype)

    def __gt__(self, value):
        return Vec(x > value for x in self.data)

    def __lt__(self, value):
        return Vec(x < value for x in self.data)

    def __mul__(self, value):
        return Vec((x * value for x in self.data), self.dtype)

    def copy_(self, other):
        assert self.shape == other.shape
        self.data[:] = [int(x) for x in other.data]


def where(mask, yes, no):
    def at(value, i):
        return value.data[i] if isinstance(value, Vec) else value

    return Vec(at(yes if active else no, i) for i, active in enumerate(mask.data))


class SplitPool(NS):
    def translate_loc_from_full_to_swa(self, loc):
        assert all(x >= 0 for x in loc.data), "never index the pool with stale padding"
        return Vec((x + 32 if x >= 128 else 0 for x in loc.data), "i64")


def load_methods():
    tree = ast.parse(SOURCE.read_text())
    backend = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AscendAttnBackend"
    )
    names = {
        "_prepare_welm_decode_metadata_device",
        "_init_cuda_graph_metadata",
        "init_cuda_graph_state",
        "forward_decode_graph",
    }
    selected = ast.parse("from __future__ import annotations\n")
    selected.body.extend(
        node for node in backend.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    )
    fake_torch = NS(
        where=where,
        int32="i32",
        int64="i64",
        bfloat16="bf16",
        iinfo=lambda _: NS(max=(1 << 31) - 1),
        arange=lambda *args, dtype, device: Vec(range(*args), dtype),
        empty=lambda shape, **_: MagicMock(shape=shape),
        empty_like=lambda source, dtype: Vec([0] * source.shape[0], dtype),
        zeros=lambda count, dtype, device: Vec([0] * count, dtype),
        tensor=lambda data, dtype, device: Vec(data, dtype),
    )
    feature_on = NS(get=lambda: True)
    symbols = dict(
        __package__="_mock_welm_attention",
        torch=fake_torch,
        SWAKVPool=SplitPool,
        ForwardMetadata=lambda **fields: NS(welm_mk_full_out_cache_loc=None, **fields),
        KVWriteLoc=lambda full, swa: (full, swa),
        envs=NS(
            SGLANG_WELMV4_MK_DECODE_FLASH_KV_BOUND=feature_on,
            SGLANG_WELMV4_MK_DECODE_FLASH_FULL_UNMASKED=feature_on,
        ),
    )
    exec(compile(selected, str(SOURCE), "exec"), symbols)
    return symbols


class TestWelmMkMultiRowAttention(unittest.TestCase):
    def setUp(self):
        self.methods = load_methods()
        pool_args = dict(
            size=524288, page_size=128, use_fia=True,
            use_scatter_pa_kv_cache=True, dtype="bf16", store_dtype="bf16",
        )
        self.pool = SplitPool(
            swa_kv_pool=NS(**pool_args), full_kv_pool=NS(**pool_args),
            set_kv_buffer=Mock(), get_key_buffer=Mock(), get_value_buffer=Mock(),
        )
        self.backend = NS(
            _use_welm_mk_decode_fast_path=Mock(return_value=True),
            _prepare_welm_graph_page_tables_device=Mock(),
            _forward_welm_flash_attention=Mock(return_value="attention-output"),
            _init_dsv4_graph_buffers=Mock(), token_to_kv_pool=self.pool,
            use_mla=False, use_welm_flash_attn=True, use_fia=True,
            is_welm_v4=True, is_hybrid_swa=True, use_sliding_window_kv_pool=True,
            capture_welm_decode_metadata=True, speculative_num_draft_tokens=None,
            model_dtype="bf16", device="npu", page_size=128, max_context_len=385,
            req_to_token=NS(shape=(16, 385)), full_to_swa_index_mapping=object(),
            q_head_num_padding=None,
        )
        self.layer = NS(layer_id=17)

    def batch(self, lengths, slots):
        return NS(
            batch_size=len(lengths), seq_lens=Vec(lengths), out_cache_loc=Vec(slots),
            req_pool_indices=Vec(range(len(lengths))), spec_info=None,
            forward_mode=NS(is_decode_or_idle=lambda: True), is_extend_in_batch=False,
        )

    def set_metadata(self, batch_size, masked):
        self.backend.forward_metadata = NS(
            block_tables=object(), block_tables_swa=object(),
            welm_mk_full_out_cache_loc=Vec([0] * batch_size) if masked else None,
            swa_out_cache_loc=Vec([0] * batch_size),
            welm_flash_seqused_q=Vec([0] * batch_size),
            welm_flash_seqused_kv=Vec([0] * batch_size),
        )
        return self.backend.forward_metadata

    def test_multirow_buffers_and_int32_bounds(self):
        init = self.methods["init_cuda_graph_state"]
        init(self.backend, 1, 1)
        self.assertNotIn("welm_mk_decode_dummy_slots", self.backend.graph_metadata)
        self.assertEqual(self.backend.swa_out_cache_loc_buf.dtype, "i32")
        init(self.backend, 4, 4)
        self.assertEqual(self.backend.swa_out_cache_loc_buf.shape, (4,))
        self.assertEqual(
            self.backend.graph_metadata["welm_mk_decode_dummy_slots"].data,
            [1, 2, 3, 4],
        )
        for bs, tokens in ((4, 8), (128, 128)):
            with self.subTest(bs=bs), self.assertRaises(RuntimeError):
                init(self.backend, bs, tokens)
        self.pool.full_kv_pool.size = (1 << 31) - 128
        with self.assertRaisesRegex(RuntimeError, "both pools"):
            init(self.backend, 4, 4)

    def test_bs1_and_multirow_in_same_graph_state_get_different_slot_paths(self):
        self.methods["init_cuda_graph_state"](self.backend, 4, 4)
        mode = NS(
            is_target_verify=lambda: False,
            is_draft_extend_v2=lambda: False,
            is_dllm_extend=lambda: False,
        )
        init = self.methods["_init_cuda_graph_metadata"]
        one = init(
            self.backend, 1, mode, Vec([0]), Vec([0]),
            welm_decode_metadata_in_graph=True,
        )
        four = init(
            self.backend, 4, mode, Vec([0] * 4), Vec([0] * 4),
            welm_decode_metadata_in_graph=True,
        )
        self.assertTrue(one.welm_decode_metadata_in_graph)
        self.assertTrue(four.welm_decode_metadata_in_graph)
        self.assertIsNone(one.welm_mk_full_out_cache_loc)
        self.assertEqual(four.welm_mk_full_out_cache_loc.shape, (4,))
        self.assertEqual(four.welm_mk_full_out_cache_loc.dtype, "i32")
        with self.assertRaisesRegex(RuntimeError, "one KV slot per row"):
            init(
                self.backend, 4, mode, Vec([0] * 4), Vec([0] * 8),
                welm_decode_metadata_in_graph=True,
            )

    def test_device_metadata_reuses_buffers_and_keeps_padding_off_live_slots(self):
        self.methods["init_cuda_graph_state"](self.backend, 4, 4)
        metadata = self.set_metadata(4, masked=True)
        full, swa = metadata.welm_mk_full_out_cache_loc, metadata.swa_out_cache_loc
        for lengths, slots, want_full, want_swa in (
            ([9, 0, 128, 0], [128, -600, 4096, -600], [128, 2, 4096, 4], [160, 2, 4128, 4]),
            ([0, 17, 0, 1], [-600, 256, -600, 512], [1, 256, 3, 512], [1, 288, 3, 544]),
            ([0, 0, 0, 0], [-600] * 4, [1, 2, 3, 4], [1, 2, 3, 4]),
        ):
            batch = self.batch(lengths, slots)
            self.methods["_prepare_welm_decode_metadata_device"](self.backend, batch)
            self.assertIs(metadata.welm_mk_full_out_cache_loc, full)
            self.assertIs(metadata.swa_out_cache_loc, swa)
            self.assertEqual(full.data, want_full)
            self.assertEqual(swa.data, want_swa)
            self.assertEqual(batch.out_cache_loc.data, slots)
            valid = [x > 0 for x in lengths]
            self.assertEqual(metadata.welm_flash_seqused_q.data, valid)
            self.assertEqual(metadata.welm_flash_seqused_kv.data, lengths)
            args = self.backend._prepare_welm_graph_page_tables_device.call_args.args
            self.assertIs(args[1], batch.req_pool_indices)
            self.assertEqual(args[3].data, valid)

    def test_bs1_still_calls_fused_page_prologue(self):
        mock = Mock()
        self.set_metadata(1, masked=False)
        with patch.dict(
            sys.modules,
            {"_mock_welm_attention.welm_decode_page_prologue": NS(prepare_page_prologue=mock)},
        ):
            self.methods["_prepare_welm_decode_metadata_device"](
                self.backend, self.batch([127], [128])
            )
        mock.assert_called_once()
        self.backend._prepare_welm_graph_page_tables_device.assert_not_called()

    def test_stock_scatter_and_flash_switch_by_physical_bucket(self):
        call = self.methods["forward_decode_graph"]
        self.set_metadata(1, masked=False)
        one = self.batch([1], [128])
        self.assertEqual(
            call(self.backend, None, None, None, self.layer, one), "attention-output"
        )
        self.assertIs(self.pool.set_kv_buffer.call_args.args[1][0], one.out_cache_loc)
        flash = self.backend._forward_welm_flash_attention
        self.assertTrue(flash.call_args.kwargs["use_fixed_decode_kv_bound"])
        self.assertTrue(flash.call_args.kwargs["use_unmasked_full_decode"])
        metadata = self.set_metadata(4, masked=True)
        several = self.batch([9, 0, 0, 0], [128, 0, 0, 0])
        call(self.backend, None, None, None, self.layer, several)
        args, kwargs = self.pool.set_kv_buffer.call_args
        self.assertEqual(
            args[1], (metadata.welm_mk_full_out_cache_loc, metadata.swa_out_cache_loc)
        )
        self.assertTrue(kwargs["use_scatter_pa_kv_cache"])
        self.assertFalse(flash.call_args.kwargs["use_fixed_decode_kv_bound"])
        self.assertFalse(flash.call_args.kwargs["use_unmasked_full_decode"])


if __name__ == "__main__":
    unittest.main()
