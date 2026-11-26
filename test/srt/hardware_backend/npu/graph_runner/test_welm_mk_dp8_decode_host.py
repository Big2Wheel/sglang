"""Exercise the DP8 stock-capture contract without importing Torch or CANN."""

import ast
import bisect
import json
import logging
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from runpy import run_path
from types import SimpleNamespace
from typing import Optional, Sequence
from unittest.mock import patch

SRT = Path(__file__).resolve().parents[5] / "python/sglang/srt"
BACKEND = SRT / "hardware_backend/npu/graph_runner/welm_mk_decode_backend.py"
ROUTER = run_path(str(SRT / "layers/welm_router_fusion.py"))
POSTMOE = run_path(str(SRT / "layers/welmv4_mk_triton.py"))


class NativeBackendStub:
    def __init__(self, runner):
        self._enable_torch_compile = False


def stock_backend(flags):
    tree = ast.parse(BACKEND.read_bytes())
    tree.body = [
        node
        for node in tree.body
        if not isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    namespace = dict(
        NPUCudaGraphBackend=NativeBackendStub,
        contextmanager=contextmanager,
        envs=SimpleNamespace(
            **{
                name: SimpleNamespace(get=lambda v=value: v)
                for name, value in flags.items()
            }
        ),
        json=json,
        logging=logging,
        os=os,
        tempfile=tempfile,
        target_native_capture=lambda: lambda f: f,
        torch=SimpleNamespace(no_grad=lambda: lambda f: f),
    )
    exec(compile(tree, str(BACKEND), "exec"), namespace)
    return namespace["WelmMKDecodeBackend"]


def load_method(name, cls_name, filename, **namespace):
    path = SRT / "model_executor/runner" / filename
    tree = ast.parse(path.read_bytes())
    cls = next(node for node in tree.body if getattr(node, "name", "") == cls_name)
    method = next(node for node in cls.body if getattr(node, "name", "") == name)
    method.decorator_list = []
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class TestWelmDP8StockDecode(unittest.TestCase):
    def setUp(self):
        self.flags = dict.fromkeys(
            (
                "SGLANG_WELMV4_MK_EP_ALLGATHER",
                "SGLANG_WELMV4_MK_DECODE_TOPK_MODE3",
                "SGLANG_WELMV4_MK_DECODE_FLASH_KV_BOUND",
                "SGLANG_WELMV4_MK_DECODE_FLASH_FULL_UNMASKED",
                "SGLANG_WELMV4_MK_DECODE_PRE_MOE_MASK_NORM",
                "SGLANG_WELMV4_MK_DECODE_KV_PROLOGUE",
                "SGLANG_DEEPEP_NORMAL_USE_ALLTOALL",
                "SGLANG_SCHEDULER_SKIP_ALL_GATHER",
            ),
            False,
        )
        self.flags.update(
            SGLANG_WELMV4_MK_DECODE_FAST_PATH=True,
            SGLANG_WELMV4_DECODE_RS_AG=True,
            SGLANG_DEEPEP_NORMAL_USE_ALLGATHER=True,
            SGLANG_ZBAL_LOCAL_MEM_SIZE=0,
        )
        self.args = SimpleNamespace(
            tp_size=8,
            dp_size=8,
            nnodes=1,
            moe_dense_tp_size=1,
            enable_dp_attention=True,
            enable_dp_lm_head=True,
            moe_a2a_backend="deepep",
            deepep_mode="auto",
            enable_memory_saver=False,
            enable_two_batch_overlap=False,
            enable_single_batch_overlap=False,
            enable_lora=False,
            enable_eplb=False,
            ep_num_redundant_experts=0,
        )
        self.ps = SimpleNamespace(
            tp_size=8,
            pp_size=1,
            attn_dp_size=8,
            attn_tp_size=1,
            attn_cp_size=1,
            dcp_size=1,
            moe_ep_size=8,
        )
        self.runner = SimpleNamespace(
            enable_pdmux=False,
            disable_padding=False,
            capture_bs=[1, 2, 4, 8],
            attn_backend=None,
            model_runner=SimpleNamespace(
                server_args=self.args,
                ps=self.ps,
                max_running_requests=8,
                welm_decode_ll_dispatch_capacity=8,
                is_draft_worker=False,
                spec_algorithm=SimpleNamespace(is_none=lambda: True),
                model_config=SimpleNamespace(
                    hf_config=SimpleNamespace(
                        architectures=["WeLMV4MoeForCausalLM"], num_hidden_layers=0
                    )
                ),
                model=SimpleNamespace(model=SimpleNamespace(layers=[])),
            ),
        )

    def create(self, *, direct_replay=True, router=None, postmoe=None):
        env = {"WELM_NPU_USE_FLASH_ATTN": "1"}
        if router is not None:
            env["WELM_DP8_ROUTER_FUSION"] = router
        if postmoe is not None:
            env["WELM_DP8_POSTMOE_FUSION"] = postmoe
        with patch.dict(os.environ, env, clear=True):
            backend = stock_backend(self.flags)(self.runner, direct_replay=direct_replay)
        self.addCleanup(backend._directory.cleanup)
        return backend

    def test_every_dp8_bucket_including_b1_uses_stock(self):
        for mode in ("auto", "low_latency"):
            self.args.deepep_mode = mode
            backend = self.create()
            for bucket in self.runner.capture_bs:
                with self.subTest(mode=mode, bucket=bucket):
                    self.assertEqual(
                        backend._decode_capture_options(bucket),
                        dict(welm_stock_decode_replay=True, welm_decode_batch_size=bucket),
                    )
                    self.assertFalse(backend._use_postmoe_fusion(bucket))

    def test_one_switch_controls_router_fusion_for_every_dp8_bucket(self):
        for mode in ("0", "1"):
            backend = self.create(router=mode)
            for bucket in self.runner.capture_bs:
                with self.subTest(mode=mode, bucket=bucket):
                    self.assertEqual(
                        backend._decode_capture_options(bucket).get("welm_fuse_router"),
                        True if mode == "1" else None,
                    )

    def test_router_fusion_rejects_non_binary_values_and_non_direct_dp8(self):
        with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
            self.create(router="scope")
        with self.assertRaisesRegex(ValueError, "requires megakernel-native DP8"):
            self.create(router="1", direct_replay=False)
        self.args.dp_size = 4
        with self.assertRaisesRegex(ValueError, "requires megakernel-native DP8"):
            self.create(router="1")

    def test_postmoe_switch_preserves_b1_and_is_independent_of_router(self):
        for postmoe in ("0", "1"):
            for router in ("0", "1"):
                backend = self.create(router=router, postmoe=postmoe)
                for bucket in self.runner.capture_bs:
                    with self.subTest(postmoe=postmoe, router=router, bucket=bucket):
                        opts = backend._decode_capture_options(bucket)
                        self.assertEqual(
                            opts.get("welm_fuse_router", False), router == "1"
                        )
                        self.assertEqual(
                            backend._use_postmoe_fusion(bucket),
                            postmoe == "1" and bucket != 1,
                        )

    def test_postmoe_switch_requires_direct_dp8(self):
        with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
            self.create(postmoe="unknown")
        with self.assertRaisesRegex(ValueError, "requires megakernel-native DP8"):
            self.create(direct_replay=False, postmoe="1")
        self.args.dp_size = self.args.tp_size = 4
        with self.assertRaisesRegex(ValueError, "requires megakernel-native DP8"):
            self.create(postmoe="1")

    def test_postmoe_capture_bucket_does_not_change_router_scope(self):
        get_bucket = POSTMOE["postmoe_capture_bucket"]
        capture_scope = POSTMOE["postmoe_capture_scope"]
        get_router_bucket = ROUTER["router_capture_bucket"]
        self.assertIsNone(get_bucket())
        with ROUTER["router_capture_scope"](2):
            with self.assertRaisesRegex(RuntimeError, "capture failed"):
                with capture_scope(8):
                    self.assertEqual(get_bucket(), 8)
                    self.assertEqual(get_router_bucket(), 2)
                    raise RuntimeError("capture failed")
            self.assertIsNone(get_bucket())
            self.assertEqual(get_router_bucket(), 2)

    def test_router_bucket_is_restored_after_capture_failure(self):
        get_bucket = ROUTER["router_capture_bucket"]
        capture_scope = ROUTER["router_capture_scope"]
        self.assertIsNone(get_bucket())
        with capture_scope(1):
            self.assertEqual(get_bucket(), 1)
        self.assertIsNone(get_bucket())
        with capture_scope(2):
            self.assertEqual(get_bucket(), 2)
            with self.assertRaisesRegex(RuntimeError, "capture failed"):
                with capture_scope(8):
                    self.assertEqual(get_bucket(), 8)
                    raise RuntimeError("capture failed")
            self.assertEqual(get_bucket(), 2)
        self.assertIsNone(get_bucket())

    def test_dp4_and_legacy_acl_capture_do_not_opt_into_stock(self):
        self.args.dp_size = self.args.tp_size = 4
        backend = self.create()
        backend._mask_norm_layers = (1, 2)
        self.assertEqual(
            backend._decode_capture_options(1), {"welm_mask_norm_layers": (1, 2)}
        )
        self.assertEqual(
            backend._decode_capture_options(4), {"welm_decode_batch_size": 4}
        )
        self.args.dp_size = self.args.tp_size = 8
        self.assertEqual(self.create(direct_replay=False)._decode_capture_options(1), {})

    def test_rejects_mismatched_topology_and_transport(self):
        for owner, field, value, message in (
            (self.args, "enable_dp_attention", False, "TP8/DP8/EP8"),
            (self.args, "enable_dp_lm_head", False, "TP8/DP8/EP8"),
            (self.args, "moe_dense_tp_size", 2, "TP8/DP8/EP8"),
            (self.args, "nnodes", 2, "TP8/DP8/EP8"),
            (self.ps, "attn_cp_size", 2, "TP8/DP8/EP8"),
            (self.ps, "moe_ep_size", 4, "TP8/DP8/EP8"),
            (self.args, "deepep_mode", "normal", "stock DeepEP LL"),
            (self.args, "moe_a2a_backend", "ascend_fuseep", "stock DeepEP LL"),
            (self.args, "enable_eplb", True, "stock DeepEP LL"),
            (self.runner, "disable_padding", True, "common bucket"),
        ):
            with self.subTest(field=field):
                previous = getattr(owner, field)
                try:
                    setattr(owner, field, value)
                    with self.assertRaisesRegex(ValueError, message):
                        self.create()
                finally:
                    setattr(owner, field, previous)

    def test_rejects_experimental_flags_and_missing_scheduler_gather(self):
        for flag in tuple(self.flags):
            if not flag.startswith("SGLANG_WELMV4_MK_") or self.flags[flag]:
                continue
            with self.subTest(flag=flag):
                self.flags[flag] = True
                with self.assertRaisesRegex(ValueError, flag):
                    self.create()
                self.flags[flag] = False
        self.flags["SGLANG_SCHEDULER_SKIP_ALL_GATHER"] = True
        with self.assertRaisesRegex(ValueError, "scheduler DP AllGather"):
            self.create()
        self.flags["SGLANG_SCHEDULER_SKIP_ALL_GATHER"] = False
        self.flags["SGLANG_DEEPEP_NORMAL_USE_ALLGATHER"] = False
        with self.assertRaisesRegex(ValueError, "NORMAL AllGather group"):
            self.create()
        self.flags["SGLANG_DEEPEP_NORMAL_USE_ALLGATHER"] = True
        self.flags["SGLANG_ZBAL_LOCAL_MEM_SIZE"] = 1
        with self.assertRaisesRegex(ValueError, "stock DeepEP LL"):
            self.create()

    def test_bucket_and_deepep_capacity_must_cover_local_requests(self):
        self.runner.capture_bs = [1, 2, 4, 16]
        with self.assertRaisesRegex(ValueError, "buckets must come"):
            self.create()
        self.runner.capture_bs = [1, 2, 4]
        with self.assertRaisesRegex(ValueError, "got 8, 4, 8"):
            self.create()
        self.runner.model_runner.max_running_requests = 4
        self.runner.model_runner.welm_decode_ll_dispatch_capacity = 2
        with self.assertRaisesRegex(ValueError, "got 4, 4, 2"):
            self.create()
        self.runner.model_runner.welm_decode_ll_dispatch_capacity = None
        with self.assertRaisesRegex(ValueError, "got 4, 4, None"):
            self.create()
        self.runner.model_runner.welm_decode_ll_dispatch_capacity = 4
        self.create()

    def test_global_selection_keeps_idle_peers_on_active_ranks_bucket(self):
        target_request_bs = load_method(
            "_welm_target_request_bs",
            "DecodeCudaGraphRunner",
            "decode_cuda_graph_runner.py",
            ForwardBatch=SimpleNamespace,
            Optional=Optional,
            _welm_target_plan=lambda _: SimpleNamespace(outer_target_dp_size=8),
        )
        pad_to_bucket = load_method(
            "_pad_to_bucket",
            "BaseCudaGraphRunner",
            "base_cuda_graph_runner.py",
            Sequence=Sequence,
            bisect=bisect,
        )
        for counts, expected in (
            ([0, 0, 0, 0, 0, 0, 0, 1], 1),
            ([2, 2, 2, 2, 2, 2, 2, 2], 2),
            ([3, 0, 0, 0, 1, 0, 0, 0], 4),
            ([0, 0, 0, 0, 5, 1, 0, 0], 8),
            ([0, 8, 0, 1, 0, 0, 0, 0], 8),
        ):
            with self.subTest(counts=counts):
                for local_bs in counts:
                    fb = SimpleNamespace(
                        batch_size=local_bs, global_num_requests_cpu=counts
                    )
                    req_bs = target_request_bs(self.runner, fb)
                    self.assertEqual(req_bs, max(counts))
                    self.assertEqual(
                        pad_to_bucket(req_bs, self.runner.capture_bs), expected
                    )
        with self.assertRaisesRegex(
            RuntimeError, "synchronized global_num_requests_cpu"
        ):
            target_request_bs(
                self.runner, SimpleNamespace(global_num_requests_cpu=[1, 0, 0, 0])
            )


if __name__ == "__main__":
    unittest.main()
