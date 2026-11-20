"""CPU-only validation of fixed WeLM EP transport admission and lifetime."""

import ast
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(os.environ.get("WELM_TRANSPORT_SGLANG_ROOT", Path(__file__).resolve().parents[4]))
DISPATCHER = ROOT / "python/sglang/srt/layers/moe/token_dispatcher"
scope_spec = importlib.util.spec_from_file_location("_welm_ep_host", DISPATCHER / "welm_ep_allgather.py")
scope_module = importlib.util.module_from_spec(scope_spec)
sys.modules[scope_spec.name] = scope_module
scope_spec.loader.exec_module(scope_module)


class Workspace:
    def __init__(self):
        self.pack, self.counts, self.result = object(), object(), object()
        self.calls = []

    def dispatch(self, h, ids, weights):
        self.calls.append(("dispatch", h, ids, weights))
        return self.pack, self.counts

    def combine(self, down, ids, weights):
        self.calls.append(("combine", down, ids, weights))
        return self.result


def dispatcher(g, **overrides):
    defaults = dict(group=g, router_topk=10, num_experts=512, num_local_experts=128,
                    hidden_size=2048, use_fp8=False, use_nvfp4=False, overlap_args=None,
                    meta_overlap_args=None, quant_config={},
                    _resolve_num_max_dispatch_tokens_per_rank=lambda _: 8)
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def cycle(scope, d):
    h, ids, weights = SimpleNamespace(shape=(1, 2048), device="npu:0"), object(), object()
    packed, counts, ticket = scope.dispatch(d, h, ids, weights)
    result = scope.engine_combine(d, ticket, object(), ids, weights)
    return packed, counts, result


class TestWelmEpAllGather(unittest.TestCase):
    def test_default_off_needs_no_npu_or_group(self):
        with scope_module.welm_ep_allgather_scope(False) as scope:
            self.assertIsNone(scope)
            self.assertIsNone(scope_module.active_welm_ep_allgather())

    def test_all_forward_passes_pair_every_layer_and_reuse_the_exact_workspace(self):
        group = object()
        layer0, layer1 = dispatcher(group), dispatcher(group)
        w0, w1 = Workspace(), Workspace()
        with patch.object(scope_module._WelmEpAllGatherScope, "_create_workspace", side_effect=[w0, w1]) as make:
            with scope_module.welm_ep_allgather_scope(True, group=group, expected_layers=2) as scope:
                for _ in range(3):
                    with scope.forward_pass():
                        self.assertEqual(cycle(scope, layer0), (w0.pack, w0.counts, w0.result))
                        self.assertEqual(cycle(scope, layer1), (w1.pack, w1.counts, w1.result))
                scope.assert_complete()
            self.assertEqual(make.call_count, 2)
        self.assertIsNone(scope_module.active_welm_ep_allgather())
        self.assertEqual([row[0] for row in w0.calls], ["dispatch", "combine"] * 3)

    def test_second_forward_cannot_inherit_previous_layer_coverage(self):
        group = object()
        d0, d1 = dispatcher(group), dispatcher(group)
        with patch.object(scope_module._WelmEpAllGatherScope, "_create_workspace", side_effect=[Workspace(), Workspace()]):
            with scope_module.welm_ep_allgather_scope(True, group=group, expected_layers=2) as scope:
                with scope.forward_pass():
                    cycle(scope, d0)
                    cycle(scope, d1)
                with self.assertRaisesRegex(RuntimeError, "captured 1/2"):
                    with scope.forward_pass():
                        cycle(scope, d0)

    def test_rejects_rank_local_shape_group_and_feature_mismatch_before_workspace(self):
        group = object()
        variants = (dispatcher(object()), dispatcher(group, router_topk=8),
                    dispatcher(group, use_fp8=True), dispatcher(group, overlap_args=object()),
                    dispatcher(group, _resolve_num_max_dispatch_tokens_per_rank=lambda _: 1))
        with patch.object(scope_module._WelmEpAllGatherScope, "_create_workspace") as make:
            for d in variants:
                with self.assertRaisesRegex(RuntimeError, "Incompatible"):
                    with scope_module.welm_ep_allgather_scope(True, group=group) as scope:
                        with scope.forward_pass():
                            cycle(scope, d)
                self.assertIsNone(scope_module.active_welm_ep_allgather())
            make.assert_not_called()

    def test_rejects_missing_combine_and_stale_or_repeated_ticket(self):
        group = object()
        d = dispatcher(group)
        w = Workspace()
        h, ids, weights = SimpleNamespace(shape=(1, 2048), device="npu:0"), object(), object()
        with patch.object(scope_module._WelmEpAllGatherScope, "_create_workspace", return_value=w):
            with scope_module.welm_ep_allgather_scope(True, group=group, expected_layers=1) as scope:
                with self.assertRaisesRegex(RuntimeError, "forward scope"):
                    cycle(scope, d)
                with scope.forward_pass():
                    _, _, ticket = scope.dispatch(d, h, ids, weights)
                    with self.assertRaisesRegex(RuntimeError, "does not match"):
                        scope.engine_combine(d, ticket, object(), object(), weights)
                    scope.engine_combine(d, ticket, object(), ids, weights)
                    with self.assertRaisesRegex(RuntimeError, "Repeated"):
                        cycle(scope, d)
                with self.assertRaisesRegex(RuntimeError, "unfinished dispatches"):
                    with scope.forward_pass():
                        scope.dispatch(d, h, ids, weights)

    def test_producer_and_consumer_keep_stock_group_and_disjoint_captured_addresses(self):
        tree = ast.parse((DISPATCHER / "welm_ep_allgather_npu.py").read_text())
        src = ast.unparse(tree)
        self.assertIn("dist.all_gather_into_tensor(out, send, group=self.group, async_op=True)", src)
        self.assertIn("work.wait()", src)
        self.assertNotIn("reduce_scatter", src)
        self.assertIn("enable_fp_fusion=False", src)
        self.assertIn("with _disable_current_modes():", src)
        self.assertNotIn(".item()", src)


if __name__ == "__main__":
    unittest.main()
