"""Validate WeLM EP AllGather capture wiring without loading TorchNPU."""

import ast
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, fields
import json
import logging
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch


BACKEND = (
    Path(__file__).resolve().parents[4] / "python/sglang/srt/hardware_backend"
    / "npu/graph_runner/welm_mk_decode_backend.py"
)
MODULE = "sglang.srt.layers.moe.token_dispatcher.welm_ep_allgather"


@dataclass
class Output:
    logits: str


@dataclass(frozen=True)
class Shape:
    size: int
    stream_idx: int | None = None
    variant_label: str | None = None


def load_backend_methods():
    # Importing the full NPU runner requires TorchNPU. Exercise the real
    # capture methods in isolation and stub only their graph/runtime helpers.
    tree = ast.parse(BACKEND.read_bytes())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "WelmMKDecodeBackend"
    )
    methods = [x for x in cls.body if isinstance(x, ast.FunctionDef)
               and x.name in ("capture_one", "_capture_one")]
    for node in methods:
        node.decorator_list.clear()
    glob = dict(fields=fields, Path=Path, LogitsProcessorOutput=Output,
                logger=logging.getLogger(__name__), json=json, os=os)
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(BACKEND), "exec"), glob)
    return type("Backend", (), {x.name: glob[x.name] for x in methods})


def module(name, **attributes):
    m = ModuleType(name)
    m.__dict__.update(attributes)
    return m


class Scope:
    def __init__(self):
        self.forwarding = False
        self.started = 0
        self.finished = 0

    @contextmanager
    def forward_pass(self):
        assert not self.forwarding
        self.started += 1
        self.forwarding = True
        try:
            yield
        finally:
            self.forwarding = False
        self.finished += 1

    def assert_complete(self):
        assert self.started == self.finished == 4  # Bootstrap, two prepares, record.


class Layer:
    def register_forward_pre_hook(self, callback):
        self.before = callback
        return NS(remove=lambda: setattr(self, "before", None))

    def register_forward_hook(self, callback):
        self.after = callback
        return NS(remove=lambda: setattr(self, "after", None))

    before = after = None


class TestBackendWiring(unittest.TestCase):
    def run_backend(self, *, enabled, fail_audit=False):
        backend = load_backend_methods()()
        group = object() if enabled else None
        scope = Scope() if enabled else None
        trace = []
        layer = Layer()
        active = NS(get=lambda: None)
        mode = NS(is_decode_or_idle=lambda: True)
        batch = NS(forward_mode=mode, spec_info=None, seq_lens=NS(shape=(1,)),
                   out_cache_loc=NS(shape=(1,)))
        metadata = NS(welm_decode_metadata_in_graph=True, seq_lens=batch.seq_lens)
        key = Shape(1)
        backend.__dict__.update(
            _direct_replay=False, _ep_allgather_group=group, _layer_count=1,
            _decoder_layers=(layer,),
            _mask_norm_layers=(0,), _native_router_sigmoid=True, _phase="DECODE",
            _fuse_decode_kv_prologue=True, _graphs={}, _outputs={}, _metadata={},
            _page_prologue_batches={}, _replays={}, uses_fixed_decode_prologue=True,
            _device_module=NS(synchronize=lambda: None), _tp_group=NS(barrier=lambda: None),
            _directory=NS(name="/tmp/not-used-by-host-test"),
            _attn_backend=NS(forward_metadata=metadata),
        )

        @contextmanager
        def providing_scope(on, *, group, expected_layers):
            self.assertTrue(on)
            self.assertIs(group, backend._ep_allgather_group)
            self.assertEqual(expected_layers, 1)
            trace.append("entered")
            yield scope
            trace.append("exited")

        def forward():
            if enabled:
                self.assertTrue(scope.forwarding)
            if metadata.welm_mk_schedule_bootstrap:
                metadata.welm_mk_schedule_recipes["fixture"] = True
            if layer.before:
                layer.before(layer, ())
                layer.after(layer, (), None)
            return Output("logits")

        def prepare(fn, _path):
            self.assertEqual(fn is forward, not enabled)
            fn()
            fn()

        def ep_audit(ranges):
            self.assertEqual(ranges, {0: [0, 1]})
            if fail_audit:
                raise RuntimeError("bad capture proof")
            trace.append("ep audit")
            return {"proof": True}

        graph = NS(
            execution_summary={}, plans=[NS(task_indices=(0,))], fused_operator_count=1,
            updates=[], host_boundary_count=0, outputs=("logits",),
            reset=lambda: trace.append("reset"), audit_welm_fixed_allgather=ep_audit,
            audit_welm_fixed_kv_prologue=lambda ranges: (1, 2, 3),
        )

        def capture(fn, _path, **kwargs):
            self.assertEqual("welm_ep_allgather_group" in kwargs, enabled)
            if enabled:
                self.assertIs(kwargs["welm_ep_allgather_group"], group)
            starts = iter((0, 1))
            active.get = lambda: NS(launches=NS(checkpoint=lambda: next(starts)))
            self.assertEqual(fn(), ("logits",))
            active.get = lambda: None
            return graph

        modules = {
            MODULE: module(MODULE, welm_ep_allgather_scope=providing_scope),
            "megakernel._fixed": module("megakernel._fixed",
                                        capture_fixed=capture, prepare_static=prepare),
            "megakernel._capture_access": module("megakernel._capture_access", _ACTIVE=active),
            "megakernel.ops.tensor_concat": module("megakernel.ops.tensor_concat",
                                                  LowerConcats=nullcontext),
        }
        with patch.dict(sys.modules, modules):
            if fail_audit:
                with self.assertRaisesRegex(RuntimeError, "bad capture proof"):
                    backend.capture_one(key, forward, batch)
                self.assertEqual(trace, ["entered", "reset"])
                self.assertEqual(backend._graphs, {})
            else:
                backend.capture_one(key, forward, batch)
                self.assertIs(backend._graphs[key], graph)
                self.assertEqual(backend._outputs[key].logits, "logits")
                self.assertEqual(trace, ["entered", "ep audit", "exited"] if enabled else [])
                if enabled:
                    self.assertIs(graph._welm_ep_allgather_owner, scope)
                else:
                    self.assertFalse(hasattr(graph, "_welm_ep_allgather_owner"))

    def test_disabled_does_not_wrap_prepare_or_pass_group(self):
        self.run_backend(enabled=False)

    def test_enabled_wraps_all_actual_forwards_and_pins_workspace(self):
        self.run_backend(enabled=True)

    def test_rejected_proof_resets_graph_and_does_not_publish(self):
        self.run_backend(enabled=True, fail_audit=True)

    def test_rejects_other_capture_shape_or_variant(self):
        backend = load_backend_methods()()
        backend._ep_allgather_group = object()
        for key in (Shape(2), Shape(1, stream_idx=2), Shape(1, variant_label="lora")):
            with self.assertRaisesRegex(ValueError, "fixed single-row"):
                backend.capture_one(key, lambda: None)


if __name__ == "__main__":
    unittest.main()
