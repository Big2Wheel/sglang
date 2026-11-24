"""Host-only contract tests for WeLMV4 MegaKernel's Decode backend."""

import ast
import os
import sys
import unittest
from contextlib import nullcontext
from dataclasses import dataclass, fields
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

SOURCE = Path(__file__).resolve().parents[4] / "python/sglang/srt"


def load_code(path, name, globals_):
    source = SOURCE / path
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    node = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name
    )
    namespace = dict(globals_)
    # Load selected repository definitions without importing Torch or NPU plugins.
    exec(compile(ast.Module(body=[node], type_ignores=[]), source, "exec"), namespace)  # noqa: S102
    return namespace[name]


@dataclass
class Logits:
    next_token_logits: object


def batch(decode_or_idle=True):
    return SimpleNamespace(
        forward_mode=SimpleNamespace(
            is_decode_or_idle=lambda: decode_or_idle,
        ),
        spec_info=None,
    )


class TestWelmMKNativeBaseline(unittest.TestCase):
    def test_cli_validation(self):
        names = ("SGLANG_NPU_USE_MULTI_STREAM", "SGLANG_ENABLE_OVERLAP_PLAN_STREAM")
        flags = SimpleNamespace(
            **{name: SimpleNamespace(get=lambda: False) for name in names}
        )
        validate = load_code(
            "server_args.py",
            "_validate_megakernel_native",
            {"Backend": SimpleNamespace(FULL="full"), "envs": flags, "os": os},
        )
        validate(SimpleNamespace(model_execution_backend="native"))
        args = {
            "model_execution_backend": "megakernel-native",
            "cuda_graph_config": SimpleNamespace(
                decode=SimpleNamespace(bs=[1], backend="full")
            ),
            "device": "npu",
            "speculative_algorithm": None,
            "enable_pdmux": False,
            "enable_two_batch_overlap": False,
            "enable_single_batch_overlap": False,
            "enable_torch_compile": False,
            "enable_memory_saver": False,
            "enable_lora": False,
            "pp_size": 1,
            "get_model_config": lambda: SimpleNamespace(
                hf_config=SimpleNamespace(architectures=["WeLMV4MoeForCausalLM"])
            ),
            "get_attention_backends": lambda: ("ascend", "ascend"),
        }
        enabled = {
            "WELM_NPU_USE_FLASH_ATTN": "1",
            "MK_FIXED_REPLAY_TRITON_GLOBALS": "1",
        }
        with patch.dict(os.environ, enabled, clear=True):
            validate(SimpleNamespace(**args))
            for key, value in (
                ("device", "cpu"),
                ("speculative_algorithm", "NEXTN"),
                ("enable_pdmux", True),
                ("enable_two_batch_overlap", True),
                ("pp_size", 2),
                (
                    "cuda_graph_config",
                    SimpleNamespace(decode=SimpleNamespace(bs=[1, 2], backend="full")),
                ),
            ):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    validate(SimpleNamespace(**(args | {key: value})))
        with (
            patch.dict(os.environ, {}, clear=True),
            self.assertRaisesRegex(ValueError, "WELM_NPU_USE_FLASH_ATTN"),
        ):
            validate(SimpleNamespace(**args))
        with (
            patch.dict(os.environ, {"WELM_NPU_USE_FLASH_ATTN": "1"}, clear=True),
            self.assertRaisesRegex(ValueError, "MK_FIXED_REPLAY_TRITON_GLOBALS"),
        ):
            validate(SimpleNamespace(**args))

    def test_default_does_not_select_mk_through_legacy_env(self):
        native, direct = Mock(), Mock()
        resolve = load_code(
            "model_executor/runner_backend/utils.py",
            "resolve_decode_backend",
            {
                "Backend": SimpleNamespace(FULL="full"),
                "BaseCudaGraphRunner": object,
                "BaseCudaGraphBackend": object,
            },
        )
        root = "sglang.srt.hardware_backend.npu.graph_runner."
        modules = {
            root + "welm_mk_decode_backend": SimpleNamespace(
                WelmMKDecodeBackend=direct
            ),
            root + "npu_cudagraph_backend": SimpleNamespace(NPUCudaGraphBackend=native),
        }
        args = SimpleNamespace(
            cuda_graph_config=None,
            enable_memory_saver=False,
            model_execution_backend="native",
        )
        runner = SimpleNamespace(
            model_runner=SimpleNamespace(device="npu", server_args=args)
        )
        with patch.dict(sys.modules, modules):
            with patch.dict(
                os.environ, {"WELM_NPU_USE_MK_TARGET_DECODE": "1"}, clear=True
            ):
                self.assertIs(resolve(runner), native.return_value)
            direct.assert_not_called()
            args.model_execution_backend = "megakernel-native"
            self.assertIs(resolve(runner), direct.return_value)

    def test_invalid_shape_is_not_sent_to_eager(self):
        check = load_code(
            "model_executor/runner/decode_cuda_graph_runner.py",
            "can_run_graph",
            {"ForwardBatch": object},
        )
        runner = SimpleNamespace(
            _can_run_graph=lambda _: False,
            model_runner=SimpleNamespace(
                server_args=SimpleNamespace(model_execution_backend="megakernel-native")
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "not eligible"):
            check(runner, batch())
        self.assertFalse(check(runner, batch(False)))
        runner.model_runner.server_args.model_execution_backend = "native"
        self.assertFalse(check(runner, batch()))

    def set_up_backend(self):
        cls = load_code(
            "hardware_backend/npu/graph_runner/welm_mk_decode_backend.py",
            "WelmMKDecodeBackend",
            {
                "BaseCudaGraphBackend": object,
                "fields": fields,
                "LogitsProcessorOutput": Logits,
                "nullcontext": nullcontext,
                "logger": Mock(),
                "torch": SimpleNamespace(no_grad=lambda: lambda fn: fn),
            },
        )
        self.backend = cls.__new__(cls)
        self.key = object()
        self.backend._shape = self.key
        self.backend._graph = SimpleNamespace(
            replay_direct=Mock(side_effect=lambda before_launch: before_launch())
        )
        self.backend._replays = 0
        self.backend._output = Logits(object())
        layout = {"shape": (32,), "dtype": "int32", "device": "npu:0"}
        self.schedule = SimpleNamespace(**layout, copy_=Mock())
        self.backend._metadata = SimpleNamespace(
            welm_mk_schedule_recipes={"full": ((24, 2, 256), {"mask_mode": 3})},
            welm_flash_schedules={"full": self.schedule},
        )
        self.backend._attn_backend = SimpleNamespace(
            forward_metadata=self.backend._metadata,
            _welm_flash_attn_metadata=Mock(
                side_effect=[SimpleNamespace(**layout, page=page) for page in (1, 2)]
            ),
        )

    def test_direct_replay_refreshes_the_same_buffers(self):
        self.set_up_backend()
        with self.backend.capture_session(object()):
            pass
        for _ in range(2):
            self.assertIs(self.backend.replay(self.key, batch()), self.backend._output)
        self.assertEqual(self.backend._replays, 2)
        self.assertEqual(
            [call.args[0].page for call in self.schedule.copy_.call_args_list],
            [1, 2],
        )
        self.assertEqual(self.backend._graph.replay_direct.call_count, 2)

    def test_metadata_and_shape_fail_before_replay(self):
        self.set_up_backend()
        with self.assertRaisesRegex(RuntimeError, "No prepared"):
            self.backend.replay(object(), batch())
        self.backend._attn_backend.forward_metadata = object()
        with self.assertRaisesRegex(RuntimeError, "bindings changed"):
            self.backend.replay(self.key, batch())
        self.backend._graph.replay_direct.assert_not_called()
        self.assertEqual(self.backend._replays, 0)

    def test_flash_schedule_layout_mismatch_fails_before_copy(self):
        for key, wrong in (("shape", (1,)), ("dtype", "int64"), ("device", "npu:1")):
            with self.subTest(key=key):
                self.set_up_backend()
                updated = SimpleNamespace(shape=(32,), dtype="int32", device="npu:0")
                setattr(updated, key, wrong)
                self.backend._attn_backend._welm_flash_attn_metadata = Mock(
                    return_value=updated
                )
                with self.assertRaisesRegex(RuntimeError, "layout changed"):
                    self.backend.replay(self.key, batch())
                self.schedule.copy_.assert_not_called()
                self.assertEqual(self.backend._replays, 0)

    def test_capture_rejects_host_work_or_missing_layer_coverage(self):
        for host_count, fused in ((0, (0, 2)), (1, (0, 2)), (0, (0,))):
            with self.subTest(host_count=host_count, fused=fused):
                self.check_capture(host_count, fused)

    def check_capture(self, host_count, fused):
        self.set_up_backend()
        hooks = [[], []]

        def register(target):
            def install(hook):
                target.append(hook)
                return SimpleNamespace(remove=lambda: target.remove(hook))

            return install

        self.backend._layers = [
            SimpleNamespace(
                register_forward_pre_hook=register(layer_hooks),
                register_forward_hook=register(layer_hooks),
            )
            for layer_hooks in hooks
        ]
        self.backend._device_module = SimpleNamespace(synchronize=Mock())
        self.backend._tp_group = SimpleNamespace(barrier=Mock())
        self.backend._graph = None
        self.backend._directory = SimpleNamespace(name="mock")
        metadata = self.backend._metadata
        metadata.welm_decode_metadata_in_graph = False

        def forward():
            if metadata.welm_mk_schedule_bootstrap:
                metadata.welm_mk_schedule_recipes["full"] = ((), {})
            for pre, post in (layer_hooks for layer_hooks in hooks if layer_hooks):
                pre()
                post()
            return Logits(object())

        graph = SimpleNamespace(
            replay_direct=Mock(),
            reset=Mock(),
            execution_summary={},
            host_boundary_count=host_count,
            updates=[],
            plans=[SimpleNamespace(task_indices=fused)],
            outputs=(object(),),
        )

        def record(fn, path, *, prepared):
            self.assertTrue(prepared)
            self.assertEqual(len(fn()), 1)
            return graph

        capture = SimpleNamespace(
            launches=SimpleNamespace(checkpoint=Mock(side_effect=range(4)))
        )
        modules = {
            "megakernel._capture_access": SimpleNamespace(
                _ACTIVE=SimpleNamespace(get=lambda: capture)
            ),
            "megakernel._fixed": SimpleNamespace(
                capture_fixed=record, prepare_static=Mock()
            ),
            "megakernel.ops.tensor_concat": SimpleNamespace(LowerConcats=nullcontext),
        }
        shape = SimpleNamespace(size=1, stream_idx=None, variant_label=None)
        with patch.dict(sys.modules, modules):
            if host_count or len(fused) < 2:
                with self.assertRaisesRegex(RuntimeError, "device-only|coverage"):
                    self.backend.capture_one(shape, forward)
                graph.reset.assert_called_once()
                self.assertIsNone(self.backend._graph)
            else:
                self.backend.capture_one(shape, forward)
                self.assertIs(self.backend._graph, graph)
        self.assertEqual(hooks, [[], []])


if __name__ == "__main__":
    unittest.main()
