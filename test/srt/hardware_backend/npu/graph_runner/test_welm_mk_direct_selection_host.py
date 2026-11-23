"""CPU-only tests for the direct-mode selector and benchmark evidence gate."""

import ast
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[5]


def load_benchmark(name):
    src = ROOT / "benchmark" / "welmv4_decode_only" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, src)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_selector(envs):
    src = ROOT / "python" / "sglang" / "srt" / "server_args.py"
    tree = ast.parse(src.read_text())
    server_args = next(
        x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "ServerArgs"
    )
    method = next(
        x
        for x in server_args.body
        if getattr(x, "name", "") == "_validate_megakernel_native"
    )
    namespace = {"Backend": SimpleNamespace(FULL="full"), "envs": envs, "os": os}
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(src), "exec"), namespace
    )
    return namespace[method.name]


def direct_log():
    layers = {layer: (2, 1) for layer in range(48)}
    for rank in range(4):
        prefix = f"[2026-11-23 DP{rank} TP{rank} EP{rank}] "
        yield f"{prefix}welm-mk-target recorded shape=ShapeKey(size=1, ) layer_launches(total,fused)={layers} coverage={{'torch_boundaries': 0}}"
        yield f"{prefix}WeLM MK direct replay: scopes=12 steps=48 acl_graph=False"
        yield f"{prefix}welm-mk-target replay shape=ShapeKey(size=1, ) count=1 backend=mk-direct"


class TestModeSelection(unittest.TestCase):
    def setUp(self):
        self.envs = SimpleNamespace(
            SGLANG_WELMV4_MK_DECODE_FAST_PATH=SimpleNamespace(
                get=Mock(return_value=True)
            ),
            SGLANG_WELMV4_DECODE_RS_AG=SimpleNamespace(get=Mock(return_value=True)),
            SGLANG_FORCE_COARSE_WAR_BARRIER=SimpleNamespace(
                get=Mock(return_value=True)
            ),
            SGLANG_NPU_USE_MULTI_STREAM=SimpleNamespace(get=Mock(return_value=False)),
            SGLANG_WELMV4_MK_EP_ALLGATHER=SimpleNamespace(get=Mock(return_value=False)),
        )
        self.check = check_selector(self.envs)
        self.valid = SimpleNamespace(
            model_execution_backend="megakernel-native",
            device="npu",
            cuda_graph_config=SimpleNamespace(decode=SimpleNamespace(backend="full")),
            enable_torch_compile=False,
            enable_memory_saver=False,
            speculative_algorithm=None,
            get_model_config=lambda: SimpleNamespace(
                hf_config=SimpleNamespace(architectures=["WeLMV4MoeForCausalLM"])
            ),
        )
        self.flags = {
            "WELM_NPU_USE_MK_TARGET_DECODE": "1",
            "WELM_NPU_USE_FLASH_ATTN": "1",
        }

    def test_valid_npu_and_bad_options(self):
        with patch.dict(os.environ, self.flags, clear=True):
            self.check(self.valid)
        self.check(SimpleNamespace(model_execution_backend="native"))
        for field, value in (
            ("device", "cpu"),
            ("speculative_algorithm", "NEXTN"),
            ("enable_torch_compile", True),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.check(SimpleNamespace(**(vars(self.valid) | {field: value})))
        self.valid.cuda_graph_config.decode.backend = "disabled"
        with self.assertRaisesRegex(ValueError, "full SGLang Decode runner"):
            self.check(self.valid)

    def test_requires_fixed_decode_flags(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "WELM_NPU_USE_MK_TARGET_DECODE=1"):
                self.check(self.valid)
        for flag in (
            "SGLANG_WELMV4_MK_DECODE_FAST_PATH",
            "SGLANG_FORCE_COARSE_WAR_BARRIER",
        ):
            with (
                self.subTest(flag=flag),
                patch.dict(os.environ, self.flags, clear=True),
            ):
                value = getattr(self.envs, flag)
                value.get.return_value = False
                with self.assertRaisesRegex(ValueError, f"{flag}=1"):
                    self.check(self.valid)
                value.get.return_value = True

    def test_rejects_multistream_and_allgather(self):
        with patch.dict(os.environ, self.flags, clear=True):
            self.envs.SGLANG_NPU_USE_MULTI_STREAM.get.return_value = True
            with self.assertRaisesRegex(ValueError, "SGLANG_NPU_USE_MULTI_STREAM=0"):
                self.check(self.valid)
            self.envs.SGLANG_NPU_USE_MULTI_STREAM.get.return_value = False
            self.envs.SGLANG_WELMV4_MK_EP_ALLGATHER.get.return_value = True
            with self.assertRaisesRegex(ValueError, "AllGather requires the ACL Graph"):
                self.check(self.valid)

    def test_direct_audit_allows_multiple_scopes_but_not_acl_or_missing_rank(self):
        audit = load_benchmark("audit").audit
        lines = list(direct_log())
        with tempfile.TemporaryDirectory() as work:
            path = Path(work) / "server.log"
            path.write_text("\n".join(lines))
            self.assertEqual(
                audit(path, direct=True)["topology_by_rank"][0]["scopes"], 12
            )
            path.write_text("\n".join(lines[:-1]))
            with self.assertRaisesRegex(RuntimeError, "Incomplete MK direct"):
                audit(path, direct=True)
            path.write_text(
                "\n".join(
                    lines
                    + [
                        "[DP0 TP0 EP0] WeLM fixed ACL graph captured: scopes=12 steps=48"
                    ]
                )
            )
            with self.assertRaisesRegex(RuntimeError, "Unexpected ACL Graph"):
                audit(path, direct=True)

    def test_acl_control_records_observed_topology(self):
        audit = load_benchmark("audit").audit
        lines = [
            line.replace(
                "WeLM MK direct replay: scopes=12 steps=48 acl_graph=False",
                "WeLM fixed ACL graph captured: scopes=14 steps=80",
            ).replace("backend=mk-direct", "backend=acl")
            for line in direct_log()
        ]
        with tempfile.TemporaryDirectory() as work:
            path = Path(work) / "server.log"
            path.write_text("\n".join(lines))
            result = audit(path)
            self.assertEqual(result["backend"], "acl")
            self.assertEqual(result["topology_by_rank"][3], {"scopes": 14, "steps": 80})
