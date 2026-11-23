"""Test WeLM direct replay dispatch without TorchNPU."""

import ast
import logging
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

BACKEND = (
    Path(__file__).resolve().parents[5]
    / "python/sglang/srt/hardware_backend/npu/graph_runner/welm_mk_decode_backend.py"
)


class ACLBackend:
    @contextmanager
    def capture_session(self, stream):
        self.create_pool(stream)
        yield


def replay_backend():
    # Load the real backend methods but avoid importing the NPU stack.
    tree = ast.parse(BACKEND.read_bytes())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    cls.body = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"capture_session", "replay"}
    ]
    namespace = dict(
        NPUCudaGraphBackend=ACLBackend,
        contextmanager=contextmanager,
        logger=logging.getLogger(__name__),
        profile_range=lambda _: nullcontext(),
    )
    exec(
        compile(ast.Module(body=[cls], type_ignores=[]), str(BACKEND), "exec"),
        namespace,
    )
    return namespace[cls.name]()


class TestDirectReplay(unittest.TestCase):
    def setUp(self):
        self.backend = replay_backend()
        self.key = "bs1"
        self.batch = object()
        self.output = object()
        self.trace = []
        self.metadata = SimpleNamespace(
            welm_mk_schedule_recipes={"flash": ((), {})},
            welm_flash_schedules={
                "flash": SimpleNamespace(
                    copy_=lambda value, **kw: self.trace.append(value)
                )
            },
        )
        self.graph = SimpleNamespace(
            replay=Mock(
                side_effect=lambda **kw: self.launch("acl", kw["before_acl_capture"])
            ),
            replay_direct=Mock(
                side_effect=lambda **kw: self.launch("mk-direct", kw["before_launch"])
            ),
        )
        self.backend._direct_replay = True
        self.backend.create_pool = Mock()
        self.backend._graphs = {self.key: self.graph}
        self.backend._outputs = {self.key: self.output}
        self.backend._metadata = {self.key: self.metadata}
        self.backend._page_prologue_batches = {self.key: self.batch}
        self.backend._replays = {self.key: 0}
        self.backend._attn_backend = SimpleNamespace(
            forward_metadata=self.metadata,
            _prepare_welm_decode_metadata_device=lambda _: self.trace.append("pages"),
            _welm_flash_attn_metadata=Mock(
                side_effect=["schedule-128", "schedule-129"]
            ),
        )

    def launch(self, mode, update):
        update()
        self.trace.append(mode)

    def test_direct_capture_does_not_create_an_acl_graph_pool(self):
        stream = object()
        with self.assertRaisesRegex(ValueError, "capture error"):
            with self.backend.capture_session(stream):
                self.assertIs(self.backend._capture_stream, stream)
                raise ValueError("capture error")
        self.assertIsNone(self.backend._capture_stream)
        self.backend.create_pool.assert_not_called()

    def test_repeated_direct_replay_updates_metadata_then_submits(self):
        for _ in range(2):
            self.assertIs(self.backend.replay(self.key, self.batch), self.output)
        self.assertEqual(
            self.trace,
            [
                "pages",
                "schedule-128",
                "mk-direct",
                "pages",
                "schedule-129",
                "mk-direct",
            ],
        )
        self.graph.replay.assert_not_called()

    def test_metadata_failure_is_not_counted_or_sent_to_acl(self):
        self.backend._attn_backend._welm_flash_attn_metadata.side_effect = ValueError(
            "bad metadata"
        )
        with self.assertRaisesRegex(ValueError, "bad metadata"):
            self.backend.replay(self.key, self.batch)
        self.assertEqual(self.trace, ["pages"])
        self.assertEqual(self.backend._replays[self.key], 0)
        self.graph.replay.assert_not_called()

    def test_acl_path_still_uses_acl_pool_and_callback(self):
        self.backend._direct_replay = False
        stream = object()
        with self.backend.capture_session(stream):
            pass
        self.backend.create_pool.assert_called_once_with(stream)
        self.assertIs(self.backend.replay(self.key, self.batch), self.output)
        self.assertEqual(self.trace, ["pages", "schedule-128", "acl"])
        self.graph.replay_direct.assert_not_called()


if __name__ == "__main__":
    unittest.main()
