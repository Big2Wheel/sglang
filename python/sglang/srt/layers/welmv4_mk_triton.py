"""Opt-in native child bindings for the WeLMV4 target graph."""

import os
from contextlib import contextmanager
from contextvars import ContextVar


_TARGET_CAPTURE = ContextVar("welmv4_mk_target_capture", default=False)


def in_target_native_capture() -> bool:
    return _TARGET_CAPTURE.get()


@contextmanager
def target_native_capture():
    token = _TARGET_CAPTURE.set(True)
    try:
        yield
    finally:
        _TARGET_CAPTURE.reset(token)


class _CaptureKernel:
    def __init__(self, ordinary, native):
        self.ordinary, self.native = ordinary, native

    def __getitem__(self, grid):
        target = self.native if _TARGET_CAPTURE.get() else self.ordinary
        return target[grid]

    def __getattr__(self, name):
        return getattr(self.ordinary, name)


def target_native(child):
    if (
        os.environ.get("WELM_NPU_USE_MK_TARGET_DECODE") == "1"
        and os.environ.get("WELM_MK_TRITON_CHILDREN") == "1"
    ):
        from megakernel.triton import native

        return lambda kernel: _CaptureKernel(kernel, native(child=child)(kernel))
    return lambda kernel: kernel
