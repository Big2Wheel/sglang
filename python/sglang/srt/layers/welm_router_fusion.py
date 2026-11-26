"""Keep the DP8 Router fusion scoped to the current MegaKernel capture."""

from contextlib import contextmanager
from contextvars import ContextVar

_capture_bucket = ContextVar("welm_router_fusion_bucket", default=None)


def router_capture_bucket():
    return _capture_bucket.get()


@contextmanager
def router_capture_scope(bucket):
    token = _capture_bucket.set(bucket)
    try:
        yield
    finally:
        _capture_bucket.reset(token)
