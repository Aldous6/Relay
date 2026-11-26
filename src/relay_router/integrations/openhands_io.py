"""Scoped retries for transient Windows failures in OpenHands persistence.

This patches one process-global SDK function while the context is active. Use
only for local conversations whose persistence belongs to ``root``; overlapping
contexts are rejected. Other SDK writers in the process are affected only when
their resolved destination is inside that root. This is not a model retry and
does not alter conversation contents, tools, generation limits or acceptance.
"""
from __future__ import annotations

from contextlib import contextmanager
import inspect
import math
from pathlib import Path
import threading
import time


_CONTEXT_LOCK = threading.RLock()
_ACTIVE = False


@contextmanager
def retry_openhands_writes(root, *, max_attempts=5, base_delay=.05, on_retry=None):
    """Retry exactly the failed atomic write, with a fresh SDK temporary file.

    Only ``PermissionError`` for destinations resolved inside ``root`` retries.
    ``max_attempts`` includes the initial write. Delays double from base_delay.
    ``on_retry`` receives only relative path, failed attempt number, next attempt,
    maximum attempts, errno and winerror; text contents and errors are omitted.
    Exhaustion preserves the last exception. No network/model function is called.
    """
    global _ACTIVE
    destination_root = Path(root).resolve()
    if destination_root.exists() and not destination_root.is_dir():
        raise ValueError("The persistence root must be a directory.")
    if type(max_attempts) is not int or not 1 <= max_attempts <= 20:
        raise ValueError("max_attempts must be an integer between 1 and 20.")
    if type(base_delay) not in (int, float) or not math.isfinite(base_delay) or not 0 <= base_delay <= 1:
        raise ValueError("base_delay must be finite and between 0 and 1 second.")
    if on_retry is not None and (not callable(on_retry) or inspect.iscoroutinefunction(on_retry)):
        raise TypeError("on_retry must be a synchronous callback or None.")
    if not _CONTEXT_LOCK.acquire(blocking=False):
        raise RuntimeError("OpenHands I/O retry contexts must not overlap.")
    if _ACTIVE:
        _CONTEXT_LOCK.release()
        raise RuntimeError("OpenHands I/O retry contexts must not overlap.")
    _ACTIVE = True
    sdk_io = original = None
    try:
        # Import only on use; the rest of Relay remains independent of the SDK.
        import openhands.sdk.io.local as sdk_io
        original = sdk_io.atomic_write_text

        def write(path, value, mode=0o600):
            resolved = Path(path).resolve()
            try:
                relative = resolved.relative_to(destination_root)
            except ValueError:
                return original(path, value, mode)
            for attempt in range(1, max_attempts + 1):
                try:
                    return original(path, value, mode)
                except PermissionError as exc:
                    if attempt == max_attempts:
                        raise
                    if on_retry is not None:
                        result = on_retry({"path": relative.as_posix(), "attempt": attempt,
                                           "next_attempt": attempt + 1, "max_attempts": max_attempts,
                                           "errno": exc.errno, "winerror": getattr(exc, "winerror", None)})
                        if inspect.isawaitable(result):
                            if inspect.iscoroutine(result):
                                result.close()
                            raise TypeError("on_retry must return synchronously.")
                    # The cap avoids unexpectedly long pauses for custom limits.
                    time.sleep(min(1., base_delay * 2 ** (attempt - 1)))

        sdk_io.atomic_write_text = write
        yield
    finally:
        if sdk_io is not None and original is not None:
            sdk_io.atomic_write_text = original
        _ACTIVE = False
        _CONTEXT_LOCK.release()
