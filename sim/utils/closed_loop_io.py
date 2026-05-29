from __future__ import annotations

import errno
import os
import pickle
import select
import time
from typing import Any


class AdCommunicationError(RuntimeError):
    pass


def write_fifo_with_ad_monitor(
    fifo_path: str,
    payload: Any,
    *,
    process=None,
    timeout_seconds: float = 300.0,
    poll_interval_seconds: float = 0.2,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    data = pickle.dumps(payload)

    fd = None
    while fd is None:
        _ensure_process_alive(process)
        remaining = _remaining_time(deadline, fifo_path, "open for writing")
        try:
            fd = os.open(fifo_path, os.O_WRONLY | os.O_NONBLOCK)
        except OSError as exc:
            if exc.errno != errno.ENXIO:
                raise
            time.sleep(min(poll_interval_seconds, remaining))

    try:
        view = memoryview(data)
        while view:
            _ensure_process_alive(process)
            remaining = _remaining_time(deadline, fifo_path, "write")
            _, writable, _ = select.select([], [fd], [], min(poll_interval_seconds, remaining))
            if not writable:
                continue
            try:
                written = os.write(fd, view)
            except BlockingIOError:
                continue
            except BrokenPipeError as exc:
                raise AdCommunicationError(f"Writer peer disappeared while writing to fifo: {fifo_path}") from exc
            view = view[written:]
    finally:
        os.close(fd)


def read_fifo_with_ad_monitor(
    fifo_path: str,
    *,
    process=None,
    timeout_seconds: float = 300.0,
    poll_interval_seconds: float = 0.2,
):
    deadline = time.monotonic() + timeout_seconds
    fd = os.open(fifo_path, os.O_RDONLY | os.O_NONBLOCK)

    try:
        chunks: list[bytes] = []
        while True:
            _ensure_process_alive(process)
            remaining = _remaining_time(deadline, fifo_path, "read")
            readable, _, _ = select.select([fd], [], [], min(poll_interval_seconds, remaining))
            if not readable:
                continue
            try:
                chunk = os.read(fd, 1024 * 1024)
            except BlockingIOError:
                continue

            if chunk:
                chunks.append(chunk)
                continue

            if chunks:
                return pickle.loads(b"".join(chunks))

            time.sleep(min(poll_interval_seconds, remaining))
    finally:
        os.close(fd)


def _ensure_process_alive(process) -> None:
    if process is None:
        return
    return_code = process.poll()
    if return_code is None:
        return
    raise AdCommunicationError(f"AD process exited with return code {return_code}.")


def _remaining_time(deadline: float, fifo_path: str, action: str) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise AdCommunicationError(f"Timed out waiting to {action} fifo: {fifo_path}")
    return remaining
