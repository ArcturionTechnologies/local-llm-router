"""Single-heavy-model lock.

Unified memory is the scarce resource on Apple Silicon: two big models
resident at once and the machine swaps itself to a standstill. This module
serialises "heavy" model use with a kernel-level ``flock`` plus a small
metadata file saying who holds it.

Two ways to use it:

* **Router side** -- tiers configured with ``exclusive = true`` are wrapped in
  :meth:`ModelLock.hold` for the duration of each call. If another process
  holds the lock the tier is skipped (``TierBusy``) and the router falls
  through to the next tier.
* **Server side** -- a launcher or wrapper for ``mlx_lm.server`` can call
  :meth:`ModelLock.acquire` before loading weights and :meth:`release` on
  idle-unload, and answer 503 ``{"error": "local-pool-busy", ...}`` when it
  cannot acquire.

A crashed holder cannot wedge the lock: ``flock`` is released by the kernel
when the process dies, and metadata whose PID is dead is ignored.

    python -m local_llm_router lock status
"""

from __future__ import annotations

import fcntl
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from .errors import TierBusy


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:      # exists, owned by someone else
        return True
    except OSError:
        return False
    return True


class ModelLock:
    def __init__(self, directory: os.PathLike):
        self.dir = Path(directory).expanduser()
        self.lock_path = self.dir / "local-model.lock"
        self.meta_path = self.dir / "local-model.meta.json"
        self._fd: Optional[int] = None
        self._held_as: Optional[str] = None

    # -- metadata -----------------------------------------------------------
    def _write_meta(self, server: str, pid: int) -> None:
        self.meta_path.write_text(json.dumps({
            "server": server,
            "pid": pid,
            "acquired_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, indent=2))

    def _read_meta(self) -> Optional[dict]:
        try:
            return json.loads(self.meta_path.read_text())
        except (OSError, ValueError):
            return None

    def _unlink_meta(self) -> None:
        try:
            self.meta_path.unlink()
        except FileNotFoundError:
            pass

    # -- API ----------------------------------------------------------------
    def acquire(self, server: str, pid: Optional[int] = None) -> bool:
        """Try to take the lock without blocking. ``True`` = you hold it.

        If the lock is held but the recorded holder PID is dead, the metadata
        is reaped and acquisition is retried once.
        """
        self.dir.mkdir(parents=True, exist_ok=True)
        pid = pid or os.getpid()
        if self._fd is not None:          # this object already holds it
            return self._held_as == server
        for attempt in (0, 1):
            fd = os.open(str(self.lock_path), os.O_RDWR | os.O_CREAT, 0o644)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(fd)
                meta = self._read_meta()
                if attempt == 0 and meta and not _pid_alive(int(meta.get("pid", -1))):
                    self._unlink_meta()    # stale holder: reap and retry once
                    continue
                return False
            self._fd, self._held_as = fd, server
            self._write_meta(server, pid)
            return True
        return False

    def release(self, server: str) -> None:
        """Release if this object holds the lock. Idempotent."""
        fd, self._fd, self._held_as = self._fd, None, None
        if fd is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
        meta = self._read_meta()
        if meta and meta.get("server") == server:
            self._unlink_meta()

    def holder(self) -> dict:
        """Metadata of the live holder, or ``{}`` when the lock is free."""
        meta = self._read_meta()
        if not meta or not _pid_alive(int(meta.get("pid", -1))):
            return {}
        return meta

    def is_held_by_other(self, server: str) -> bool:
        h = self.holder()
        return bool(h) and h.get("server") != server

    def force_release(self) -> None:
        """Emergency: delete the metadata and lock file (the kernel flock of a
        live foreign holder is not affected)."""
        self._unlink_meta()
        try:
            self.lock_path.unlink()
        except FileNotFoundError:
            pass

    @contextmanager
    def hold(self, server: str) -> Iterator[None]:
        """Hold the lock for a ``with`` block; raise :class:`TierBusy` if it is taken."""
        if not self.acquire(server):
            h = self.holder()
            raise TierBusy(
                f"local-pool-busy -- held by {h.get('server', 'another process')} "
                f"(pid {h.get('pid')}); fall through",
                holder=h.get("server"), holder_pid=h.get("pid"))
        try:
            yield
        finally:
            self.release(server)
