"""Process-level locking to prevent multiple Enyal instances on the same database."""

import contextlib
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


class ProcessLock:
    """Prevent multiple Enyal server instances from running against the same database.

    Uses OS-level file locking (flock on Unix, msvcrt on Windows) to ensure
    only one server process operates on a given database at a time. The lock
    is automatically released by the OS when the process exits, even on crash.

    The lock file is placed next to the database file as ``.enyal.lock``.
    """

    def __init__(self, db_path: Path):
        self.lock_path = db_path.parent / ".enyal.lock"
        self._lock_file: object | None = None
        self._locked: bool = False

    def acquire(self) -> bool:
        """Try to acquire the process lock.

        Returns:
            True if the lock was acquired, False if another instance is running.
        """
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self._lock_file = open(self.lock_path, "w")  # noqa: SIM115
        except OSError as e:
            logger.error(f"Cannot create lock file {self.lock_path}: {e}")
            return False

        if sys.platform == "win32":
            return self._acquire_windows()
        else:
            return self._acquire_unix()

    def _acquire_unix(self) -> bool:
        """Acquire lock using fcntl.flock()."""
        import fcntl

        try:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)  # type: ignore[union-attr]
            self._write_pid()
            self._locked = True
            logger.info(f"Process lock acquired (PID {os.getpid()})")
            return True
        except OSError:
            existing_pid = self._read_existing_pid()
            logger.warning(
                f"Another Enyal instance (PID {existing_pid}) is running "
                f"against the same database. Lock file: {self.lock_path}"
            )
            self._lock_file.close()  # type: ignore[union-attr]
            self._lock_file = None
            return False

    def _acquire_windows(self) -> bool:
        """Acquire lock using msvcrt.locking()."""
        import msvcrt

        try:
            msvcrt.locking(self._lock_file.fileno(), msvcrt.LK_NBLCK, 1)  # type: ignore[union-attr]
            self._write_pid()
            self._locked = True
            logger.info(f"Process lock acquired (PID {os.getpid()})")
            return True
        except OSError:
            existing_pid = self._read_existing_pid()
            logger.warning(
                f"Another Enyal instance (PID {existing_pid}) is running "
                f"against the same database. Lock file: {self.lock_path}"
            )
            self._lock_file.close()  # type: ignore[union-attr]
            self._lock_file = None
            return False

    def _write_pid(self) -> None:
        """Write current PID to lock file."""
        self._lock_file.seek(0)  # type: ignore[union-attr]
        self._lock_file.truncate()  # type: ignore[union-attr]
        self._lock_file.write(str(os.getpid()))  # type: ignore[union-attr]
        self._lock_file.flush()  # type: ignore[union-attr]

    def _read_existing_pid(self) -> str:
        """Read PID from existing lock file."""
        try:
            with open(self.lock_path) as f:
                return f.read().strip() or "unknown"
        except Exception:
            return "unknown"

    def release(self) -> None:
        """Release the process lock."""
        if not self._lock_file or not self._locked:
            return

        try:
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(self._lock_file.fileno(), msvcrt.LK_UNLCK, 1)  # type: ignore[union-attr]
            else:
                import fcntl

                fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)  # type: ignore[union-attr]
        except Exception:
            pass

        with contextlib.suppress(Exception):
            self._lock_file.close()  # type: ignore[union-attr]
        self._lock_file = None
        self._locked = False

        with contextlib.suppress(OSError):
            self.lock_path.unlink(missing_ok=True)

        logger.info("Process lock released")

    @property
    def is_locked(self) -> bool:
        """Whether this instance currently holds the lock."""
        return self._locked

    def __del__(self) -> None:
        self.release()
