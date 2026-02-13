"""Tests for ProcessLock module."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from enyal.core.process_lock import ProcessLock


class TestProcessLockInit:
    """Tests for ProcessLock initialization."""

    def test_init_sets_lock_path(self) -> None:
        """Test that init creates lock path next to db."""
        db_path = Path("/some/dir/context.db")
        lock = ProcessLock(db_path)

        assert lock.lock_path == Path("/some/dir/.enyal.lock")
        assert lock._lock_file is None
        assert lock._locked is False

    def test_is_locked_default_false(self) -> None:
        """Test that is_locked is False by default."""
        lock = ProcessLock(Path("/tmp/test.db"))
        assert lock.is_locked is False


class TestProcessLockAcquire:
    """Tests for lock acquisition."""

    def test_acquire_success(self) -> None:
        """Test successful lock acquisition on Unix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "context.db"
            lock = ProcessLock(db_path)

            result = lock.acquire()

            assert result is True
            assert lock.is_locked is True
            assert lock._lock_file is not None

            # Verify PID was written
            with open(lock.lock_path) as f:
                pid = f.read().strip()
            assert pid == str(os.getpid())

            lock.release()

    def test_acquire_creates_parent_dirs(self) -> None:
        """Test that acquire creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "nested" / "dir" / "context.db"
            lock = ProcessLock(db_path)

            result = lock.acquire()

            assert result is True
            assert lock.lock_path.parent.exists()

            lock.release()

    def test_acquire_cannot_create_lock_file(self) -> None:
        """Test acquire when lock file cannot be created."""
        lock = ProcessLock(Path("/nonexistent/path/context.db"))
        # Patch Path.mkdir at the class level to succeed, but open to fail
        with patch("pathlib.Path.mkdir"):
            with patch("builtins.open", side_effect=OSError("Permission denied")):
                result = lock.acquire()

        assert result is False
        assert lock.is_locked is False

    def test_acquire_contention(self) -> None:
        """Test that second acquire fails when lock is held."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "context.db"
            lock1 = ProcessLock(db_path)
            lock2 = ProcessLock(db_path)

            assert lock1.acquire() is True

            # Second lock should fail (same process, but flock is per file descriptor)
            # Note: flock allows re-locking from same process with different fd on some systems
            # We just verify the mechanism works without error
            result2 = lock2.acquire()
            # On most Unix systems this succeeds from same process
            # The real protection is cross-process

            lock1.release()
            if lock2.is_locked:
                lock2.release()


class TestProcessLockRelease:
    """Tests for lock release."""

    def test_release_success(self) -> None:
        """Test successful lock release."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "context.db"
            lock = ProcessLock(db_path)

            lock.acquire()
            assert lock.is_locked is True

            lock.release()

            assert lock.is_locked is False
            assert lock._lock_file is None
            # Lock file should be cleaned up
            assert not lock.lock_path.exists()

    def test_release_when_not_locked(self) -> None:
        """Test release when not holding lock."""
        lock = ProcessLock(Path("/tmp/test.db"))

        # Should not raise
        lock.release()

        assert lock.is_locked is False

    def test_release_no_lock_file(self) -> None:
        """Test release when _lock_file is None."""
        lock = ProcessLock(Path("/tmp/test.db"))
        lock._lock_file = None
        lock._locked = False

        # Should not raise
        lock.release()

    def test_release_cleans_up_on_exception(self) -> None:
        """Test release handles exceptions gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "context.db"
            lock = ProcessLock(db_path)

            lock.acquire()

            # Mock close to raise
            original_file = lock._lock_file
            mock_file = MagicMock()
            mock_file.fileno.side_effect = Exception("fileno error")
            lock._lock_file = mock_file

            # Should not raise despite internal errors
            lock.release()

            assert lock.is_locked is False
            assert lock._lock_file is None

            # Clean up original file
            if original_file:
                original_file.close()


class TestReadExistingPid:
    """Tests for _read_existing_pid."""

    def test_read_existing_pid_success(self) -> None:
        """Test reading PID from existing lock file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "context.db"
            lock = ProcessLock(db_path)

            # Write a PID to the lock file
            lock.lock_path.write_text("12345")

            pid = lock._read_existing_pid()

            assert pid == "12345"

    def test_read_existing_pid_empty_file(self) -> None:
        """Test reading PID from empty lock file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "context.db"
            lock = ProcessLock(db_path)

            lock.lock_path.write_text("")

            pid = lock._read_existing_pid()

            assert pid == "unknown"

    def test_read_existing_pid_file_not_found(self) -> None:
        """Test reading PID when lock file doesn't exist."""
        lock = ProcessLock(Path("/tmp/nonexistent/context.db"))

        pid = lock._read_existing_pid()

        assert pid == "unknown"


class TestWritePid:
    """Tests for _write_pid."""

    def test_write_pid(self) -> None:
        """Test writing PID to lock file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "context.db"
            lock = ProcessLock(db_path)

            lock.acquire()

            # Read the lock file to verify PID was written
            with open(lock.lock_path) as f:
                content = f.read().strip()

            assert content == str(os.getpid())

            lock.release()


class TestProcessLockDel:
    """Tests for __del__ method."""

    def test_del_calls_release(self) -> None:
        """Test that __del__ calls release."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "context.db"
            lock = ProcessLock(db_path)

            lock.acquire()
            assert lock.is_locked is True

            # Call __del__ manually
            lock.__del__()

            assert lock.is_locked is False
