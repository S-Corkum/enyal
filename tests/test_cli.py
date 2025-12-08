"""Tests for CLI module."""

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from enyal.cli.main import (
    cmd_forget,
    cmd_get,
    cmd_recall,
    cmd_remember,
    cmd_stats,
    get_store,
    main,
)
from enyal.models.context import (
    ContextEntry,
    ContextSearchResult,
    ContextStats,
    ContextType,
    ScopeLevel,
)


class TestGetStore:
    """Tests for get_store function."""

    def test_get_store_default_path(self) -> None:
        """Test get_store with default path."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("enyal.cli.main.ContextStore") as mock_store_class:
                mock_store_class.return_value = MagicMock()
                store = get_store()
                mock_store_class.assert_called_once_with("~/.enyal/context.db")

    def test_get_store_custom_path(self) -> None:
        """Test get_store with custom path."""
        with patch("enyal.cli.main.ContextStore") as mock_store_class:
            mock_store_class.return_value = MagicMock()
            store = get_store("/custom/path/to/db")
            mock_store_class.assert_called_once_with("/custom/path/to/db")

    def test_get_store_env_var_path(self) -> None:
        """Test get_store uses environment variable."""
        with patch.dict(os.environ, {"ENYAL_DB_PATH": "/env/path/db"}, clear=True):
            with patch("enyal.cli.main.ContextStore") as mock_store_class:
                mock_store_class.return_value = MagicMock()
                store = get_store()
                mock_store_class.assert_called_once_with("/env/path/db")


class TestCmdRemember:
    """Tests for cmd_remember function."""

    def test_cmd_remember_basic(self, sample_entry: ContextEntry) -> None:
        """Test basic remember command."""
        args = argparse.Namespace(
            content="Test content",
            type="fact",
            scope="project",
            scope_path=None,
            tags=None,
            db=None,
            json=False,
        )

        with patch("enyal.cli.main.get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.remember.return_value = "test-id-123"
            mock_get_store.return_value = mock_store

            result = cmd_remember(args)

            assert result == 0
            mock_store.remember.assert_called_once_with(
                content="Test content",
                content_type=ContextType.FACT,
                scope_level=ScopeLevel.PROJECT,
                scope_path=None,
                tags=[],
            )

    def test_cmd_remember_with_tags(self) -> None:
        """Test remember command with tags."""
        args = argparse.Namespace(
            content="Test content",
            type="decision",
            scope="file",
            scope_path="/path/to/file.py",
            tags="tag1,tag2,tag3",
            db=None,
            json=False,
        )

        with patch("enyal.cli.main.get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.remember.return_value = "test-id-123"
            mock_get_store.return_value = mock_store

            result = cmd_remember(args)

            assert result == 0
            mock_store.remember.assert_called_once_with(
                content="Test content",
                content_type=ContextType.DECISION,
                scope_level=ScopeLevel.FILE,
                scope_path="/path/to/file.py",
                tags=["tag1", "tag2", "tag3"],
            )

    def test_cmd_remember_json_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test remember command with JSON output."""
        args = argparse.Namespace(
            content="Test content",
            type="fact",
            scope="project",
            scope_path=None,
            tags=None,
            db=None,
            json=True,
        )

        with patch("enyal.cli.main.get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.remember.return_value = "test-id-123"
            mock_get_store.return_value = mock_store

            result = cmd_remember(args)

            assert result == 0
            captured = capsys.readouterr()
            output = json.loads(captured.out)
            assert output["success"] is True
            assert output["entry_id"] == "test-id-123"


class TestCmdRecall:
    """Tests for cmd_recall function."""

    def test_cmd_recall_with_results(
        self, sample_entry: ContextEntry, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test recall command with results."""
        args = argparse.Namespace(
            query="test query",
            limit=10,
            type=None,
            scope=None,
            scope_path=None,
            min_confidence=0.3,
            db=None,
            json=False,
        )

        mock_result = ContextSearchResult(entry=sample_entry, distance=0.25, score=0.8)

        with patch("enyal.cli.main.get_store") as mock_get_store:
            with patch("enyal.cli.main.RetrievalEngine") as mock_retrieval_class:
                mock_retrieval = MagicMock()
                mock_retrieval.search.return_value = [mock_result]
                mock_retrieval_class.return_value = mock_retrieval

                result = cmd_recall(args)

                assert result == 0
                captured = capsys.readouterr()
                assert "Test content" in captured.out

    def test_cmd_recall_no_results(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test recall command with no results."""
        args = argparse.Namespace(
            query="nonexistent query",
            limit=10,
            type=None,
            scope=None,
            scope_path=None,
            min_confidence=0.3,
            db=None,
            json=False,
        )

        with patch("enyal.cli.main.get_store"):
            with patch("enyal.cli.main.RetrievalEngine") as mock_retrieval_class:
                mock_retrieval = MagicMock()
                mock_retrieval.search.return_value = []
                mock_retrieval_class.return_value = mock_retrieval

                result = cmd_recall(args)

                assert result == 0
                captured = capsys.readouterr()
                assert "No results found" in captured.out

    def test_cmd_recall_json_output(
        self, sample_entry: ContextEntry, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test recall command with JSON output."""
        args = argparse.Namespace(
            query="test query",
            limit=10,
            type=None,
            scope=None,
            scope_path=None,
            min_confidence=0.3,
            db=None,
            json=True,
        )

        mock_result = ContextSearchResult(entry=sample_entry, distance=0.25, score=0.8)

        with patch("enyal.cli.main.get_store"):
            with patch("enyal.cli.main.RetrievalEngine") as mock_retrieval_class:
                mock_retrieval = MagicMock()
                mock_retrieval.search.return_value = [mock_result]
                mock_retrieval_class.return_value = mock_retrieval

                result = cmd_recall(args)

                assert result == 0
                captured = capsys.readouterr()
                output = json.loads(captured.out)
                assert isinstance(output, list)
                assert len(output) == 1
                assert output[0]["content"] == "Test content for unit tests"

    def test_cmd_recall_with_filters(
        self, sample_entry: ContextEntry, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test recall command with scope and type filters."""
        args = argparse.Namespace(
            query="test query",
            limit=5,
            type="fact",
            scope="project",
            scope_path="/test/path",
            min_confidence=0.5,
            db=None,
            json=False,
        )

        mock_result = ContextSearchResult(entry=sample_entry, distance=0.25, score=0.8)

        with patch("enyal.cli.main.get_store"):
            with patch("enyal.cli.main.RetrievalEngine") as mock_retrieval_class:
                mock_retrieval = MagicMock()
                mock_retrieval.search.return_value = [mock_result]
                mock_retrieval_class.return_value = mock_retrieval

                result = cmd_recall(args)

                assert result == 0
                mock_retrieval.search.assert_called_once_with(
                    query="test query",
                    limit=5,
                    scope_level=ScopeLevel.PROJECT,
                    scope_path="/test/path",
                    content_type=ContextType.FACT,
                    min_confidence=0.5,
                )


class TestCmdForget:
    """Tests for cmd_forget function."""

    def test_cmd_forget_success(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test successful forget command (soft delete)."""
        args = argparse.Namespace(
            entry_id="test-entry-id",
            hard=False,
            db=None,
            json=False,
        )

        with patch("enyal.cli.main.get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.forget.return_value = True
            mock_get_store.return_value = mock_store

            result = cmd_forget(args)

            assert result == 0
            captured = capsys.readouterr()
            assert "deprecated" in captured.out
            mock_store.forget.assert_called_once_with("test-entry-id", hard_delete=False)

    def test_cmd_forget_hard_delete(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test forget command with hard delete."""
        args = argparse.Namespace(
            entry_id="test-entry-id",
            hard=True,
            db=None,
            json=False,
        )

        with patch("enyal.cli.main.get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.forget.return_value = True
            mock_get_store.return_value = mock_store

            result = cmd_forget(args)

            assert result == 0
            captured = capsys.readouterr()
            assert "permanently deleted" in captured.out
            mock_store.forget.assert_called_once_with("test-entry-id", hard_delete=True)

    def test_cmd_forget_not_found(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test forget command when entry not found."""
        args = argparse.Namespace(
            entry_id="nonexistent-id",
            hard=False,
            db=None,
            json=False,
        )

        with patch("enyal.cli.main.get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.forget.return_value = False
            mock_get_store.return_value = mock_store

            result = cmd_forget(args)

            assert result == 1
            captured = capsys.readouterr()
            assert "not found" in captured.out

    def test_cmd_forget_json_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test forget command with JSON output."""
        args = argparse.Namespace(
            entry_id="test-entry-id",
            hard=False,
            db=None,
            json=True,
        )

        with patch("enyal.cli.main.get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.forget.return_value = True
            mock_get_store.return_value = mock_store

            result = cmd_forget(args)

            assert result == 0
            captured = capsys.readouterr()
            output = json.loads(captured.out)
            assert output["success"] is True


class TestCmdGet:
    """Tests for cmd_get function."""

    def test_cmd_get_success(
        self, sample_entry: ContextEntry, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test successful get command."""
        args = argparse.Namespace(
            entry_id="test-entry-id",
            db=None,
            json=False,
        )

        with patch("enyal.cli.main.get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.get.return_value = sample_entry
            mock_get_store.return_value = mock_store

            result = cmd_get(args)

            assert result == 0
            captured = capsys.readouterr()
            assert "Test content" in captured.out
            assert "fact" in captured.out.lower()

    def test_cmd_get_not_found(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test get command when entry not found."""
        args = argparse.Namespace(
            entry_id="nonexistent-id",
            db=None,
            json=False,
        )

        with patch("enyal.cli.main.get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.get.return_value = None
            mock_get_store.return_value = mock_store

            result = cmd_get(args)

            assert result == 1
            captured = capsys.readouterr()
            assert "not found" in captured.out

    def test_cmd_get_json_output(
        self, sample_entry: ContextEntry, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test get command with JSON output."""
        args = argparse.Namespace(
            entry_id="test-entry-id",
            db=None,
            json=True,
        )

        with patch("enyal.cli.main.get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.get.return_value = sample_entry
            mock_get_store.return_value = mock_store

            result = cmd_get(args)

            assert result == 0
            captured = capsys.readouterr()
            output = json.loads(captured.out)
            assert output["content"] == "Test content for unit tests"
            assert output["type"] == "fact"

    def test_cmd_get_json_not_found(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test get command with JSON output when entry not found."""
        args = argparse.Namespace(
            entry_id="nonexistent-id",
            db=None,
            json=True,
        )

        with patch("enyal.cli.main.get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.get.return_value = None
            mock_get_store.return_value = mock_store

            result = cmd_get(args)

            assert result == 1
            captured = capsys.readouterr()
            output = json.loads(captured.out)
            assert "error" in output


class TestCmdStats:
    """Tests for cmd_stats function."""

    def test_cmd_stats_basic(
        self, sample_stats: ContextStats, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test basic stats command."""
        args = argparse.Namespace(
            db=None,
            json=False,
        )

        with patch("enyal.cli.main.get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.stats.return_value = sample_stats
            mock_get_store.return_value = mock_store

            result = cmd_stats(args)

            assert result == 0
            captured = capsys.readouterr()
            assert "Total entries:" in captured.out
            assert "100" in captured.out

    def test_cmd_stats_json_output(
        self, sample_stats: ContextStats, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test stats command with JSON output."""
        args = argparse.Namespace(
            db=None,
            json=True,
        )

        with patch("enyal.cli.main.get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.stats.return_value = sample_stats
            mock_get_store.return_value = mock_store

            result = cmd_stats(args)

            assert result == 0
            captured = capsys.readouterr()
            output = json.loads(captured.out)
            assert output["total_entries"] == 100
            assert output["active_entries"] == 90
            assert output["deprecated_entries"] == 10

    def test_cmd_stats_with_entries_by_type(
        self, sample_stats: ContextStats, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test stats command displays entries by type."""
        args = argparse.Namespace(
            db=None,
            json=False,
        )

        with patch("enyal.cli.main.get_store") as mock_get_store:
            mock_store = MagicMock()
            mock_store.stats.return_value = sample_stats
            mock_get_store.return_value = mock_store

            result = cmd_stats(args)

            assert result == 0
            captured = capsys.readouterr()
            assert "By type:" in captured.out
            assert "fact" in captured.out


class TestMainEntrypoint:
    """Tests for main CLI entry point."""

    def test_main_remember_command(self) -> None:
        """Test main function with remember command."""
        test_args = ["remember", "Test content"]

        with patch("sys.argv", ["enyal", *test_args]):
            with patch("enyal.cli.main.get_store") as mock_get_store:
                mock_store = MagicMock()
                mock_store.remember.return_value = "test-id"
                mock_get_store.return_value = mock_store

                result = main()

                assert result == 0
                mock_store.remember.assert_called_once()

    def test_main_recall_command(self) -> None:
        """Test main function with recall command."""
        test_args = ["recall", "test query"]

        with patch("sys.argv", ["enyal", *test_args]):
            with patch("enyal.cli.main.get_store"):
                with patch("enyal.cli.main.RetrievalEngine") as mock_retrieval_class:
                    mock_retrieval = MagicMock()
                    mock_retrieval.search.return_value = []
                    mock_retrieval_class.return_value = mock_retrieval

                    result = main()

                    assert result == 0

    def test_main_no_command(self) -> None:
        """Test main function with no command raises error."""
        with patch("sys.argv", ["enyal"]):
            with pytest.raises(SystemExit):
                main()

    def test_main_stats_command(self, sample_stats: ContextStats) -> None:
        """Test main function with stats command."""
        test_args = ["stats"]

        with patch("sys.argv", ["enyal", *test_args]):
            with patch("enyal.cli.main.get_store") as mock_get_store:
                mock_store = MagicMock()
                mock_store.stats.return_value = sample_stats
                mock_get_store.return_value = mock_store

                result = main()

                assert result == 0

    def test_main_get_command(self, sample_entry: ContextEntry) -> None:
        """Test main function with get command."""
        test_args = ["get", "test-entry-id"]

        with patch("sys.argv", ["enyal", *test_args]):
            with patch("enyal.cli.main.get_store") as mock_get_store:
                mock_store = MagicMock()
                mock_store.get.return_value = sample_entry
                mock_get_store.return_value = mock_store

                result = main()

                assert result == 0

    def test_main_forget_command(self) -> None:
        """Test main function with forget command."""
        test_args = ["forget", "test-entry-id"]

        with patch("sys.argv", ["enyal", *test_args]):
            with patch("enyal.cli.main.get_store") as mock_get_store:
                mock_store = MagicMock()
                mock_store.forget.return_value = True
                mock_get_store.return_value = mock_store

                result = main()

                assert result == 0

    def test_main_with_global_db_flag(self, sample_stats: ContextStats) -> None:
        """Test main function with --db flag."""
        test_args = ["--db", "/custom/path.db", "stats"]

        with patch("sys.argv", ["enyal", *test_args]):
            with patch("enyal.cli.main.ContextStore") as mock_store_class:
                mock_store = MagicMock()
                mock_store.stats.return_value = sample_stats
                mock_store_class.return_value = mock_store

                result = main()

                assert result == 0
                mock_store_class.assert_called_with("/custom/path.db")

    def test_main_with_json_flag(self, sample_stats: ContextStats) -> None:
        """Test main function with --json flag."""
        test_args = ["--json", "stats"]

        with patch("sys.argv", ["enyal", *test_args]):
            with patch("enyal.cli.main.get_store") as mock_get_store:
                mock_store = MagicMock()
                mock_store.stats.return_value = sample_stats
                mock_get_store.return_value = mock_store

                result = main()

                assert result == 0
