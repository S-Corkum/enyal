"""Tests for CLI module."""

import argparse
import json
import os
from unittest.mock import MagicMock, patch

import pytest

from enyal.cli.main import (
    cmd_forget,
    cmd_get,
    cmd_model_download,
    cmd_model_status,
    cmd_model_verify,
    cmd_recall,
    cmd_remember,
    cmd_serve,
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
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("enyal.cli.main.ContextStore") as mock_store_class,
            patch("enyal.cli.main.EmbeddingEngine"),
            patch("enyal.cli.main.ModelConfig"),
        ):
            mock_store_class.return_value = MagicMock()
            get_store()
            assert mock_store_class.call_args[0][0] == "~/.enyal/context.db"

    def test_get_store_custom_path(self) -> None:
        """Test get_store with custom path."""
        with (
            patch("enyal.cli.main.ContextStore") as mock_store_class,
            patch("enyal.cli.main.EmbeddingEngine"),
            patch("enyal.cli.main.ModelConfig"),
        ):
            mock_store_class.return_value = MagicMock()
            get_store("/custom/path/to/db")
            assert mock_store_class.call_args[0][0] == "/custom/path/to/db"

    def test_get_store_env_var_path(self) -> None:
        """Test get_store uses environment variable."""
        with (
            patch.dict(os.environ, {"ENYAL_DB_PATH": "/env/path/db"}, clear=True),
            patch("enyal.cli.main.ContextStore") as mock_store_class,
            patch("enyal.cli.main.EmbeddingEngine"),
            patch("enyal.cli.main.ModelConfig"),
        ):
            mock_store_class.return_value = MagicMock()
            get_store()
            assert mock_store_class.call_args[0][0] == "/env/path/db"


class TestCmdRemember:
    """Tests for cmd_remember function."""

    def test_cmd_remember_basic(self) -> None:
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

        with (
            patch("enyal.cli.main.get_store"),
            patch("enyal.cli.main.RetrievalEngine") as mock_retrieval_class,
        ):
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

        with (
            patch("enyal.cli.main.get_store"),
            patch("enyal.cli.main.RetrievalEngine") as mock_retrieval_class,
        ):
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

        with (
            patch("enyal.cli.main.get_store"),
            patch("enyal.cli.main.RetrievalEngine") as mock_retrieval_class,
        ):
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

    def test_cmd_recall_with_filters(self, sample_entry: ContextEntry) -> None:
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

        with (
            patch("enyal.cli.main.get_store"),
            patch("enyal.cli.main.RetrievalEngine") as mock_retrieval_class,
        ):
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

        with (
            patch("sys.argv", ["enyal", *test_args]),
            patch("enyal.cli.main.get_store") as mock_get_store,
        ):
            mock_store = MagicMock()
            mock_store.remember.return_value = "test-id"
            mock_get_store.return_value = mock_store

            result = main()

            assert result == 0
            mock_store.remember.assert_called_once()

    def test_main_recall_command(self) -> None:
        """Test main function with recall command."""
        test_args = ["recall", "test query"]

        with (
            patch("sys.argv", ["enyal", *test_args]),
            patch("enyal.cli.main.get_store"),
            patch("enyal.cli.main.RetrievalEngine") as mock_retrieval_class,
        ):
            mock_retrieval = MagicMock()
            mock_retrieval.search.return_value = []
            mock_retrieval_class.return_value = mock_retrieval

            result = main()

            assert result == 0

    def test_main_no_command(self) -> None:
        """Test main function with no command raises error."""
        with patch("sys.argv", ["enyal"]), pytest.raises(SystemExit):
            main()

    def test_main_stats_command(self, sample_stats: ContextStats) -> None:
        """Test main function with stats command."""
        test_args = ["stats"]

        with (
            patch("sys.argv", ["enyal", *test_args]),
            patch("enyal.cli.main.get_store") as mock_get_store,
        ):
            mock_store = MagicMock()
            mock_store.stats.return_value = sample_stats
            mock_get_store.return_value = mock_store

            result = main()

            assert result == 0

    def test_main_get_command(self, sample_entry: ContextEntry) -> None:
        """Test main function with get command."""
        test_args = ["get", "test-entry-id"]

        with (
            patch("sys.argv", ["enyal", *test_args]),
            patch("enyal.cli.main.get_store") as mock_get_store,
        ):
            mock_store = MagicMock()
            mock_store.get.return_value = sample_entry
            mock_get_store.return_value = mock_store

            result = main()

            assert result == 0

    def test_main_forget_command(self) -> None:
        """Test main function with forget command."""
        test_args = ["forget", "test-entry-id"]

        with (
            patch("sys.argv", ["enyal", *test_args]),
            patch("enyal.cli.main.get_store") as mock_get_store,
        ):
            mock_store = MagicMock()
            mock_store.forget.return_value = True
            mock_get_store.return_value = mock_store

            result = main()

            assert result == 0

    def test_main_with_global_db_flag(self, sample_stats: ContextStats) -> None:
        """Test main function with --db flag."""
        test_args = ["--db", "/custom/path.db", "stats"]

        with (
            patch("sys.argv", ["enyal", *test_args]),
            patch("enyal.cli.main.ContextStore") as mock_store_class,
            patch("enyal.cli.main.EmbeddingEngine"),
            patch("enyal.cli.main.ModelConfig"),
        ):
            mock_store = MagicMock()
            mock_store.stats.return_value = sample_stats
            mock_store_class.return_value = mock_store

            result = main()

            assert result == 0
            assert mock_store_class.call_args[0][0] == "/custom/path.db"

    def test_main_with_json_flag(self, sample_stats: ContextStats) -> None:
        """Test main function with --json flag."""
        test_args = ["--json", "stats"]

        with (
            patch("sys.argv", ["enyal", *test_args]),
            patch("enyal.cli.main.get_store") as mock_get_store,
        ):
            mock_store = MagicMock()
            mock_store.stats.return_value = sample_stats
            mock_get_store.return_value = mock_store

            result = main()

            assert result == 0


class TestCmdServe:
    """Tests for cmd_serve function."""

    def test_cmd_serve_basic(self) -> None:
        """Test basic serve command."""
        args = argparse.Namespace(
            db=None,
            preload=False,
            log_level=None,
            json=False,
        )

        with patch("enyal.cli.main.main") as mock_server_main:
            # We need to patch the imported main from mcp.server
            with patch.dict("sys.modules", {"enyal.mcp.server": MagicMock()}):
                with patch("enyal.cli.main.main") as _:
                    # Patch the import inside cmd_serve
                    mock_module = MagicMock()
                    with patch.dict("sys.modules", {"enyal.mcp": MagicMock(), "enyal.mcp.server": mock_module}):
                        result = cmd_serve(args)

                        assert result == 0

    def test_cmd_serve_with_db(self) -> None:
        """Test serve command with custom db path."""
        args = argparse.Namespace(
            db="/custom/db/path.db",
            preload=False,
            log_level=None,
            json=False,
        )

        mock_module = MagicMock()
        with patch.dict("sys.modules", {"enyal.mcp": MagicMock(), "enyal.mcp.server": mock_module}):
            result = cmd_serve(args)

            assert result == 0
            assert os.environ.get("ENYAL_DB_PATH") == "/custom/db/path.db"

        # Cleanup
        os.environ.pop("ENYAL_DB_PATH", None)

    def test_cmd_serve_with_preload(self) -> None:
        """Test serve command with preload."""
        args = argparse.Namespace(
            db=None,
            preload=True,
            log_level=None,
            json=False,
        )

        mock_module = MagicMock()
        with patch.dict("sys.modules", {"enyal.mcp": MagicMock(), "enyal.mcp.server": mock_module}):
            result = cmd_serve(args)

            assert result == 0
            assert os.environ.get("ENYAL_PRELOAD_MODEL") == "true"

        # Cleanup
        os.environ.pop("ENYAL_PRELOAD_MODEL", None)

    def test_cmd_serve_with_log_level(self) -> None:
        """Test serve command with custom log level."""
        args = argparse.Namespace(
            db=None,
            preload=False,
            log_level="DEBUG",
            json=False,
        )

        mock_module = MagicMock()
        with patch.dict("sys.modules", {"enyal.mcp": MagicMock(), "enyal.mcp.server": mock_module}):
            result = cmd_serve(args)

            assert result == 0
            assert os.environ.get("ENYAL_LOG_LEVEL") == "DEBUG"

        # Cleanup
        os.environ.pop("ENYAL_LOG_LEVEL", None)


class TestCmdModelDownload:
    """Tests for cmd_model_download function."""

    def test_cmd_model_download_success(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test successful model download."""
        args = argparse.Namespace(
            model=None,
            cache_dir=None,
            db=None,
            json=False,
        )

        with patch("enyal.core.ssl_config.download_model") as mock_download:
            mock_download.return_value = "/path/to/model"

            result = cmd_model_download(args)

            assert result == 0
            captured = capsys.readouterr()
            assert "downloaded successfully" in captured.out

    def test_cmd_model_download_custom_model(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test model download with custom model name."""
        args = argparse.Namespace(
            model="custom-model-v2",
            cache_dir="/custom/cache",
            db=None,
            json=False,
        )

        with patch("enyal.core.ssl_config.download_model") as mock_download:
            mock_download.return_value = "/custom/cache/custom-model-v2"

            result = cmd_model_download(args)

            assert result == 0
            mock_download.assert_called_once_with("custom-model-v2", cache_dir="/custom/cache")

    def test_cmd_model_download_json_success(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test model download with JSON output."""
        args = argparse.Namespace(
            model=None,
            cache_dir=None,
            db=None,
            json=True,
        )

        with patch("enyal.core.ssl_config.download_model") as mock_download:
            mock_download.return_value = "/path/to/model"

            result = cmd_model_download(args)

            assert result == 0
            captured = capsys.readouterr()
            # cmd_model_download prints status lines before JSON; extract last line
            json_line = [l for l in captured.out.strip().split("\n") if l.startswith("{")][-1]
            output = json.loads(json_line)
            assert output["success"] is True

    def test_cmd_model_download_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test model download with error."""
        args = argparse.Namespace(
            model=None,
            cache_dir=None,
            db=None,
            json=False,
        )

        with patch("enyal.core.ssl_config.download_model") as mock_download:
            mock_download.side_effect = Exception("SSL error")

            result = cmd_model_download(args)

            assert result == 1
            captured = capsys.readouterr()
            assert "Error downloading model" in captured.out

    def test_cmd_model_download_error_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test model download error with JSON output."""
        args = argparse.Namespace(
            model=None,
            cache_dir=None,
            db=None,
            json=True,
        )

        with patch("enyal.core.ssl_config.download_model") as mock_download:
            mock_download.side_effect = Exception("Network error")

            result = cmd_model_download(args)

            assert result == 1
            captured = capsys.readouterr()
            json_line = [l for l in captured.out.strip().split("\n") if l.startswith("{")][-1]
            output = json.loads(json_line)
            assert output["success"] is False


class TestCmdModelVerify:
    """Tests for cmd_model_verify function."""

    def test_cmd_model_verify_success(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test successful model verification."""
        args = argparse.Namespace(
            model=None,
            db=None,
            json=False,
        )

        with patch("enyal.core.ssl_config.verify_model") as mock_verify:
            mock_verify.return_value = True

            result = cmd_model_verify(args)

            assert result == 0
            captured = capsys.readouterr()
            assert "verification successful" in captured.out

    def test_cmd_model_verify_failure(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test model verification failure."""
        args = argparse.Namespace(
            model="/bad/path",
            db=None,
            json=False,
        )

        with patch("enyal.core.ssl_config.verify_model") as mock_verify:
            mock_verify.return_value = False

            result = cmd_model_verify(args)

            assert result == 1
            captured = capsys.readouterr()
            assert "verification failed" in captured.out

    def test_cmd_model_verify_json_success(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test model verification with JSON output."""
        args = argparse.Namespace(
            model=None,
            db=None,
            json=True,
        )

        with patch("enyal.core.ssl_config.verify_model") as mock_verify:
            mock_verify.return_value = True

            result = cmd_model_verify(args)

            assert result == 0
            captured = capsys.readouterr()
            json_line = [l for l in captured.out.strip().split("\n") if l.startswith("{")][-1]
            output = json.loads(json_line)
            assert output["success"] is True

    def test_cmd_model_verify_json_failure(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test model verification failure with JSON output."""
        args = argparse.Namespace(
            model="some-model",
            db=None,
            json=True,
        )

        with patch("enyal.core.ssl_config.verify_model") as mock_verify:
            mock_verify.return_value = False

            result = cmd_model_verify(args)

            assert result == 1
            captured = capsys.readouterr()
            json_line = [l for l in captured.out.strip().split("\n") if l.startswith("{")][-1]
            output = json.loads(json_line)
            assert output["success"] is False


class TestCmdModelStatus:
    """Tests for cmd_model_status function."""

    def test_cmd_model_status_basic(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test basic model status output."""
        args = argparse.Namespace(
            db=None,
            json=False,
        )

        status = {
            "ssl_verify": True,
            "cert_file": None,
            "cert_file_exists": False,
            "model_path": None,
            "model_path_exists": False,
            "offline_mode": False,
            "hf_home": None,
            "system_ca_bundle": "/etc/ssl/cert.pem",
            "huggingface_hub_version": "0.20.0",
            "sentence_transformers_version": "2.7.0",
        }

        with patch("enyal.core.ssl_config.check_ssl_health") as mock_health:
            mock_health.return_value = status

            result = cmd_model_status(args)

            assert result == 0
            captured = capsys.readouterr()
            assert "SSL/Network Configuration" in captured.out
            assert "Enabled" in captured.out

    def test_cmd_model_status_with_cert(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test model status with cert file configured."""
        args = argparse.Namespace(
            db=None,
            json=False,
        )

        status = {
            "ssl_verify": True,
            "cert_file": "/path/to/cert.pem",
            "cert_file_exists": True,
            "model_path": "/path/to/model",
            "model_path_exists": True,
            "offline_mode": True,
            "hf_home": "/custom/hf",
            "system_ca_bundle": None,
            "huggingface_hub_version": None,
            "sentence_transformers_version": None,
        }

        with patch("enyal.core.ssl_config.check_ssl_health") as mock_health:
            mock_health.return_value = status

            result = cmd_model_status(args)

            assert result == 0
            captured = capsys.readouterr()
            assert "/path/to/cert.pem" in captured.out
            assert "Enabled" in captured.out  # offline mode

    def test_cmd_model_status_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test model status with JSON output."""
        args = argparse.Namespace(
            db=None,
            json=True,
        )

        status = {
            "ssl_verify": True,
            "cert_file": None,
            "offline_mode": False,
        }

        with patch("enyal.core.ssl_config.check_ssl_health") as mock_health:
            mock_health.return_value = status

            result = cmd_model_status(args)

            assert result == 0
            captured = capsys.readouterr()
            output = json.loads(captured.out)
            assert output["ssl_verify"] is True
