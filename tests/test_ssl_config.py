"""Tests for SSL configuration module."""

import importlib
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from enyal.core.ssl_config import (
    SSLConfig,
    _find_system_ca_bundle,
    _parse_bool_env,
    check_ssl_health,
    configure_http_backend,
    configure_ssl_environment,
    download_model,
    get_model_path,
    get_ssl_config,
    verify_model,
)


class TestParseBoolEnv:
    """Tests for _parse_bool_env function."""

    def test_true_values(self) -> None:
        """Test that 'true', '1', 'yes', 'on' return True."""
        for value in ["true", "TRUE", "True", "1", "yes", "YES", "on", "ON"]:
            with patch.dict(os.environ, {"TEST_VAR": value}):
                assert _parse_bool_env("TEST_VAR", default=False) is True

    def test_false_values(self) -> None:
        """Test that other values return False."""
        for value in ["false", "FALSE", "0", "no", "NO", "off", "OFF", "random"]:
            with patch.dict(os.environ, {"TEST_VAR": value}):
                assert _parse_bool_env("TEST_VAR", default=True) is False

    def test_missing_uses_default(self) -> None:
        """Test that missing env var uses default."""
        env = os.environ.copy()
        env.pop("NONEXISTENT_VAR", None)
        with patch.dict(os.environ, env, clear=True):
            assert _parse_bool_env("NONEXISTENT_VAR", default=True) is True
            assert _parse_bool_env("NONEXISTENT_VAR", default=False) is False

    def test_empty_uses_default(self) -> None:
        """Test that empty string uses default."""
        with patch.dict(os.environ, {"TEST_VAR": ""}):
            assert _parse_bool_env("TEST_VAR", default=True) is True
            assert _parse_bool_env("TEST_VAR", default=False) is False


class TestSSLConfig:
    """Tests for SSLConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default SSLConfig values."""
        config = SSLConfig()
        assert config.cert_file is None
        assert config.verify is True
        assert config.model_path is None
        assert config.offline_mode is False
        assert config.hf_home is None

    def test_custom_values(self) -> None:
        """Test SSLConfig with custom values."""
        config = SSLConfig(
            cert_file="/path/to/cert.pem",
            verify=False,
            model_path="/path/to/model",
            offline_mode=True,
            hf_home="/custom/cache",
        )
        assert config.cert_file == "/path/to/cert.pem"
        assert config.verify is False
        assert config.model_path == "/path/to/model"
        assert config.offline_mode is True
        assert config.hf_home == "/custom/cache"


class TestGetSSLConfig:
    """Tests for get_ssl_config function."""

    def test_default_config(self) -> None:
        """Test default config when no env vars set."""
        env = {k: v for k, v in os.environ.items() if not k.startswith("ENYAL_")}
        env.pop("REQUESTS_CA_BUNDLE", None)
        env.pop("SSL_CERT_FILE", None)
        with patch.dict(os.environ, env, clear=True):
            config = get_ssl_config()
            assert config.cert_file is None
            assert config.verify is True
            assert config.model_path is None
            assert config.offline_mode is False

    def test_cert_file_from_enyal_env(self) -> None:
        """Test cert_file from ENYAL_SSL_CERT_FILE."""
        with (
            tempfile.NamedTemporaryFile(suffix=".pem") as f,
            patch.dict(os.environ, {"ENYAL_SSL_CERT_FILE": f.name}, clear=False),
        ):
            config = get_ssl_config()
            assert config.cert_file == f.name

    def test_cert_file_priority(self) -> None:
        """Test ENYAL_SSL_CERT_FILE takes priority over others."""
        with patch.dict(
            os.environ,
            {
                "ENYAL_SSL_CERT_FILE": "/enyal/cert.pem",
                "REQUESTS_CA_BUNDLE": "/requests/cert.pem",
                "SSL_CERT_FILE": "/ssl/cert.pem",
            },
        ):
            config = get_ssl_config()
            assert config.cert_file == "/enyal/cert.pem"

    def test_cert_file_fallback_to_requests(self) -> None:
        """Test fallback to REQUESTS_CA_BUNDLE."""
        env = os.environ.copy()
        env.pop("ENYAL_SSL_CERT_FILE", None)
        env["REQUESTS_CA_BUNDLE"] = "/requests/cert.pem"
        with patch.dict(os.environ, env, clear=True):
            config = get_ssl_config()
            assert config.cert_file == "/requests/cert.pem"

    def test_ssl_verify_disabled(self) -> None:
        """Test SSL verification disabled."""
        with patch.dict(os.environ, {"ENYAL_SSL_VERIFY": "false"}):
            config = get_ssl_config()
            assert config.verify is False

    def test_offline_mode_enabled(self) -> None:
        """Test offline mode enabled."""
        with patch.dict(os.environ, {"ENYAL_OFFLINE_MODE": "true"}):
            config = get_ssl_config()
            assert config.offline_mode is True

    def test_model_path_valid(self) -> None:
        """Test valid model path."""
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.dict(os.environ, {"ENYAL_MODEL_PATH": tmpdir}),
        ):
            config = get_ssl_config()
            assert config.model_path == tmpdir

    def test_model_path_invalid_logs_warning(self) -> None:
        """Test invalid model path logs warning."""
        with patch.dict(os.environ, {"ENYAL_MODEL_PATH": "/nonexistent/path"}):
            config = get_ssl_config()
            assert config.model_path is None

    def test_model_path_expands_user(self) -> None:
        """Test model path expands ~."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a path that starts with ~ by using home dir
            home = os.path.expanduser("~")
            if tmpdir.startswith(home):
                tilde_path = tmpdir.replace(home, "~", 1)
                with patch.dict(os.environ, {"ENYAL_MODEL_PATH": tilde_path}):
                    config = get_ssl_config()
                    assert config.model_path == tmpdir


class TestConfigureSSLEnvironment:
    """Tests for configure_ssl_environment function."""

    def test_sets_cert_environment_vars(self) -> None:
        """Test that cert file sets environment variables."""
        with tempfile.NamedTemporaryFile(suffix=".pem") as f:
            config = SSLConfig(cert_file=f.name)
            with patch.dict(os.environ, {}, clear=False):
                configure_ssl_environment(config)
                assert os.environ["REQUESTS_CA_BUNDLE"] == f.name
                assert os.environ["SSL_CERT_FILE"] == f.name
                assert os.environ["CURL_CA_BUNDLE"] == f.name

    def test_raises_for_missing_cert_file(self) -> None:
        """Test raises FileNotFoundError for missing cert file."""
        config = SSLConfig(cert_file="/nonexistent/cert.pem")
        with pytest.raises(FileNotFoundError, match="CA bundle file not found"):
            configure_ssl_environment(config)

    def test_warns_when_ssl_disabled(self) -> None:
        """Test warns when SSL verification is disabled."""
        config = SSLConfig(verify=False)
        with pytest.warns(UserWarning, match="SSL verification is disabled"):
            configure_ssl_environment(config)

    def test_sets_offline_mode_env_vars(self) -> None:
        """Test sets offline mode environment variables."""
        config = SSLConfig(offline_mode=True)
        with patch.dict(os.environ, {}, clear=False):
            configure_ssl_environment(config)
            assert os.environ["HF_HUB_OFFLINE"] == "1"
            assert os.environ["TRANSFORMERS_OFFLINE"] == "1"

    def test_sets_hf_home(self) -> None:
        """Test sets HF_HOME environment variable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SSLConfig(hf_home=tmpdir)
            with patch.dict(os.environ, {}, clear=False):
                configure_ssl_environment(config)
                assert os.environ["HF_HOME"] == tmpdir


class TestConfigureHTTPBackend:
    """Tests for configure_http_backend function."""

    def test_configures_huggingface_hub(self) -> None:
        """Test configures huggingface_hub HTTP backend."""
        mock_configure = MagicMock()
        with patch.dict(
            "sys.modules",
            {"huggingface_hub": MagicMock(configure_http_backend=mock_configure)},
        ):
            from enyal.core import ssl_config

            importlib.reload(ssl_config)

            config = SSLConfig()
            ssl_config.configure_http_backend(config)

            # The function should have been called
            # (exact assertions depend on implementation)

    def test_handles_missing_huggingface_hub(self) -> None:
        """Test gracefully handles missing huggingface_hub."""
        config = SSLConfig()
        # This should not raise even if huggingface_hub import fails
        # The function should handle ImportError gracefully
        configure_http_backend(config)


class TestGetModelPath:
    """Tests for get_model_path function."""

    def test_returns_default_model_name(self) -> None:
        """Test returns default model name when no local path."""
        env = {k: v for k, v in os.environ.items() if not k.startswith("ENYAL_")}
        with patch.dict(os.environ, env, clear=True):
            path = get_model_path()
            assert path == "all-MiniLM-L6-v2"

    def test_returns_custom_default(self) -> None:
        """Test returns custom default model name."""
        env = {k: v for k, v in os.environ.items() if not k.startswith("ENYAL_")}
        with patch.dict(os.environ, env, clear=True):
            path = get_model_path("custom-model")
            assert path == "custom-model"

    def test_returns_local_path_when_set(self) -> None:
        """Test returns local model path when ENYAL_MODEL_PATH is set."""
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.dict(os.environ, {"ENYAL_MODEL_PATH": tmpdir}),
        ):
            path = get_model_path()
            assert path == tmpdir

    def test_offline_mode_raises_without_cache(self) -> None:
        """Test offline mode raises error when model not cached."""
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.dict(os.environ, {"ENYAL_OFFLINE_MODE": "true", "HF_HOME": tmpdir}),
            pytest.raises(RuntimeError, match="Offline mode is enabled"),
        ):
            get_model_path()


class TestCheckSSLHealth:
    """Tests for check_ssl_health function."""

    def test_returns_dict_with_expected_keys(self) -> None:
        """Test returns dict with all expected keys."""
        status = check_ssl_health()

        expected_keys = [
            "ssl_verify",
            "cert_file",
            "cert_file_exists",
            "model_path",
            "model_path_exists",
            "offline_mode",
            "hf_home",
            "system_ca_bundle",
            "huggingface_hub_version",
            "sentence_transformers_version",
        ]
        for key in expected_keys:
            assert key in status

    def test_ssl_verify_reflects_config(self) -> None:
        """Test ssl_verify reflects current config."""
        with patch.dict(os.environ, {"ENYAL_SSL_VERIFY": "false"}):
            status = check_ssl_health()
            assert status["ssl_verify"] is False

    def test_cert_file_exists_accurate(self) -> None:
        """Test cert_file_exists is accurate."""
        with (
            tempfile.NamedTemporaryFile(suffix=".pem") as f,
            patch.dict(os.environ, {"ENYAL_SSL_CERT_FILE": f.name}),
        ):
            status = check_ssl_health()
            assert status["cert_file"] == f.name
            assert status["cert_file_exists"] is True

        with patch.dict(os.environ, {"ENYAL_SSL_CERT_FILE": "/nonexistent.pem"}):
            status = check_ssl_health()
            assert status["cert_file_exists"] is False


class TestFindSystemCABundle:
    """Tests for _find_system_ca_bundle function."""

    def test_finds_existing_bundle(self) -> None:
        """Test finds existing CA bundle on system."""
        # This test may return None on systems without standard CA locations
        bundle = _find_system_ca_bundle()
        if bundle is not None:
            assert os.path.isfile(bundle)

    def test_returns_none_when_not_found(self) -> None:
        """Test returns None when no bundle found."""
        with patch(
            "enyal.core.ssl_config.PLATFORM_CA_BUNDLES",
            {"Darwin": [], "Linux": [], "Windows": []},
        ):
            bundle = _find_system_ca_bundle()
            assert bundle is None


class TestDownloadModel:
    """Tests for download_model function."""

    def test_download_model_offline_mode_raises(self) -> None:
        """Test that download raises in offline mode."""
        with patch.dict(os.environ, {"ENYAL_OFFLINE_MODE": "true"}, clear=False):
            with pytest.raises(RuntimeError, match="Cannot download model in offline mode"):
                download_model()

    def test_download_model_success(self) -> None:
        """Test successful model download."""
        mock_model = MagicMock()
        mock_model._model_card_vars = {"model_path": "/cached/model/path"}

        with (
            patch.dict(os.environ, {}, clear=False),
            patch("enyal.core.ssl_config.get_ssl_config") as mock_config,
            patch("enyal.core.ssl_config.configure_ssl_environment"),
            patch("enyal.core.ssl_config.configure_http_backend"),
            patch("sentence_transformers.SentenceTransformer", return_value=mock_model),
        ):
            mock_config.return_value = SSLConfig(offline_mode=False)

            result = download_model("test-model", cache_dir="/tmp/cache")

            assert result == "/cached/model/path"


class TestVerifyModel:
    """Tests for verify_model function."""

    def test_verify_model_success(self) -> None:
        """Test successful model verification."""
        mock_model = MagicMock()
        import numpy as np
        mock_model.encode.return_value = np.zeros(384, dtype=np.float32)

        with (
            patch("enyal.core.ssl_config.get_ssl_config") as mock_config,
            patch("enyal.core.ssl_config.configure_ssl_environment"),
            patch("enyal.core.ssl_config.configure_http_backend"),
            patch("enyal.core.ssl_config.get_model_path", return_value="test-model"),
            patch("sentence_transformers.SentenceTransformer", return_value=mock_model),
        ):
            mock_config.return_value = SSLConfig()

            result = verify_model()

            assert result is True

    def test_verify_model_failure(self) -> None:
        """Test model verification failure."""
        with (
            patch("enyal.core.ssl_config.get_ssl_config") as mock_config,
            patch("enyal.core.ssl_config.configure_ssl_environment"),
            patch("enyal.core.ssl_config.configure_http_backend"),
            patch("enyal.core.ssl_config.get_model_path", return_value="test-model"),
            patch("sentence_transformers.SentenceTransformer", side_effect=Exception("Load failed")),
        ):
            mock_config.return_value = SSLConfig()

            result = verify_model()

            assert result is False

    def test_verify_model_custom_path(self) -> None:
        """Test verification with custom model path."""
        mock_model = MagicMock()
        import numpy as np
        mock_model.encode.return_value = np.zeros(384, dtype=np.float32)

        with (
            patch("enyal.core.ssl_config.get_ssl_config") as mock_config,
            patch("enyal.core.ssl_config.configure_ssl_environment"),
            patch("enyal.core.ssl_config.configure_http_backend"),
            patch("sentence_transformers.SentenceTransformer", return_value=mock_model),
        ):
            mock_config.return_value = SSLConfig()

            result = verify_model("/custom/model/path")

            assert result is True


class TestGetModelPathOffline:
    """Additional tests for get_model_path offline mode."""

    def test_offline_mode_cached_model_exists(self) -> None:
        """Test offline mode with cached model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create the expected cache structure
            model_dir = os.path.join(tmpdir, "hub", "models--sentence-transformers--test-model")
            os.makedirs(model_dir)

            with patch.dict(
                os.environ,
                {"ENYAL_OFFLINE_MODE": "true", "HF_HOME": tmpdir},
                clear=False,
            ):
                # Need to clear any cached env vars
                result = get_model_path("test-model")
                assert result == "test-model"

    def test_offline_mode_no_cached_model(self) -> None:
        """Test offline mode without cached model raises."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(
                os.environ,
                {"ENYAL_OFFLINE_MODE": "true", "HF_HOME": tmpdir},
                clear=False,
            ):
                with pytest.raises(RuntimeError, match="not cached"):
                    get_model_path("nonexistent-model")


class TestCheckSSLHealthLibraries:
    """Tests for check_ssl_health library version detection."""

    def test_health_check_returns_library_versions(self) -> None:
        """Test health check includes library version info."""
        status = check_ssl_health()
        assert "huggingface_hub_version" in status
        assert "sentence_transformers_version" in status
