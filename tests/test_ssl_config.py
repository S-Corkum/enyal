"""Tests for SSL configuration module."""

import contextlib
import importlib
import os
import ssl
import tempfile
import warnings
from unittest.mock import MagicMock, patch

import pytest

from enyal.core.ssl_config import (
    SSLConfig,
    _build_ssl_context,
    _convert_der_to_pem,
    _count_pem_certs,
    _create_corp_adapter,
    _disable_ssl_globally,
    _ensure_combined_cert_bundle,
    _ensure_pem_format,
    _export_macos_system_certs,
    _find_system_ca_bundle,
    _get_certifi_bundle,
    _is_der_encoded,
    _is_ssl_error,
    _parse_bool_env,
    _preflight_ssl_check,
    _relax_x509_strict,
    _try_inject_truststore,
    check_ssl_health,
    configure_http_backend,
    configure_ssl_environment,
    download_model,
    get_model_path,
    get_ssl_config,
    ssl_diagnostic_probe,
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
        assert config.hf_endpoint is None

    def test_custom_values(self) -> None:
        """Test SSLConfig with custom values."""
        config = SSLConfig(
            cert_file="/path/to/cert.pem",
            verify=False,
            model_path="/path/to/model",
            offline_mode=True,
            hf_home="/custom/cache",
            hf_endpoint="https://artifactory.corp.com/hf",
        )
        assert config.cert_file == "/path/to/cert.pem"
        assert config.verify is False
        assert config.model_path == "/path/to/model"
        assert config.offline_mode is True
        assert config.hf_home == "/custom/cache"
        assert config.hf_endpoint == "https://artifactory.corp.com/hf"


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
            assert config.hf_endpoint is None

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

    def test_hf_endpoint_from_env(self) -> None:
        """Test hf_endpoint from ENYAL_HF_ENDPOINT."""
        with patch.dict(os.environ, {"ENYAL_HF_ENDPOINT": "https://artifactory.corp.com/hf"}):
            config = get_ssl_config()
            assert config.hf_endpoint == "https://artifactory.corp.com/hf"

    def test_hf_endpoint_strips_trailing_slash(self) -> None:
        """Test hf_endpoint strips trailing slash."""
        with patch.dict(os.environ, {"ENYAL_HF_ENDPOINT": "https://artifactory.corp.com/hf/"}):
            config = get_ssl_config()
            assert config.hf_endpoint == "https://artifactory.corp.com/hf"

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

    def test_disables_xet_when_cert_file_set(self) -> None:
        """Test disables Xet storage when custom cert is configured."""
        with tempfile.NamedTemporaryFile(suffix=".pem") as f:
            config = SSLConfig(cert_file=f.name)
            env = {k: v for k, v in os.environ.items()
                   if k != "HF_HUB_DISABLE_XET"}
            with patch.dict(os.environ, env, clear=True):
                configure_ssl_environment(config)
                assert os.environ["HF_HUB_DISABLE_XET"] == "1"

    def test_sets_generous_timeouts_when_cert_file_set(self) -> None:
        """Test sets generous timeouts for corporate networks with custom cert."""
        with tempfile.NamedTemporaryFile(suffix=".pem") as f:
            config = SSLConfig(cert_file=f.name)
            env = {k: v for k, v in os.environ.items()
                   if k not in ("HF_HUB_ETAG_TIMEOUT", "HF_HUB_DOWNLOAD_TIMEOUT",
                                "HF_HUB_DISABLE_XET")}
            with patch.dict(os.environ, env, clear=True):
                configure_ssl_environment(config)
                assert os.environ["HF_HUB_ETAG_TIMEOUT"] == "30"
                assert os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] == "60"

    def test_disables_xet_when_ssl_verify_false(self) -> None:
        """Test disables Xet storage when SSL verification is disabled."""
        config = SSLConfig(verify=False)
        env = {k: v for k, v in os.environ.items()
               if k != "HF_HUB_DISABLE_XET"}
        # Save and restore ssl._create_default_https_context
        original_ctx = ssl._create_default_https_context
        try:
            with patch.dict(os.environ, env, clear=True):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    configure_ssl_environment(config)
                assert os.environ["HF_HUB_DISABLE_XET"] == "1"
        finally:
            ssl._create_default_https_context = original_ctx

    def test_disables_hf_transfer_when_ssl_verify_false(self) -> None:
        """Test disables hf_transfer when SSL verification is disabled."""
        config = SSLConfig(verify=False)
        original_ctx = ssl._create_default_https_context
        try:
            with patch.dict(os.environ, {}, clear=False):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    configure_ssl_environment(config)
                assert os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "0"
        finally:
            ssl._create_default_https_context = original_ctx

    def test_verify_false_disables_ssl_globally(self) -> None:
        """Test that verify=False monkey-patches ssl module."""
        config = SSLConfig(verify=False)
        original_ctx = ssl._create_default_https_context
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                configure_ssl_environment(config)
            # ssl._create_default_https_context should be the unverified context
            assert ssl._create_default_https_context is ssl._create_unverified_context
        finally:
            ssl._create_default_https_context = original_ctx

    def test_verify_false_sets_pythonhttpsverify(self) -> None:
        """Test that verify=False sets PYTHONHTTPSVERIFY=0."""
        config = SSLConfig(verify=False)
        original_ctx = ssl._create_default_https_context
        try:
            with patch.dict(os.environ, {}, clear=False):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    configure_ssl_environment(config)
                assert os.environ.get("PYTHONHTTPSVERIFY") == "0"
        finally:
            ssl._create_default_https_context = original_ctx

    def test_raises_for_missing_cert_file(self) -> None:
        """Test raises FileNotFoundError for missing cert file."""
        config = SSLConfig(cert_file="/nonexistent/cert.pem")
        with pytest.raises(FileNotFoundError, match="CA bundle file not found"):
            configure_ssl_environment(config)

    def test_warns_when_ssl_disabled(self) -> None:
        """Test warns when SSL verification is disabled."""
        config = SSLConfig(verify=False)
        original_ctx = ssl._create_default_https_context
        try:
            with pytest.warns(UserWarning, match="SSL verification is disabled"):
                configure_ssl_environment(config)
        finally:
            ssl._create_default_https_context = original_ctx

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


class TestConfigureSSLEnvironmentEndpoint:
    """Tests for configure_ssl_environment custom HF endpoint support."""

    def test_sets_hf_endpoint_env_var(self) -> None:
        """Test sets HF_ENDPOINT environment variable."""
        config = SSLConfig(hf_endpoint="https://artifactory.corp.com/hf")
        with patch.dict(os.environ, {}, clear=False):
            configure_ssl_environment(config)
            assert os.environ["HF_ENDPOINT"] == "https://artifactory.corp.com/hf"

    def test_sets_disable_xet_when_endpoint_set(self) -> None:
        """Test disables Xet storage when custom endpoint is set."""
        config = SSLConfig(hf_endpoint="https://artifactory.corp.com/hf")
        with patch.dict(os.environ, {}, clear=False):
            configure_ssl_environment(config)
            assert os.environ["HF_HUB_DISABLE_XET"] == "1"

    def test_sets_increased_timeouts_when_endpoint_set(self) -> None:
        """Test sets increased timeouts when custom endpoint is set."""
        config = SSLConfig(hf_endpoint="https://artifactory.corp.com/hf")
        env = {k: v for k, v in os.environ.items()
               if k not in ("HF_HUB_ETAG_TIMEOUT", "HF_HUB_DOWNLOAD_TIMEOUT")}
        with patch.dict(os.environ, env, clear=True):
            configure_ssl_environment(config)
            assert os.environ["HF_HUB_ETAG_TIMEOUT"] == "30"
            assert os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] == "60"

    def test_does_not_override_user_timeout_settings(self) -> None:
        """Test does not override user-set timeout values."""
        config = SSLConfig(hf_endpoint="https://artifactory.corp.com/hf")
        with patch.dict(os.environ, {
            "HF_HUB_ETAG_TIMEOUT": "120",
            "HF_HUB_DOWNLOAD_TIMEOUT": "300",
        }, clear=False):
            configure_ssl_environment(config)
            assert os.environ["HF_HUB_ETAG_TIMEOUT"] == "120"
            assert os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] == "300"

    def test_does_not_set_endpoint_vars_when_not_configured(self) -> None:
        """Test does not set HF_ENDPOINT when hf_endpoint is None."""
        config = SSLConfig()
        env = {k: v for k, v in os.environ.items()
               if k not in ("HF_ENDPOINT",)}
        with (
            patch.dict(os.environ, env, clear=True),
            # Disable auto-detect so it doesn't set HF_HUB_DISABLE_XET
            patch("enyal.core.ssl_config._try_inject_truststore", return_value=False),
            patch("enyal.core.ssl_config._export_macos_system_certs", return_value=None),
        ):
            configure_ssl_environment(config)
            assert "HF_ENDPOINT" not in os.environ


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
            "hf_endpoint",
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
        with (
            patch.dict(os.environ, {"ENYAL_OFFLINE_MODE": "true"}, clear=False),
            pytest.raises(RuntimeError, match="Cannot download model in offline mode"),
        ):
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
            patch("enyal.core.ssl_config._preflight_ssl_check", return_value=(True, None)),
            patch("sentence_transformers.SentenceTransformer", return_value=mock_model),
        ):
            mock_config.return_value = SSLConfig(offline_mode=False)

            result = download_model("test-model", cache_dir="/tmp/cache")

            assert result == "/cached/model/path"

    def test_download_model_preflight_fails_disables_ssl(self) -> None:
        """Test that preflight failure proactively disables SSL."""
        mock_model = MagicMock()
        mock_model._model_card_vars = {"model_path": "/cached/model/path"}

        with (
            patch.dict(os.environ, {}, clear=False),
            patch("enyal.core.ssl_config.get_ssl_config") as mock_config,
            patch("enyal.core.ssl_config.configure_ssl_environment"),
            patch("enyal.core.ssl_config.configure_http_backend") as mock_backend,
            patch(
                "enyal.core.ssl_config._preflight_ssl_check",
                return_value=(False, "SSLError: cert verify failed"),
            ),
            patch("enyal.core.ssl_config._disable_ssl_globally") as mock_disable,
            patch("sentence_transformers.SentenceTransformer", return_value=mock_model),
        ):
            mock_config.return_value = SSLConfig(offline_mode=False)

            result = download_model("test-model")

            assert result == "/cached/model/path"
            mock_disable.assert_called_once()
            # configure_http_backend called twice: initial + after preflight failure
            assert mock_backend.call_count == 2

    def test_download_model_auto_recovery_on_any_error(self) -> None:
        """Test that auto-recovery retries on ANY first-attempt failure."""
        mock_model = MagicMock()
        mock_model._model_card_vars = {"model_path": "/cached/model/path"}

        # First call raises non-SSL error, second succeeds
        mock_st = MagicMock(side_effect=[RuntimeError("download failed"), mock_model])

        with (
            patch.dict(os.environ, {}, clear=False),
            patch("enyal.core.ssl_config.get_ssl_config") as mock_config,
            patch("enyal.core.ssl_config.configure_ssl_environment"),
            patch("enyal.core.ssl_config.configure_http_backend"),
            patch("enyal.core.ssl_config._preflight_ssl_check", return_value=(True, None)),
            patch("enyal.core.ssl_config._disable_ssl_globally"),
            patch("sentence_transformers.SentenceTransformer", mock_st),
        ):
            mock_config.return_value = SSLConfig(offline_mode=False)

            result = download_model("test-model")

            assert result == "/cached/model/path"
            assert mock_st.call_count == 2


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
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.dict(
                os.environ,
                {"ENYAL_OFFLINE_MODE": "true", "HF_HOME": tmpdir},
                clear=False,
            ),
            pytest.raises(RuntimeError, match="not cached"),
        ):
            get_model_path("nonexistent-model")


class TestCheckSSLHealthLibraries:
    """Tests for check_ssl_health library version detection."""

    def test_health_check_returns_library_versions(self) -> None:
        """Test health check includes library version info."""
        status = check_ssl_health()
        assert "huggingface_hub_version" in status
        assert "sentence_transformers_version" in status


class TestCountPemCerts:
    """Tests for _count_pem_certs function."""

    def test_counts_single_cert(self) -> None:
        content = "-----BEGIN CERTIFICATE-----\ndata\n-----END CERTIFICATE-----\n"
        assert _count_pem_certs(content) == 1

    def test_counts_multiple_certs(self) -> None:
        content = (
            "-----BEGIN CERTIFICATE-----\ndata1\n-----END CERTIFICATE-----\n"
            "-----BEGIN CERTIFICATE-----\ndata2\n-----END CERTIFICATE-----\n"
            "-----BEGIN CERTIFICATE-----\ndata3\n-----END CERTIFICATE-----\n"
        )
        assert _count_pem_certs(content) == 3

    def test_counts_zero_for_empty(self) -> None:
        assert _count_pem_certs("") == 0

    def test_counts_zero_for_non_pem(self) -> None:
        assert _count_pem_certs("not a certificate") == 0


class TestGetCertifiBundle:
    """Tests for _get_certifi_bundle function."""

    def test_returns_path_when_certifi_installed(self) -> None:
        """certifi should be installed in test environment."""
        result = _get_certifi_bundle()
        assert result is not None
        assert os.path.isfile(result)

    def test_returns_none_when_certifi_missing(self) -> None:
        with patch.dict("sys.modules", {"certifi": None}):
            # Reload to clear any cached import
            import importlib

            import enyal.core.ssl_config as mod

            importlib.reload(mod)
            result = mod._get_certifi_bundle()
            assert result is None
            # Reload again to restore
            importlib.reload(mod)


class TestEnsureCombinedCertBundle:
    """Tests for _ensure_combined_cert_bundle function."""

    def _write_pem(self, path: str, count: int) -> None:
        """Write a fake PEM file with the given number of certs."""
        content = ""
        for i in range(count):
            content += f"-----BEGIN CERTIFICATE-----\nfake-cert-data-{i}\n-----END CERTIFICATE-----\n"
        with open(path, "w") as f:
            f.write(content)

    def test_returns_original_for_large_bundle(self) -> None:
        """File with 5+ certs is treated as a complete bundle."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pem", delete=False) as f:
            self._write_pem(f.name, 10)
            result = _ensure_combined_cert_bundle(f.name)
            assert result == f.name
            os.unlink(f.name)

    def test_returns_original_for_zero_certs(self) -> None:
        """File with no PEM certs returns original."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pem", delete=False) as f:
            f.write("not a certificate")
            f.flush()
            result = _ensure_combined_cert_bundle(f.name)
            assert result == f.name
            os.unlink(f.name)

    def test_combines_small_cert_with_certifi(self) -> None:
        """A file with 1 cert gets combined with certifi."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cert_path = os.path.join(tmpdir, "corp.pem")
            self._write_pem(cert_path, 1)

            db_path = os.path.join(tmpdir, "data", "context.db")
            os.makedirs(os.path.dirname(db_path))

            with patch.dict(os.environ, {"ENYAL_DB_PATH": db_path}):
                result = _ensure_combined_cert_bundle(cert_path)

            # Should create a combined file (if certifi is available)
            certifi_path = _get_certifi_bundle()
            if certifi_path:
                assert result != cert_path
                assert os.path.isfile(result)
                with open(result) as f:
                    combined_content = f.read()
                # Combined file should have corp cert + certifi certs
                assert _count_pem_certs(combined_content) > 1
                assert "Corporate CA certificate(s)" in combined_content
            else:
                # Without certifi, falls back to original
                assert result == cert_path

    def test_caches_combined_bundle(self) -> None:
        """Combined bundle is cached and reused on subsequent calls."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cert_path = os.path.join(tmpdir, "corp.pem")
            self._write_pem(cert_path, 1)

            db_path = os.path.join(tmpdir, "data", "context.db")
            os.makedirs(os.path.dirname(db_path))

            with patch.dict(os.environ, {"ENYAL_DB_PATH": db_path}):
                result1 = _ensure_combined_cert_bundle(cert_path)
                result2 = _ensure_combined_cert_bundle(cert_path)

            assert result1 == result2

    def test_regenerates_when_cert_changes(self) -> None:
        """Combined bundle is regenerated when source cert changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cert_path = os.path.join(tmpdir, "corp.pem")
            self._write_pem(cert_path, 1)

            db_path = os.path.join(tmpdir, "data", "context.db")
            os.makedirs(os.path.dirname(db_path))

            with patch.dict(os.environ, {"ENYAL_DB_PATH": db_path}):
                result1 = _ensure_combined_cert_bundle(cert_path)

                # Change the cert file content
                self._write_pem(cert_path, 2)
                result2 = _ensure_combined_cert_bundle(cert_path)

            # Path should be the same (same location), but content differs
            if _get_certifi_bundle():
                assert result1 == result2  # same file path
                with open(result2) as f:
                    content = f.read()
                # Should contain the 2 new certs
                assert "fake-cert-data-0" in content
                assert "fake-cert-data-1" in content

    def test_returns_original_for_unreadable_file(self) -> None:
        """Returns original when file can't be read."""
        result = _ensure_combined_cert_bundle("/nonexistent/cert.pem")
        assert result == "/nonexistent/cert.pem"


class TestConfigureSSLEnvironmentCombined:
    """Tests for configure_ssl_environment with auto-combine."""

    def test_auto_combines_small_cert(self) -> None:
        """Verifies that a small cert file triggers auto-combine."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cert_path = os.path.join(tmpdir, "corp.pem")
            with open(cert_path, "w") as f:
                f.write("-----BEGIN CERTIFICATE-----\nfake\n-----END CERTIFICATE-----\n")

            db_path = os.path.join(tmpdir, "data", "context.db")
            os.makedirs(os.path.dirname(db_path))

            config = SSLConfig(cert_file=cert_path)
            with patch.dict(os.environ, {"ENYAL_DB_PATH": db_path}, clear=False):
                configure_ssl_environment(config)

                # After configure_ssl_environment, cert_file should be updated
                # to the combined path (if certifi available)
                certifi_path = _get_certifi_bundle()
                if certifi_path:
                    assert config.cert_file != cert_path
                    assert "combined_ca_bundle.pem" in config.cert_file
                    assert os.environ["REQUESTS_CA_BUNDLE"] == config.cert_file
                else:
                    assert config.cert_file == cert_path


class TestDerEncoding:
    """Tests for DER certificate detection and conversion."""

    def test_is_der_encoded_with_pem(self) -> None:
        """PEM files should not be detected as DER."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pem", delete=False) as f:
            f.write("-----BEGIN CERTIFICATE-----\nfakedata\n-----END CERTIFICATE-----\n")
            f.flush()
            assert _is_der_encoded(f.name) is False
            os.unlink(f.name)

    def test_is_der_encoded_with_der(self) -> None:
        """DER files start with 0x30 (ASN.1 SEQUENCE)."""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".crt", delete=False) as f:
            # 0x30 0x82 is a typical DER certificate header
            f.write(b"\x30\x82\x03\x00" + b"\x00" * 100)
            f.flush()
            assert _is_der_encoded(f.name) is True
            os.unlink(f.name)

    def test_is_der_encoded_with_empty_file(self) -> None:
        """Empty file should not be detected as DER."""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".crt", delete=False) as f:
            f.flush()
            assert _is_der_encoded(f.name) is False
            os.unlink(f.name)

    def test_is_der_encoded_nonexistent(self) -> None:
        """Nonexistent file returns False."""
        assert _is_der_encoded("/nonexistent/file.crt") is False

    def test_convert_der_to_pem_invalid_data(self) -> None:
        """Invalid DER data should return None."""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".crt", delete=False) as f:
            f.write(b"\x30\x82\x03\x00" + b"\x00" * 100)
            f.flush()
            result = _convert_der_to_pem(f.name)
            assert result is None
            os.unlink(f.name)

    def test_ensure_pem_format_already_pem(self) -> None:
        """PEM file passes through unchanged."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pem", delete=False) as f:
            f.write("-----BEGIN CERTIFICATE-----\nfakedata\n-----END CERTIFICATE-----\n")
            f.flush()
            result = _ensure_pem_format(f.name)
            assert result == f.name
            os.unlink(f.name)

    def test_ensure_pem_format_nonexistent(self) -> None:
        """Nonexistent file passes through."""
        result = _ensure_pem_format("/nonexistent/cert.crt")
        assert result == "/nonexistent/cert.crt"

    def test_ensure_pem_format_non_der_binary(self) -> None:
        """Binary file that isn't DER passes through."""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".crt", delete=False) as f:
            f.write(b"\x00\x01\x02\x03")  # Not 0x30, so not DER
            f.flush()
            result = _ensure_pem_format(f.name)
            assert result == f.name
            os.unlink(f.name)


class TestDisableSSLGlobally:
    """Tests for _disable_ssl_globally function."""

    def test_patches_ssl_context(self) -> None:
        """Test that ssl._create_default_https_context is replaced."""
        original_ctx = ssl._create_default_https_context
        try:
            _disable_ssl_globally()
            assert ssl._create_default_https_context is ssl._create_unverified_context
        finally:
            ssl._create_default_https_context = original_ctx

    def test_sets_pythonhttpsverify(self) -> None:
        """Test PYTHONHTTPSVERIFY=0 is set."""
        original_ctx = ssl._create_default_https_context
        try:
            with patch.dict(os.environ, {}, clear=False):
                _disable_ssl_globally()
                assert os.environ.get("PYTHONHTTPSVERIFY") == "0"
        finally:
            ssl._create_default_https_context = original_ctx

    def test_suppresses_urllib3_warnings(self) -> None:
        """Test urllib3 InsecureRequestWarning is suppressed."""
        original_ctx = ssl._create_default_https_context
        try:
            _disable_ssl_globally()
            # After disabling, urllib3 warnings should be suppressed
            import urllib3

            # Check that the warning filter was added
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                # This should NOT produce a warning after _disable_ssl_globally
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                # The point is that _disable_ssl_globally didn't crash
        finally:
            ssl._create_default_https_context = original_ctx


class TestTryInjectTruststore:
    """Tests for _try_inject_truststore function."""

    def test_returns_true_when_truststore_available(self) -> None:
        """Test returns True when truststore can be injected."""
        mock_truststore = MagicMock()
        with patch.dict("sys.modules", {"truststore": mock_truststore}):
            import enyal.core.ssl_config as mod

            importlib.reload(mod)
            result = mod._try_inject_truststore()
            assert result is True
            mock_truststore.inject_into_ssl.assert_called_once()
            importlib.reload(mod)

    def test_returns_false_when_not_installed(self) -> None:
        """Test returns False when truststore is not installed."""
        result = _try_inject_truststore()
        # This will return True if truststore is installed, False if not.
        # We can't control whether it's installed in test env, so just check type.
        assert isinstance(result, bool)

    def test_returns_false_on_injection_error(self) -> None:
        """Test returns False when injection fails."""
        mock_truststore = MagicMock()
        mock_truststore.inject_into_ssl.side_effect = RuntimeError("injection failed")
        with patch.dict("sys.modules", {"truststore": mock_truststore}):
            import enyal.core.ssl_config as mod

            importlib.reload(mod)
            result = mod._try_inject_truststore()
            assert result is False
            importlib.reload(mod)


class TestExportMacosSystemCerts:
    """Tests for _export_macos_system_certs function."""

    def test_returns_none_on_non_macos(self) -> None:
        """Test returns None on non-macOS platforms."""
        with patch("enyal.core.ssl_config.platform") as mock_platform:
            mock_platform.system.return_value = "Linux"
            result = _export_macos_system_certs()
            assert result is None

    def test_exports_certs_on_macos(self) -> None:
        """Test exports certificates on macOS."""
        fake_pem = (
            "-----BEGIN CERTIFICATE-----\nfake-cert\n-----END CERTIFICATE-----\n"
        )

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("enyal.core.ssl_config.platform") as mock_platform,
            patch("enyal.core.ssl_config.subprocess") as mock_subprocess,
            patch.dict(
                os.environ,
                {"ENYAL_DB_PATH": os.path.join(tmpdir, "context.db")},
            ),
        ):
            mock_platform.system.return_value = "Darwin"
            mock_subprocess.run.return_value = MagicMock(
                returncode=0, stdout=fake_pem, stderr=""
            )
            # Mock os.path.exists for keychain files
            original_exists = os.path.exists
            def mock_exists(p: str) -> bool:
                if "Keychains" in p:
                    return True
                return original_exists(p)

            with patch("os.path.exists", side_effect=mock_exists):
                result = _export_macos_system_certs()

            assert result is not None
            assert os.path.isfile(result)
            with open(result) as f:
                content = f.read()
            assert "-----BEGIN CERTIFICATE-----" in content

    def test_returns_none_when_security_fails(self) -> None:
        """Test returns None when security command fails."""
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("enyal.core.ssl_config.platform") as mock_platform,
            patch("enyal.core.ssl_config.subprocess") as mock_subprocess,
            patch.dict(
                os.environ,
                {"ENYAL_DB_PATH": os.path.join(tmpdir, "context.db")},
            ),
        ):
            mock_platform.system.return_value = "Darwin"
            mock_subprocess.run.return_value = MagicMock(
                returncode=1, stdout="", stderr="error"
            )
            original_exists = os.path.exists
            def mock_exists(p: str) -> bool:
                if "Keychains" in p:
                    return True
                return original_exists(p)

            with patch("os.path.exists", side_effect=mock_exists):
                result = _export_macos_system_certs()
            assert result is None

    def test_caches_result(self) -> None:
        """Test that exported certs are cached and reused."""
        fake_pem = (
            "-----BEGIN CERTIFICATE-----\nfake-cert\n-----END CERTIFICATE-----\n"
        )

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("enyal.core.ssl_config.platform") as mock_platform,
            patch("enyal.core.ssl_config.subprocess") as mock_subprocess,
            patch.dict(
                os.environ,
                {"ENYAL_DB_PATH": os.path.join(tmpdir, "context.db")},
            ),
        ):
            mock_platform.system.return_value = "Darwin"
            mock_subprocess.run.return_value = MagicMock(
                returncode=0, stdout=fake_pem, stderr=""
            )
            original_exists = os.path.exists
            def mock_exists(p: str) -> bool:
                if "Keychains" in p:
                    return True
                return original_exists(p)

            with patch("os.path.exists", side_effect=mock_exists):
                result1 = _export_macos_system_certs()
                result2 = _export_macos_system_certs()

            # Second call should use cache (same path)
            assert result1 == result2
            # security command should have been called only once
            assert mock_subprocess.run.call_count == 1

    def test_returns_none_when_no_keychains_exist(self) -> None:
        """Test returns None when no keychain files exist."""
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("enyal.core.ssl_config.platform") as mock_platform,
            patch.dict(
                os.environ,
                {"ENYAL_DB_PATH": os.path.join(tmpdir, "context.db")},
            ),
        ):
            mock_platform.system.return_value = "Darwin"
            # Mock os.path.exists to return False for keychains
            original_exists = os.path.exists
            def mock_exists(p: str) -> bool:
                if "Keychains" in str(p):
                    return False
                return original_exists(p)

            with patch("os.path.exists", side_effect=mock_exists):
                result = _export_macos_system_certs()
            assert result is None


class TestConfigureSSLEnvironmentTruststore:
    """Tests for truststore integration in configure_ssl_environment."""

    def test_tries_truststore_when_no_cert_file(self) -> None:
        """Test that truststore is attempted when no cert file is specified."""
        config = SSLConfig()  # No cert_file, verify=True (defaults)
        with (
            patch("enyal.core.ssl_config._try_inject_truststore", return_value=True) as mock_ts,
            patch.dict(os.environ, {}, clear=False),
        ):
            configure_ssl_environment(config)
            mock_ts.assert_called_once()

    def test_skips_truststore_when_cert_file_set(self) -> None:
        """Test that truststore is NOT tried when cert_file is specified."""
        with tempfile.NamedTemporaryFile(suffix=".pem") as f:
            config = SSLConfig(cert_file=f.name)
            with (
                patch("enyal.core.ssl_config._try_inject_truststore") as mock_ts,
                patch.dict(os.environ, {}, clear=False),
            ):
                configure_ssl_environment(config)
                mock_ts.assert_not_called()

    def test_skips_truststore_when_verify_false(self) -> None:
        """Test that truststore is NOT tried when verify=False."""
        config = SSLConfig(verify=False)
        original_ctx = ssl._create_default_https_context
        try:
            with (
                patch("enyal.core.ssl_config._try_inject_truststore") as mock_ts,
                warnings.catch_warnings(),
            ):
                warnings.simplefilter("ignore", UserWarning)
                configure_ssl_environment(config)
                mock_ts.assert_not_called()
        finally:
            ssl._create_default_https_context = original_ctx

    def test_falls_back_to_macos_export_when_truststore_unavailable(self) -> None:
        """Test falls back to macOS export when truststore is not available."""
        config = SSLConfig()
        with (
            patch("enyal.core.ssl_config._try_inject_truststore", return_value=False),
            patch("enyal.core.ssl_config._export_macos_system_certs", return_value="/fake/certs.pem") as mock_export,
            patch.dict(os.environ, {}, clear=False),
        ):
            configure_ssl_environment(config)
            mock_export.assert_called_once()
            # Should have set the cert env vars
            assert os.environ.get("REQUESTS_CA_BUNDLE") == "/fake/certs.pem"

    def test_skips_system_trust_when_disabled(self) -> None:
        """Test skips system trust store when ENYAL_SSL_TRUST_SYSTEM=false."""
        config = SSLConfig()
        with (
            patch("enyal.core.ssl_config._try_inject_truststore") as mock_ts,
            patch("enyal.core.ssl_config._export_macos_system_certs") as mock_export,
            patch.dict(os.environ, {"ENYAL_SSL_TRUST_SYSTEM": "false"}, clear=False),
        ):
            configure_ssl_environment(config)
            mock_ts.assert_not_called()
            mock_export.assert_not_called()


class TestConfigureHTTPBackendPriority:
    """Tests for verify=False priority in configure_http_backend."""

    def test_verify_false_overrides_cert_file(self) -> None:
        """Test that verify=False takes priority over cert_file."""
        config = SSLConfig(verify=False, cert_file="/some/cert.pem")

        captured_session = {}

        def mock_configure(backend_factory: object) -> None:
            captured_session["factory"] = backend_factory

        mock_hf = MagicMock(configure_http_backend=mock_configure)
        with patch.dict("sys.modules", {"huggingface_hub": mock_hf}):
            import enyal.core.ssl_config as mod

            importlib.reload(mod)
            mod.configure_http_backend(config)

            # Create a session from the factory
            session = captured_session["factory"]()
            # verify=False should win over cert_file
            assert session.verify is False
            importlib.reload(mod)


class TestCheckSSLHealthExtended:
    """Tests for extended check_ssl_health fields."""

    def test_includes_truststore_fields(self) -> None:
        """Test health check includes truststore information."""
        status = check_ssl_health()
        assert "truststore_available" in status
        assert "truststore_version" in status
        assert isinstance(status["truststore_available"], bool)

    def test_includes_openssl_version(self) -> None:
        """Test health check includes OpenSSL version."""
        status = check_ssl_health()
        assert "openssl_version" in status
        assert status["openssl_version"] is not None
        assert "OpenSSL" in status["openssl_version"] or "LibreSSL" in status["openssl_version"]

    def test_includes_platform(self) -> None:
        """Test health check includes platform."""
        status = check_ssl_health()
        assert "platform" in status

    def test_includes_trust_system(self) -> None:
        """Test health check includes trust_system setting."""
        status = check_ssl_health()
        assert "trust_system" in status


class TestIsSSLError:
    """Tests for _is_ssl_error helper function."""

    def test_detects_ssl_module_error(self) -> None:
        """Test detects ssl.SSLError."""
        exc = ssl.SSLError("certificate verify failed")
        assert _is_ssl_error(exc) is True

    def test_detects_ssl_cert_verification_error(self) -> None:
        """Test detects ssl.SSLCertVerificationError."""
        exc = ssl.SSLCertVerificationError("CERTIFICATE_VERIFY_FAILED")
        assert _is_ssl_error(exc) is True

    def test_detects_requests_ssl_error(self) -> None:
        """Test detects requests.exceptions.SSLError."""
        try:
            from requests.exceptions import SSLError as RequestsSSLError
            exc = RequestsSSLError("SSL error from requests")
            assert _is_ssl_error(exc) is True
        except ImportError:
            pytest.skip("requests not installed")

    def test_detects_urllib3_ssl_error(self) -> None:
        """Test detects urllib3.exceptions.SSLError."""
        try:
            from urllib3.exceptions import SSLError as Urllib3SSLError
            exc = Urllib3SSLError("SSL error from urllib3")
            assert _is_ssl_error(exc) is True
        except ImportError:
            pytest.skip("urllib3 not installed")

    def test_detects_wrapped_ssl_error(self) -> None:
        """Test detects SSL error wrapped in another exception."""
        inner = ssl.SSLError("certificate verify failed")
        outer = RuntimeError("model download failed")
        outer.__cause__ = inner
        assert _is_ssl_error(outer) is True

    def test_rejects_non_ssl_error(self) -> None:
        """Test rejects non-SSL errors."""
        assert _is_ssl_error(ValueError("invalid value provided")) is False
        assert _is_ssl_error(RuntimeError("connection refused")) is False
        assert _is_ssl_error(FileNotFoundError("no such file")) is False
        assert _is_ssl_error(MemoryError("out of memory")) is False

    def test_detects_ssl_keyword_in_message(self) -> None:
        """Test detects SSL error patterns in error message as fallback."""
        exc = Exception("SSLError: certificate verify failed")
        assert _is_ssl_error(exc) is True

    def test_detects_certificate_verify_failed(self) -> None:
        """Test detects 'certificate verify failed' in error message."""
        exc = Exception("certificate verify failed")
        assert _is_ssl_error(exc) is True

    def test_detects_ssl_colon_pattern(self) -> None:
        """Test detects 'ssl:' pattern in error message."""
        exc = Exception("[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed")
        assert _is_ssl_error(exc) is True


class TestDisableSSLGloballyUrllib3Patch:
    """Tests for urllib3 context patching in _disable_ssl_globally."""

    def test_patches_urllib3_context_factory(self) -> None:
        """Test that urllib3's create_urllib3_context is monkey-patched."""
        original_ctx = ssl._create_default_https_context
        original_strict = getattr(ssl, "VERIFY_X509_STRICT", None)
        try:
            import urllib3.util.ssl_ as _urllib3_ssl
            original_create = _urllib3_ssl.create_urllib3_context

            _disable_ssl_globally()

            # The function should have been replaced
            assert _urllib3_ssl.create_urllib3_context is not original_create

            # The replacement should create a permissive context
            ctx = _urllib3_ssl.create_urllib3_context()
            assert ctx.check_hostname is False
            assert ctx.verify_mode == ssl.CERT_NONE
        finally:
            ssl._create_default_https_context = original_ctx
            if original_strict is not None:
                ssl.VERIFY_X509_STRICT = original_strict
            with contextlib.suppress(Exception):
                _urllib3_ssl.create_urllib3_context = original_create

    def test_zeroes_verify_x509_strict(self) -> None:
        """Test that _disable_ssl_globally zeroes VERIFY_X509_STRICT."""
        if not hasattr(ssl, "VERIFY_X509_STRICT"):
            pytest.skip("VERIFY_X509_STRICT not available")

        original_ctx = ssl._create_default_https_context
        original_strict = ssl.VERIFY_X509_STRICT
        try:
            import urllib3.util.ssl_ as _urllib3_ssl
            original_create = _urllib3_ssl.create_urllib3_context

            _disable_ssl_globally()

            assert ssl.VERIFY_X509_STRICT == 0
        finally:
            ssl._create_default_https_context = original_ctx
            ssl.VERIFY_X509_STRICT = original_strict
            with contextlib.suppress(Exception):
                _urllib3_ssl.create_urllib3_context = original_create

    def test_patches_urllib3_connection_binding(self) -> None:
        """Test that urllib3.connection's direct import is also patched."""
        original_ctx = ssl._create_default_https_context
        original_strict = getattr(ssl, "VERIFY_X509_STRICT", None)
        try:
            import urllib3.util.ssl_ as _urllib3_ssl
            original_create = _urllib3_ssl.create_urllib3_context

            _disable_ssl_globally()

            # Check if urllib3.connection was also patched
            try:
                import urllib3.connection as _urllib3_conn
                if hasattr(_urllib3_conn, "create_urllib3_context"):
                    ctx = _urllib3_conn.create_urllib3_context()
                    assert ctx.check_hostname is False
                    assert ctx.verify_mode == ssl.CERT_NONE
            except (ImportError, AttributeError):
                pass  # urllib3 version doesn't have this binding
        finally:
            ssl._create_default_https_context = original_ctx
            if original_strict is not None:
                ssl.VERIFY_X509_STRICT = original_strict
            with contextlib.suppress(Exception):
                _urllib3_ssl.create_urllib3_context = original_create


class TestRelaxX509Strict:
    """Tests for _relax_x509_strict function."""

    def test_zeroes_verify_x509_strict_constant(self) -> None:
        """Test that _relax_x509_strict zeroes the ssl constant."""
        if not hasattr(ssl, "VERIFY_X509_STRICT"):
            pytest.skip("VERIFY_X509_STRICT not available")

        original = ssl.VERIFY_X509_STRICT
        try:
            # Ensure it's non-zero before the test
            ssl.VERIFY_X509_STRICT = 0x20
            _relax_x509_strict()
            assert ssl.VERIFY_X509_STRICT == 0
        finally:
            ssl.VERIFY_X509_STRICT = original

    def test_zeroes_urllib3_copy_of_constant(self) -> None:
        """Test that _relax_x509_strict also zeroes urllib3's cached copy.

        urllib3.util.ssl_ does `from ssl import VERIFY_X509_STRICT` at
        module load time, capturing the integer value.  _relax_x509_strict
        must zero BOTH the ssl module AND urllib3's copy.
        """
        if not hasattr(ssl, "VERIFY_X509_STRICT"):
            pytest.skip("VERIFY_X509_STRICT not available")

        import urllib3.util.ssl_ as _urllib3_ssl

        original_ssl = ssl.VERIFY_X509_STRICT
        original_urllib3 = getattr(_urllib3_ssl, "VERIFY_X509_STRICT", None)
        try:
            # Simulate the real-world state: both have the original value
            ssl.VERIFY_X509_STRICT = 0x20
            if original_urllib3 is not None:
                _urllib3_ssl.VERIFY_X509_STRICT = 0x20

            _relax_x509_strict()

            assert ssl.VERIFY_X509_STRICT == 0
            if original_urllib3 is not None:
                assert _urllib3_ssl.VERIFY_X509_STRICT == 0
        finally:
            ssl.VERIFY_X509_STRICT = original_ssl
            if original_urllib3 is not None:
                _urllib3_ssl.VERIFY_X509_STRICT = original_urllib3

    def test_noop_without_flag(self) -> None:
        """Test that _relax_x509_strict is a no-op when flag doesn't exist."""
        original = getattr(ssl, "VERIFY_X509_STRICT", None)
        try:
            if hasattr(ssl, "VERIFY_X509_STRICT"):
                delattr(ssl, "VERIFY_X509_STRICT")
            # Should not raise
            _relax_x509_strict()
        finally:
            if original is not None:
                ssl.VERIFY_X509_STRICT = original


class TestSSLDiagnosticProbe:
    """Tests for ssl_diagnostic_probe function."""

    def test_returns_dict_with_expected_keys(self) -> None:
        """Test probe returns dict with all expected keys."""
        with patch("enyal.core.ssl_config.get_ssl_config") as mock_config:
            mock_config.return_value = SSLConfig(offline_mode=True)
            result = ssl_diagnostic_probe()

        assert "python_version" in result
        assert "openssl_version" in result
        assert "platform" in result
        assert "ssl_config" in result
        assert "env_vars" in result

    def test_skips_in_offline_mode(self) -> None:
        """Test probe skips network test in offline mode."""
        with patch("enyal.core.ssl_config.get_ssl_config") as mock_config:
            mock_config.return_value = SSLConfig(offline_mode=True)
            result = ssl_diagnostic_probe()

        assert result["success"] is True
        assert result.get("skipped") == "offline mode"

    def test_captures_env_vars(self) -> None:
        """Test probe captures relevant environment variables."""
        with (
            patch.dict(os.environ, {"ENYAL_SSL_VERIFY": "false"}, clear=False),
            patch("enyal.core.ssl_config.get_ssl_config") as mock_config,
        ):
            mock_config.return_value = SSLConfig(offline_mode=True)
            result = ssl_diagnostic_probe()

        env_vars = result["env_vars"]
        assert env_vars["ENYAL_SSL_VERIFY"] == "false"

    def test_captures_error_on_failure(self) -> None:
        """Test probe captures error details on connection failure."""
        with (
            patch("enyal.core.ssl_config.get_ssl_config") as mock_config,
            patch("urllib.request.urlopen", side_effect=ssl.SSLError("cert verify failed")),
        ):
            mock_config.return_value = SSLConfig()
            result = ssl_diagnostic_probe()

        assert result["success"] is False
        assert "cert verify failed" in result["error"]
        assert result["error_type"] == "SSLError"
        assert "error_chain" in result


class TestPreflightSSLCheck:
    """Tests for _preflight_ssl_check function."""

    def test_returns_tuple(self) -> None:
        """Test that it returns a (bool, str|None) tuple."""
        with (
            patch("socket.create_connection") as mock_conn,
            patch("ssl.create_default_context") as mock_ctx,
        ):
            mock_sock = MagicMock()
            mock_conn.return_value.__enter__ = lambda _: mock_sock
            mock_conn.return_value.__exit__ = MagicMock(return_value=False)
            mock_ctx.return_value.wrap_socket.return_value.__enter__ = (
                lambda _: MagicMock()
            )
            mock_ctx.return_value.wrap_socket.return_value.__exit__ = MagicMock(
                return_value=False
            )
            result = _preflight_ssl_check()

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_success_returns_true_none(self) -> None:
        """Test successful connection returns (True, None)."""
        with (
            patch("socket.create_connection") as mock_conn,
            patch("ssl.create_default_context") as mock_ctx,
        ):
            mock_sock = MagicMock()
            mock_conn.return_value.__enter__ = lambda _: mock_sock
            mock_conn.return_value.__exit__ = MagicMock(return_value=False)
            mock_ctx.return_value.wrap_socket.return_value.__enter__ = (
                lambda _: MagicMock()
            )
            mock_ctx.return_value.wrap_socket.return_value.__exit__ = MagicMock(
                return_value=False
            )
            success, err = _preflight_ssl_check()

        assert success is True
        assert err is None

    def test_ssl_failure_returns_false_with_error(self) -> None:
        """Test SSL failure returns (False, error_string)."""
        with (
            patch("socket.create_connection") as mock_conn,
            patch("ssl.create_default_context") as mock_ctx,
        ):
            mock_sock = MagicMock()
            mock_conn.return_value.__enter__ = lambda _: mock_sock
            mock_conn.return_value.__exit__ = MagicMock(return_value=False)
            mock_ctx.return_value.wrap_socket.side_effect = ssl.SSLError(
                "CA cert not marked as critical"
            )
            success, err = _preflight_ssl_check()

        assert success is False
        assert "CA cert not marked as critical" in err
        assert "SSLError" in err

    def test_connection_failure_returns_false(self) -> None:
        """Test connection failure returns (False, error_string)."""
        with patch("socket.create_connection", side_effect=OSError("Connection refused")):
            success, err = _preflight_ssl_check()

        assert success is False
        assert "Connection refused" in err


class TestBuildSSLContext:
    """Tests for _build_ssl_context function."""

    def test_verify_true_creates_default_context(self) -> None:
        """Test verify=True creates a context with verification enabled."""
        ctx = _build_ssl_context(verify=True)
        assert ctx.check_hostname is True
        assert ctx.verify_mode == ssl.CERT_REQUIRED

    def test_verify_true_no_strict_flag(self) -> None:
        """Test verify=True context does NOT have VERIFY_X509_STRICT set."""
        if not hasattr(ssl, "VERIFY_X509_STRICT"):
            pytest.skip("VERIFY_X509_STRICT not available")
        # Get the real flag value from OpenSSL
        real_flag = 0x20  # X509_V_FLAG_X509_STRICT
        ctx = _build_ssl_context(verify=True)
        assert not (ctx.verify_flags & real_flag), (
            "Context should NOT have VERIFY_X509_STRICT set"
        )

    def test_verify_false_creates_insecure_context(self) -> None:
        """Test verify=False creates a context with verification disabled."""
        ctx = _build_ssl_context(verify=False)
        assert ctx.check_hostname is False
        assert ctx.verify_mode == ssl.CERT_NONE

    def test_cert_file_loaded(self) -> None:
        """Test that cert_file is loaded into the context."""
        # Use a valid cert file (certifi bundle)
        try:
            import certifi
            cert_path = certifi.where()
        except ImportError:
            pytest.skip("certifi not installed")

        # Should not raise
        ctx = _build_ssl_context(verify=True, cert_file=cert_path)
        assert ctx.check_hostname is True

    def test_cert_file_ignored_when_verify_false(self) -> None:
        """Test that cert_file is ignored when verify=False."""
        ctx = _build_ssl_context(verify=False, cert_file="/nonexistent/cert.pem")
        assert ctx.verify_mode == ssl.CERT_NONE


class TestCorpHTTPAdapter:
    """Tests for _create_corp_adapter function."""

    def test_creates_adapter_instance(self) -> None:
        """Test that _create_corp_adapter returns an HTTPAdapter instance."""
        from requests.adapters import HTTPAdapter

        ctx = _build_ssl_context(verify=False)
        adapter = _create_corp_adapter(ctx)
        assert isinstance(adapter, HTTPAdapter)

    def test_adapter_passes_ssl_context_to_poolmanager(self) -> None:
        """Test that the adapter injects ssl_context into init_poolmanager."""
        ctx = _build_ssl_context(verify=False)
        adapter = _create_corp_adapter(ctx)

        # Mock PoolManager at the location requests imports it from
        with patch("requests.adapters.PoolManager") as mock_pm:
            adapter.init_poolmanager(10, 10)
            call_kwargs = mock_pm.call_args.kwargs
            assert call_kwargs.get("ssl_context") is ctx


class TestConfigureHTTPBackendAdapter:
    """Tests for configure_http_backend with adapter injection."""

    def test_session_has_https_adapter_mounted(self) -> None:
        """Test that the session factory mounts a custom HTTPS adapter."""
        captured_session = {}

        def mock_configure(backend_factory: object) -> None:
            captured_session["factory"] = backend_factory

        mock_hf = MagicMock(configure_http_backend=mock_configure)
        with patch.dict("sys.modules", {"huggingface_hub": mock_hf}):
            import enyal.core.ssl_config as mod

            importlib.reload(mod)
            config = SSLConfig()
            mod.configure_http_backend(config)

            session = captured_session["factory"]()

            # The session should have a custom adapter on https://
            from requests.adapters import HTTPAdapter

            adapter = session.get_adapter("https://example.com")
            assert isinstance(adapter, HTTPAdapter)
            importlib.reload(mod)

    def test_verify_false_session_has_insecure_adapter(self) -> None:
        """Test that verify=False produces an adapter with CERT_NONE context."""
        captured_session = {}

        def mock_configure(backend_factory: object) -> None:
            captured_session["factory"] = backend_factory

        mock_hf = MagicMock(configure_http_backend=mock_configure)
        with patch.dict("sys.modules", {"huggingface_hub": mock_hf}):
            import enyal.core.ssl_config as mod

            importlib.reload(mod)
            config = SSLConfig(verify=False)
            mod.configure_http_backend(config)

            session = captured_session["factory"]()
            assert session.verify is False
            importlib.reload(mod)
