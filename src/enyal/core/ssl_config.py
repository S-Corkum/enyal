"""SSL and network configuration for corporate environments.

This module handles SSL certificate configuration for users in corporate networks
with SSL inspection (e.g., Zscaler, BlueCoat, corporate proxies) that inject
enterprise CA certificates into the SSL chain.

Environment Variables:
    ENYAL_SSL_CERT_FILE: Path to corporate CA bundle file
    ENYAL_SSL_VERIFY: Enable/disable SSL verification (default: "true")
    ENYAL_MODEL_PATH: Local path to pre-downloaded model
    ENYAL_OFFLINE_MODE: Force offline-only operation (default: "false")
    HF_HOME: Hugging Face cache directory (default: ~/.cache/huggingface)

Platform-specific certificate locations:
    - macOS: /etc/ssl/cert.pem or Keychain certificates
    - Linux: /etc/ssl/certs/ca-certificates.crt or /etc/pki/tls/certs/ca-bundle.crt
    - Windows: Uses certifi bundle by default

Security Notes:
    - Disabling SSL verification is NOT recommended and will produce warnings
    - Prefer using ENYAL_SSL_CERT_FILE to specify corporate CA bundle
    - Use ENYAL_OFFLINE_MODE with pre-downloaded models for air-gapped environments
"""

import logging
import os
import platform
import warnings
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Common certificate bundle locations by platform
PLATFORM_CA_BUNDLES: dict[str, list[str]] = {
    "Darwin": [
        "/etc/ssl/cert.pem",
        "/usr/local/etc/openssl/cert.pem",
        "/usr/local/etc/openssl@1.1/cert.pem",
    ],
    "Linux": [
        "/etc/ssl/certs/ca-certificates.crt",  # Debian/Ubuntu
        "/etc/pki/tls/certs/ca-bundle.crt",  # RHEL/CentOS/Fedora
        "/etc/ssl/ca-bundle.pem",  # OpenSUSE
        "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem",  # RHEL/CentOS 7+
    ],
    "Windows": [],  # Uses certifi by default
}


@dataclass
class SSLConfig:
    """SSL configuration settings."""

    # Path to CA certificate bundle (None = use system default)
    cert_file: str | None = None

    # Whether to verify SSL certificates
    verify: bool = True

    # Local model path (bypasses network entirely)
    model_path: str | None = None

    # Offline mode (fail instead of attempting network calls)
    offline_mode: bool = False

    # Hugging Face cache directory
    hf_home: str | None = None


def _parse_bool_env(key: str, default: bool = True) -> bool:
    """Parse a boolean environment variable."""
    value = os.environ.get(key, "").lower()
    if not value:
        return default
    return value in ("true", "1", "yes", "on")


def _find_system_ca_bundle() -> str | None:
    """Find the system CA certificate bundle."""
    system = platform.system()
    candidates = PLATFORM_CA_BUNDLES.get(system, [])

    for path in candidates:
        if os.path.isfile(path):
            return path

    return None


def get_ssl_config() -> SSLConfig:
    """
    Get SSL configuration from environment variables.

    Returns:
        SSLConfig with settings from environment.

    Environment Variables:
        ENYAL_SSL_CERT_FILE: Path to CA certificate bundle
        ENYAL_SSL_VERIFY: "true" or "false" (default: "true")
        ENYAL_MODEL_PATH: Local path to pre-downloaded model
        ENYAL_OFFLINE_MODE: "true" or "false" (default: "false")
        HF_HOME: Hugging Face cache directory
        REQUESTS_CA_BUNDLE: Fallback CA bundle (standard requests env var)
        SSL_CERT_FILE: Fallback CA bundle (standard Python env var)
    """
    # Check for CA bundle in priority order
    cert_file = (
        os.environ.get("ENYAL_SSL_CERT_FILE")
        or os.environ.get("REQUESTS_CA_BUNDLE")
        or os.environ.get("SSL_CERT_FILE")
    )

    # Parse SSL verification setting
    verify = _parse_bool_env("ENYAL_SSL_VERIFY", default=True)

    # Check for local model path
    model_path = os.environ.get("ENYAL_MODEL_PATH")
    if model_path:
        model_path = os.path.expanduser(model_path)
        if not os.path.isdir(model_path):
            logger.warning(f"ENYAL_MODEL_PATH does not exist: {model_path}")
            model_path = None

    # Check offline mode
    offline_mode = _parse_bool_env("ENYAL_OFFLINE_MODE", default=False)

    # Hugging Face home directory
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        hf_home = os.path.expanduser(hf_home)

    return SSLConfig(
        cert_file=cert_file,
        verify=verify,
        model_path=model_path,
        offline_mode=offline_mode,
        hf_home=hf_home,
    )


def configure_ssl_environment(config: SSLConfig | None = None) -> None:
    """
    Configure SSL environment variables for huggingface_hub and requests.

    This should be called BEFORE importing sentence_transformers or
    any library that makes HTTP requests to Hugging Face Hub.

    Args:
        config: SSLConfig to apply. If None, reads from environment.

    Warning:
        Disabling SSL verification is insecure and should only be used
        as a last resort in controlled environments.
    """
    if config is None:
        config = get_ssl_config()

    # Handle SSL verification
    if not config.verify:
        warnings.warn(
            "SSL verification is disabled. This is insecure and should only be used "
            "as a last resort. Consider using ENYAL_SSL_CERT_FILE to specify your "
            "corporate CA bundle instead.",
            UserWarning,
            stacklevel=2,
        )
        logger.warning("SSL verification disabled - connections are NOT secure")

    # Set CA bundle if specified
    if config.cert_file:
        if not os.path.isfile(config.cert_file):
            logger.error(f"CA bundle file not found: {config.cert_file}")
            raise FileNotFoundError(f"CA bundle file not found: {config.cert_file}")

        logger.info(f"Using CA bundle: {config.cert_file}")
        os.environ["REQUESTS_CA_BUNDLE"] = config.cert_file
        os.environ["SSL_CERT_FILE"] = config.cert_file
        os.environ["CURL_CA_BUNDLE"] = config.cert_file

    # Set HF_HOME if specified
    if config.hf_home:
        os.environ["HF_HOME"] = config.hf_home
        logger.info(f"Hugging Face cache directory: {config.hf_home}")

    # Set offline mode for huggingface_hub
    if config.offline_mode:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        logger.info("Offline mode enabled - no network calls will be made")


def configure_http_backend(config: SSLConfig | None = None) -> None:
    """
    Configure the HTTP backend for huggingface_hub with custom SSL settings.

    This provides more fine-grained control than environment variables by
    configuring the requests Session directly.

    Args:
        config: SSLConfig to apply. If None, reads from environment.

    Note:
        This function should be called BEFORE any huggingface_hub operations.
        It configures a custom HTTP backend factory that creates sessions
        with the appropriate SSL settings.
    """
    if config is None:
        config = get_ssl_config()

    try:
        from huggingface_hub import configure_http_backend as hf_configure_http_backend
    except ImportError:
        logger.debug("huggingface_hub not available, skipping HTTP backend configuration")
        return

    import requests

    def create_session() -> requests.Session:
        """Create a requests session with custom SSL configuration."""
        session = requests.Session()

        # Configure SSL verification
        if config.cert_file:
            session.verify = config.cert_file
        elif not config.verify:
            session.verify = False
        # else: use default (True with system certs)

        return session

    hf_configure_http_backend(backend_factory=create_session)
    logger.debug("Configured huggingface_hub HTTP backend with custom SSL settings")


def get_model_path(default_name: str = "all-MiniLM-L6-v2") -> str:
    """
    Get the model path to use for SentenceTransformer.

    Returns either:
    - Local path if ENYAL_MODEL_PATH is set and valid
    - The model name for download from Hugging Face Hub

    Args:
        default_name: Default model name if no local path is set.

    Returns:
        Path to local model or model name for download.
    """
    config = get_ssl_config()

    if config.model_path and os.path.isdir(config.model_path):
        logger.info(f"Using local model: {config.model_path}")
        return config.model_path

    if config.offline_mode:
        # In offline mode, check if model is cached
        cache_dir = config.hf_home or os.path.expanduser("~/.cache/huggingface")
        cached_model = Path(cache_dir) / "hub" / f"models--sentence-transformers--{default_name}"
        if cached_model.exists():
            logger.info(f"Using cached model: {cached_model}")
            # Return the model name - sentence_transformers will find it in cache
            return default_name
        else:
            raise RuntimeError(
                f"Offline mode is enabled but model '{default_name}' is not cached. "
                f"Expected cache location: {cached_model}\n"
                "To download the model, run: enyal model download"
            )

    return default_name


def check_ssl_health() -> dict[str, str | bool | None]:
    """
    Check SSL configuration health and connectivity.

    Returns:
        Dictionary with health check results.
    """
    config = get_ssl_config()

    result: dict[str, str | bool | None] = {
        "ssl_verify": config.verify,
        "cert_file": config.cert_file,
        "cert_file_exists": config.cert_file is not None and os.path.isfile(config.cert_file),
        "model_path": config.model_path,
        "model_path_exists": config.model_path is not None and os.path.isdir(config.model_path),
        "offline_mode": config.offline_mode,
        "hf_home": config.hf_home,
        "system_ca_bundle": _find_system_ca_bundle(),
    }

    # Check if we can import huggingface_hub
    try:
        import huggingface_hub

        result["huggingface_hub_version"] = huggingface_hub.__version__
    except ImportError:
        result["huggingface_hub_version"] = None

    # Check if we can import sentence_transformers
    try:
        import sentence_transformers

        result["sentence_transformers_version"] = sentence_transformers.__version__
    except ImportError:
        result["sentence_transformers_version"] = None

    return result


def download_model(
    model_name: str = "all-MiniLM-L6-v2",
    cache_dir: str | None = None,
) -> str:
    """
    Download the embedding model for offline use.

    This function should be called with proper SSL configuration when
    network access is available, to cache the model for later offline use.

    Args:
        model_name: Name of the sentence-transformers model.
        cache_dir: Custom cache directory (default: HF_HOME or ~/.cache/huggingface).

    Returns:
        Path to the downloaded model.

    Raises:
        SSLError: If SSL verification fails (configure ENYAL_SSL_CERT_FILE).
        ConnectionError: If network is unavailable.
    """
    # Configure SSL before importing sentence_transformers
    config = get_ssl_config()

    # Don't allow download in offline mode
    if config.offline_mode:
        raise RuntimeError(
            "Cannot download model in offline mode. "
            "Set ENYAL_OFFLINE_MODE=false or unset it to allow downloads."
        )

    configure_ssl_environment(config)
    configure_http_backend(config)

    logger.info(f"Downloading model: {model_name}")

    from sentence_transformers import SentenceTransformer

    # Download the model (this triggers the actual download)
    model = SentenceTransformer(model_name, cache_folder=cache_dir)

    # Get the actual path where it was saved
    model_path = model._model_card_vars.get("model_path", model_name)
    logger.info(f"Model downloaded successfully: {model_path}")

    return str(model_path)


def verify_model(model_path: str | None = None) -> bool:
    """
    Verify that the model can be loaded successfully.

    Args:
        model_path: Path to model or model name. If None, uses default.

    Returns:
        True if model loads successfully, False otherwise.
    """
    try:
        path = model_path or get_model_path()
        logger.info(f"Verifying model: {path}")

        # Configure SSL for download if needed
        config = get_ssl_config()
        configure_ssl_environment(config)
        configure_http_backend(config)

        from sentence_transformers import SentenceTransformer

        # Try to load the model
        model = SentenceTransformer(path)

        # Try a simple encode to verify it works
        _ = model.encode("test", convert_to_numpy=True)

        logger.info("Model verification successful")
        return True

    except Exception as e:
        logger.error(f"Model verification failed: {e}")
        return False
