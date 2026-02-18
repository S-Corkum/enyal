"""SSL and network configuration for corporate environments.

This module handles SSL certificate configuration for users in corporate networks
with SSL inspection (e.g., Zscaler, BlueCoat, corporate proxies) that inject
enterprise CA certificates into the SSL chain.

Environment Variables:
    ENYAL_SSL_CERT_FILE: Path to corporate CA certificate or bundle file.
        If this contains only the corporate CA cert(s), it will be automatically
        combined with the standard CA bundle (via certifi) so that both corporate
        and public HTTPS connections work correctly.
    ENYAL_SSL_VERIFY: Enable/disable SSL verification (default: "true")
    ENYAL_MODEL_PATH: Local path to pre-downloaded model
    ENYAL_OFFLINE_MODE: Force offline-only operation (default: "false")
    ENYAL_HF_ENDPOINT: Custom HuggingFace Hub endpoint URL (e.g., Artifactory proxy)
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

import hashlib
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

    # Custom HuggingFace Hub endpoint (e.g., Artifactory proxy)
    hf_endpoint: str | None = None


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


def _count_pem_certs(content: str) -> int:
    """Count the number of PEM certificates in a string."""
    return content.count("-----BEGIN CERTIFICATE-----")


def _is_der_encoded(file_path: str) -> bool:
    """Check if a certificate file is DER-encoded (binary) rather than PEM.

    DER files are binary and won't contain PEM text markers. Common extensions
    are .crt, .cer, .der. This checks the actual content, not the extension.
    """
    try:
        with open(file_path, "rb") as f:
            header = f.read(2)
        # DER-encoded certificates start with 0x30 (ASN.1 SEQUENCE tag)
        return len(header) >= 2 and header[0] == 0x30
    except OSError:
        return False


def _convert_der_to_pem(der_path: str) -> str | None:
    """Convert a DER-encoded certificate to PEM format.

    Args:
        der_path: Path to DER-encoded certificate file.

    Returns:
        PEM-encoded certificate string, or None if conversion fails.
    """
    import ssl

    try:
        with open(der_path, "rb") as f:
            der_data = f.read()

        # Validate and convert using ssl module
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.load_verify_locations(cadata=ssl.DER_cert_to_PEM_cert(der_data))

        return ssl.DER_cert_to_PEM_cert(der_data)
    except Exception as e:
        logger.debug(f"DER-to-PEM conversion failed for {der_path}: {e}")
        return None


def _ensure_pem_format(cert_file: str) -> str:
    """Ensure a certificate file is in PEM format.

    If the file is DER-encoded (.crt, .cer, .der), convert it to PEM and
    write it alongside the original file.

    Args:
        cert_file: Path to certificate file (PEM or DER).

    Returns:
        Path to a PEM-format certificate file.
    """
    # Quick check: does it already have PEM content?
    try:
        with open(cert_file, errors="replace") as f:
            content = f.read(100)
        if "-----BEGIN" in content:
            return cert_file
    except OSError:
        return cert_file

    # Check if it's DER-encoded
    if not _is_der_encoded(cert_file):
        return cert_file

    logger.info(f"Detected DER-encoded certificate: {cert_file}")

    pem_content = _convert_der_to_pem(cert_file)
    if pem_content is None:
        logger.warning(
            f"Could not convert DER certificate to PEM: {cert_file}. "
            "If this is a .crt file, it may need to be in PEM format. "
            "Convert with: openssl x509 -inform DER -in cert.crt -out cert.pem"
        )
        return cert_file

    # Write PEM version next to original
    pem_path = Path(cert_file).with_suffix(".pem")

    # If .pem already exists, use enyal data dir instead
    if pem_path.exists() and str(pem_path) != cert_file:
        enyal_dir = Path(
            os.environ.get("ENYAL_DB_PATH", "~/.enyal/context.db")
        ).expanduser().parent
        enyal_dir.mkdir(parents=True, exist_ok=True)
        pem_path = enyal_dir / f"converted_{Path(cert_file).stem}.pem"

    try:
        pem_path.write_text(pem_content)
        logger.info(f"Converted DER to PEM: {cert_file} -> {pem_path}")
        return str(pem_path)
    except OSError as e:
        logger.warning(f"Could not write PEM file {pem_path}: {e}")
        return cert_file


def _get_certifi_bundle() -> str | None:
    """Get the certifi CA bundle path, if available."""
    try:
        import certifi

        path = certifi.where()
        if os.path.isfile(path):
            return path
    except ImportError:
        pass
    return None


def _ensure_combined_cert_bundle(cert_file: str) -> str:
    """Ensure the cert file includes standard CA roots for full HTTPS coverage.

    Corporate environments with SSL inspection typically provide only the
    corporate CA certificate. When this is used as the sole trust store
    (via REQUESTS_CA_BUNDLE), connections to non-proxied endpoints (CDNs,
    model download mirrors) will fail because the standard root CAs are missing.

    This function detects small cert files (fewer than 5 certificates) and
    automatically combines them with the certifi CA bundle, writing the result
    to ~/.enyal/combined_ca_bundle.pem. The combined bundle is cached and only
    regenerated when the source cert file changes.

    Args:
        cert_file: Path to the user's certificate file.

    Returns:
        Path to the combined bundle, or the original cert_file if combining
        is not needed or not possible.
    """
    try:
        with open(cert_file, errors="replace") as f:
            user_cert_content = f.read()
    except OSError as e:
        logger.warning(f"Cannot read cert file {cert_file}: {e}")
        return cert_file

    user_cert_count = _count_pem_certs(user_cert_content)

    # If the file already has many certs, it's likely a complete bundle
    if user_cert_count >= 5:
        logger.debug(
            f"Cert file has {user_cert_count} certificates, "
            "treating as complete bundle"
        )
        return cert_file

    if user_cert_count == 0:
        logger.warning(f"Cert file contains no PEM certificates: {cert_file}")
        return cert_file

    # Try to get certifi bundle for combining
    certifi_path = _get_certifi_bundle()
    if certifi_path is None:
        # Try system CA bundle as fallback
        certifi_path = _find_system_ca_bundle()

    if certifi_path is None:
        logger.warning(
            f"Cert file has only {user_cert_count} certificate(s) but no "
            "standard CA bundle found (certifi not installed, no system bundle). "
            "HTTPS connections to non-corporate endpoints may fail."
        )
        return cert_file

    # Compute hash of user cert to detect changes
    cert_hash = hashlib.sha256(user_cert_content.encode()).hexdigest()[:12]

    # Write combined bundle to enyal data directory
    enyal_dir = Path(
        os.environ.get("ENYAL_DB_PATH", "~/.enyal/context.db")
    ).expanduser().parent
    enyal_dir.mkdir(parents=True, exist_ok=True)

    combined_path = enyal_dir / "combined_ca_bundle.pem"
    hash_path = enyal_dir / "combined_ca_bundle.hash"

    # Check if cached bundle is still valid
    if combined_path.exists() and hash_path.exists():
        try:
            cached_hash = hash_path.read_text().strip()
            if cached_hash == cert_hash:
                logger.debug("Using cached combined CA bundle")
                return str(combined_path)
        except OSError:
            pass

    # Build combined bundle: certifi roots + user's corporate cert(s)
    try:
        with open(certifi_path, errors="replace") as f:
            base_bundle = f.read()

        base_cert_count = _count_pem_certs(base_bundle)

        # Ensure newline separation between bundles
        if not base_bundle.endswith("\n"):
            base_bundle += "\n"

        combined = base_bundle + "\n# Corporate CA certificate(s)\n" + user_cert_content

        combined_path.write_text(combined)
        hash_path.write_text(cert_hash)

        combined_cert_count = _count_pem_certs(combined)
        logger.info(
            f"Created combined CA bundle: {combined_path} "
            f"({base_cert_count} standard + {user_cert_count} corporate = "
            f"{combined_cert_count} total certificates)"
        )
        return str(combined_path)

    except OSError as e:
        logger.warning(f"Failed to create combined CA bundle: {e}")
        return cert_file


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
        ENYAL_HF_ENDPOINT: Custom HuggingFace Hub endpoint URL (e.g., Artifactory proxy)
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
    if cert_file:
        cert_file = os.path.expanduser(cert_file)

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

    # Custom HuggingFace Hub endpoint (e.g., Artifactory proxy)
    hf_endpoint = os.environ.get("ENYAL_HF_ENDPOINT")
    if hf_endpoint:
        hf_endpoint = hf_endpoint.rstrip("/")

    return SSLConfig(
        cert_file=cert_file,
        verify=verify,
        model_path=model_path,
        offline_mode=offline_mode,
        hf_home=hf_home,
        hf_endpoint=hf_endpoint,
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

        # Handle DER-encoded (.crt, .cer, .der) certificates by converting to PEM.
        # requests and urllib3 require PEM format.
        config.cert_file = _ensure_pem_format(config.cert_file)

        # Auto-combine with standard CA roots if the cert file appears incomplete.
        # This handles the common case where users provide only their corporate CA
        # cert, which would break connections to non-proxied endpoints.
        effective_cert = _ensure_combined_cert_bundle(config.cert_file)
        if effective_cert != config.cert_file:
            logger.info(
                f"Using combined CA bundle: {effective_cert} "
                f"(original: {config.cert_file})"
            )
            # Update config so downstream consumers (configure_http_backend) also
            # use the combined bundle
            config.cert_file = effective_cert
        else:
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

    # Set custom HuggingFace Hub endpoint (e.g., Artifactory proxy)
    if config.hf_endpoint:
        os.environ["HF_ENDPOINT"] = config.hf_endpoint
        # Xet storage bypasses HTTP proxies entirely, so disable it
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        # First fetch through Artifactory can be slow; set generous defaults
        # but allow user overrides via env vars
        os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
        logger.info(f"Using custom HF endpoint: {config.hf_endpoint}")


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
    if config.cert_file:
        logger.debug(
            f"Configured huggingface_hub HTTP backend with cert: {config.cert_file}"
        )
    elif not config.verify:
        logger.debug("Configured huggingface_hub HTTP backend with SSL verify=False")
    else:
        logger.debug("Configured huggingface_hub HTTP backend with system defaults")


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
        "hf_endpoint": config.hf_endpoint,
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
