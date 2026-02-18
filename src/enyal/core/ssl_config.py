"""SSL and network configuration for corporate environments.

This module handles SSL certificate configuration for users in corporate networks
with SSL inspection (e.g., Netskope, Zscaler, BlueCoat, corporate proxies) that
inject enterprise CA certificates into the SSL chain.

SSL Resolution Order:
    1. ENYAL_SSL_VERIFY=false  -> Disable ALL SSL verification (nuclear option)
    2. ENYAL_SSL_CERT_FILE     -> Use explicit certificate bundle
    3. truststore package       -> Use OS native trust store (recommended for corporate)
    4. macOS Keychain export    -> Auto-extract system-trusted certs (macOS only)
    5. certifi (default)        -> Python's bundled CA certificates

Environment Variables:
    ENYAL_SSL_CERT_FILE: Path to corporate CA certificate or bundle file.
        If this contains only the corporate CA cert(s), it will be automatically
        combined with the standard CA bundle (via certifi) so that both corporate
        and public HTTPS connections work correctly.
    ENYAL_SSL_VERIFY: Enable/disable SSL verification (default: "true")
    ENYAL_SSL_TRUST_SYSTEM: Use OS native trust store when no cert file specified
        (default: "true"). Tries truststore package first, then macOS Keychain export.
    ENYAL_MODEL_PATH: Local path to pre-downloaded model
    ENYAL_OFFLINE_MODE: Force offline-only operation (default: "false")
    ENYAL_HF_ENDPOINT: Custom HuggingFace Hub endpoint URL (e.g., Artifactory proxy)
    HF_HOME: Hugging Face cache directory (default: ~/.cache/huggingface)

Platform-specific certificate locations:
    - macOS: /etc/ssl/cert.pem or Keychain certificates (auto-detected)
    - Linux: /etc/ssl/certs/ca-certificates.crt or /etc/pki/tls/certs/ca-bundle.crt
    - Windows: Uses certifi bundle by default

Corporate SSL Notes:
    For corporate environments with SSL inspection (Netskope, Zscaler, etc.):
    - RECOMMENDED: pip install truststore (auto-uses OS trust store)
    - ALTERNATIVE: Set ENYAL_SSL_CERT_FILE to your corporate CA bundle
    - LAST RESORT: Set ENYAL_SSL_VERIFY=false to disable verification

Security Notes:
    - Disabling SSL verification is NOT recommended and will produce warnings
    - Prefer using ENYAL_SSL_CERT_FILE to specify corporate CA bundle
    - Use ENYAL_OFFLINE_MODE with pre-downloaded models for air-gapped environments
"""

import hashlib
import logging
import os
import platform
import subprocess
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


def _try_inject_truststore() -> bool:
    """Try to inject OS native trust store via the ``truststore`` package.

    When ``truststore`` is installed, it monkey-patches Python's ``ssl`` module
    so that ``ssl.SSLContext`` uses the operating system's certificate trust
    store instead of certifi's bundled CA certificates.  This means that any
    CA certificate trusted by the OS (including corporate proxies like Netskope,
    Zscaler, BlueCoat, etc.) will automatically be trusted by Python.

    Returns:
        True if truststore was successfully injected, False otherwise.
    """
    try:
        import truststore  # type: ignore[import-untyped,unused-ignore]

        truststore.inject_into_ssl()
        logger.info("SSL: Using OS native trust store (truststore package)")
        return True
    except ImportError:
        logger.debug("truststore package not installed (pip install truststore)")
        return False
    except Exception as e:
        logger.warning(f"truststore injection failed: {e}")
        return False


def _export_macos_system_certs() -> str | None:
    """Export macOS system certificates to a PEM bundle for use by requests/urllib3.

    On macOS, corporate CA certificates (Netskope, Zscaler, etc.) are installed
    into the System Keychain by MDM or IT policy.  However, Python's ``requests``
    library uses ``certifi``'s CA bundle, which does **not** include these
    corporate certificates.  This causes SSL errors when corporate proxies
    perform SSL inspection.

    This function exports all trusted certificates from the macOS Keychains,
    combines them with the certifi bundle, and caches the result.  The combined
    bundle includes both standard root CAs and corporate proxy certificates.

    Keychains exported:
        - /System/Library/Keychains/SystemRootCertificates.keychain (Apple roots)
        - /Library/Keychains/System.keychain (system-wide, MDM-deployed certs)

    Returns:
        Path to the combined PEM bundle, or None if export fails.
    """
    if platform.system() != "Darwin":
        return None

    enyal_dir = Path(
        os.environ.get("ENYAL_DB_PATH", "~/.enyal/context.db")
    ).expanduser().parent
    enyal_dir.mkdir(parents=True, exist_ok=True)
    system_certs_path = enyal_dir / "macos_system_certs.pem"

    # Re-export if file doesn't exist or is older than 24 hours
    if system_certs_path.exists():
        import time

        age_hours = (time.time() - system_certs_path.stat().st_mtime) / 3600
        if age_hours < 24:
            cert_count = _count_pem_certs(system_certs_path.read_text(errors="replace"))
            if cert_count > 0:
                logger.debug(
                    f"Using cached macOS system certificates "
                    f"({cert_count} certs, {age_hours:.1f}h old)"
                )
                return str(system_certs_path)

    # Export certificates from macOS Keychains
    keychains = [
        "/System/Library/Keychains/SystemRootCertificates.keychain",
        "/Library/Keychains/System.keychain",
    ]

    # Filter to keychains that actually exist
    existing_keychains = [k for k in keychains if os.path.exists(k)]
    if not existing_keychains:
        logger.debug("No macOS system keychains found")
        return None

    try:
        result = subprocess.run(
            ["security", "find-certificate", "-a", "-p", *existing_keychains],
            capture_output=True,
            text=True,
            timeout=15,
        )

        if result.returncode != 0:
            logger.debug(f"security find-certificate failed: {result.stderr}")
            return None

        keychain_certs = result.stdout
        if "-----BEGIN CERTIFICATE-----" not in keychain_certs:
            logger.debug("No PEM certificates found in macOS Keychains")
            return None

        keychain_cert_count = _count_pem_certs(keychain_certs)

        # Combine with certifi for maximum coverage
        certifi_path = _get_certifi_bundle()
        if certifi_path:
            with open(certifi_path, errors="replace") as f:
                certifi_content = f.read()

            if not certifi_content.endswith("\n"):
                certifi_content += "\n"

            combined = (
                certifi_content
                + "\n# macOS System Keychain certificates\n"
                + keychain_certs
            )
        else:
            combined = keychain_certs

        system_certs_path.write_text(combined)
        total_count = _count_pem_certs(combined)
        logger.info(
            f"Exported macOS system certificates: {system_certs_path} "
            f"({keychain_cert_count} from Keychain, {total_count} total)"
        )
        return str(system_certs_path)

    except subprocess.TimeoutExpired:
        logger.warning("Timed out exporting macOS Keychain certificates")
        return None
    except OSError as e:
        logger.debug(f"Failed to export macOS system certificates: {e}")
        return None


def _is_ssl_error(exc: BaseException) -> bool:
    """Check if an exception is SSL-related.

    Detects SSL errors from any layer: ssl module, urllib3, requests, or
    huggingface_hub.  This is used by the auto-recovery logic to decide
    whether to retry with SSL disabled.

    Args:
        exc: The exception to check.

    Returns:
        True if the exception is SSL-related.
    """
    import ssl

    # Direct SSL module errors
    if isinstance(exc, ssl.SSLError):
        return True

    # requests wraps SSL errors
    try:
        from requests.exceptions import SSLError as RequestsSSLError

        if isinstance(exc, RequestsSSLError):
            return True
    except ImportError:
        pass

    # urllib3 wraps SSL errors
    try:
        from urllib3.exceptions import SSLError as Urllib3SSLError

        if isinstance(exc, Urllib3SSLError):
            return True
    except ImportError:
        pass

    # Check the exception chain (SSL errors are often wrapped)
    cause = exc.__cause__ or exc.__context__
    if cause is not None and cause is not exc:
        return _is_ssl_error(cause)

    # Check string representation as last resort (use specific error patterns,
    # not bare keywords like "ssl" which would false-positive on unrelated messages)
    exc_str = str(exc).lower()
    ssl_patterns = [
        "sslerror",
        "ssl error",
        "ssl:",
        "sslcertverificationerror",
        "certificate verify failed",
        "certificate_verify_failed",
        "cert_required",
        "[ssl]",
    ]
    return any(p in exc_str for p in ssl_patterns)


def _disable_ssl_globally() -> None:
    """Disable SSL certificate verification for the entire Python process.

    This is the nuclear option -- it disables SSL verification at every layer:

    1. Python ``ssl`` module: Replaces the default HTTPS context with an
       unverified one, so that ``urllib.request``, ``http.client``, and any
       other stdlib networking that uses ``ssl.create_default_context()``
       will skip certificate validation.

    2. ``urllib3`` SSL context: Monkey-patches ``create_urllib3_context`` to
       lower OpenSSL 3.x security level from 2 to 1.  Security level 2
       rejects connections during the TLS handshake if the server certificate
       uses SHA-1 signatures, small keys, or non-critical Basic Constraints
       (common with corporate SSL inspection proxies like Netskope).  This
       rejection happens *before* certificate verification, so ``verify=False``
       alone cannot prevent it.

    3. ``urllib3`` warnings: Suppresses ``InsecureRequestWarning`` to avoid
       flooding logs with warnings on every HTTPS request.

    4. Environment variables: Sets ``PYTHONHTTPSVERIFY=0`` as a belt-and-
       suspenders measure for any subprocess or library that checks it.

    Warning:
        This should only be used as an absolute last resort when both
        ``truststore`` and ``ENYAL_SSL_CERT_FILE`` are not viable options.
    """
    import ssl

    # ── Layer 1: Python stdlib ──────────────────────────────────────────
    # Replace the default HTTPS context factory so ALL Python SSL connections
    # skip certificate verification.  This covers urllib, http.client, and any
    # library that calls ssl.create_default_context().
    ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore[assignment]  # noqa: SLF001

    # ── Layer 2: urllib3 SSL context ────────────────────────────────────
    # urllib3 2.x creates its own SSLContext via create_urllib3_context(),
    # bypassing ssl._create_default_https_context entirely.  We monkey-patch
    # that factory to lower the OpenSSL security level.
    #
    # OpenSSL 3.x defaults to security level 2 with PROTOCOL_TLS_CLIENT.
    # Level 2 rejects:
    #   - RSA keys < 2048 bits
    #   - SHA-1 in certificate signatures (OpenSSL 3.2+)
    #   - Non-critical Basic Constraints on CA certificates
    # Corporate SSL inspection proxies (Netskope, Zscaler) commonly trigger
    # these rejections.  Level 1 relaxes these requirements.
    try:
        import urllib3.util.ssl_ as _urllib3_ssl

        _original_create_ctx = _urllib3_ssl.create_urllib3_context

        def _permissive_urllib3_context(
            *args: object, **kwargs: object
        ) -> ssl.SSLContext:
            ctx = _original_create_ctx(*args, **kwargs)
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            # Lower security level to allow corporate proxy certs
            try:
                ctx.set_ciphers("DEFAULT:@SECLEVEL=1")
            except ssl.SSLError:
                pass  # Older OpenSSL versions without SECLEVEL support
            return ctx

        _urllib3_ssl.create_urllib3_context = _permissive_urllib3_context  # type: ignore[assignment]
        logger.debug("Patched urllib3 SSL context factory (SECLEVEL=1)")
    except (ImportError, AttributeError) as e:
        logger.debug(f"Could not patch urllib3 SSL context: {e}")

    # ── Layer 3: Suppress warnings ──────────────────────────────────────
    try:
        import urllib3

        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except ImportError:
        pass

    # ── Layer 4: Environment variables ──────────────────────────────────
    # Belt-and-suspenders: env var checked by some libraries / subprocesses
    os.environ["PYTHONHTTPSVERIFY"] = "0"

    logger.warning("SSL verification disabled globally for this process")


def get_ssl_config() -> SSLConfig:
    """
    Get SSL configuration from environment variables.

    Returns:
        SSLConfig with settings from environment.

    Environment Variables:
        ENYAL_SSL_CERT_FILE: Path to CA certificate bundle
        ENYAL_SSL_VERIFY: "true" or "false" (default: "true")
        ENYAL_SSL_TRUST_SYSTEM: "true" or "false" (default: "true")
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

    SSL Resolution Order:
        1. ENYAL_SSL_VERIFY=false  -> Disable ALL SSL verification globally
        2. ENYAL_SSL_CERT_FILE     -> Use explicit certificate bundle
        3. truststore package       -> Use OS native trust store (if installed)
        4. macOS Keychain export    -> Auto-extract system-trusted certs
        5. certifi (default)        -> Python's bundled CA certificates

    Args:
        config: SSLConfig to apply. If None, reads from environment.

    Warning:
        Disabling SSL verification is insecure and should only be used
        as a last resort in controlled environments.
    """
    if config is None:
        config = get_ssl_config()

    # ── APPROACH 1: Nuclear disable ──────────────────────────────────────
    # When verify=False, disable SSL verification at EVERY layer to ensure
    # no code path can still fail on certificate validation.
    if not config.verify:
        warnings.warn(
            "SSL verification is disabled (ENYAL_SSL_VERIFY=false). "
            "This is insecure and should only be used as a last resort. "
            "Consider: pip install truststore (uses OS trust store), or "
            "set ENYAL_SSL_CERT_FILE to your corporate CA bundle.",
            UserWarning,
            stacklevel=2,
        )
        _disable_ssl_globally()

        # Disable Rust-based download paths that bypass Python SSL entirely
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        # Corporate networks need generous timeouts
        os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

        # Continue to handle offline mode, HF home, and endpoint below

    # ── APPROACH 2: Explicit cert file ───────────────────────────────────
    elif config.cert_file:
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
            config.cert_file = effective_cert
        else:
            logger.info(f"Using CA bundle: {config.cert_file}")

        os.environ["REQUESTS_CA_BUNDLE"] = config.cert_file
        os.environ["SSL_CERT_FILE"] = config.cert_file
        os.environ["CURL_CA_BUNDLE"] = config.cert_file

        # Disable Rust-based download paths that bypass Python SSL
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        logger.debug(
            "Disabled HF Xet/hf_transfer (incompatible with custom SSL configuration)"
        )
        os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

    # ── APPROACH 3 & 4: Auto-detect system trust store ───────────────────
    else:
        trust_system = _parse_bool_env("ENYAL_SSL_TRUST_SYSTEM", default=True)
        if trust_system:
            # Try truststore first (cross-platform, preferred)
            if _try_inject_truststore():
                # truststore handles everything via OS trust store.
                # Disable Rust-based paths that bypass Python SSL.
                os.environ["HF_HUB_DISABLE_XET"] = "1"
                os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
                os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
                os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
            else:
                # Fallback: macOS Keychain certificate export
                system_cert_path = _export_macos_system_certs()
                if system_cert_path:
                    config.cert_file = system_cert_path
                    os.environ["REQUESTS_CA_BUNDLE"] = system_cert_path
                    os.environ["SSL_CERT_FILE"] = system_cert_path
                    os.environ["CURL_CA_BUNDLE"] = system_cert_path
                    os.environ["HF_HUB_DISABLE_XET"] = "1"
                    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
                    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
                    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
                # else: fall through to default certifi behavior

    # ── Common configuration (applies regardless of SSL approach) ────────

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

        # Configure SSL verification (verify=False takes highest priority)
        if not config.verify:
            session.verify = False
        elif config.cert_file:
            session.verify = config.cert_file
        # else: use default (True with system certs / truststore)

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

    # Check truststore availability
    truststore_available = False
    truststore_version: str | None = None
    try:
        import truststore  # type: ignore[import-untyped,unused-ignore]

        truststore_available = True
        truststore_version = getattr(truststore, "__version__", "installed")
    except ImportError:
        pass

    result: dict[str, str | bool | None] = {
        "ssl_verify": config.verify,
        "cert_file": config.cert_file,
        "cert_file_exists": config.cert_file is not None and os.path.isfile(config.cert_file),
        "trust_system": _parse_bool_env("ENYAL_SSL_TRUST_SYSTEM", default=True),
        "truststore_available": truststore_available,
        "truststore_version": truststore_version,
        "model_path": config.model_path,
        "model_path_exists": config.model_path is not None and os.path.isdir(config.model_path),
        "offline_mode": config.offline_mode,
        "hf_home": config.hf_home,
        "hf_endpoint": config.hf_endpoint,
        "system_ca_bundle": _find_system_ca_bundle(),
        "platform": platform.system(),
    }

    # macOS-specific: check Keychain availability
    if platform.system() == "Darwin":
        result["macos_system_keychain"] = os.path.exists(
            "/Library/Keychains/System.keychain"
        )

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

    # Report OpenSSL version (critical for diagnosing cert issues)
    try:
        import ssl

        result["openssl_version"] = ssl.OPENSSL_VERSION
    except Exception:
        result["openssl_version"] = None

    return result


def ssl_diagnostic_probe() -> dict[str, object]:
    """Run an SSL connectivity probe and return diagnostic information.

    Attempts a lightweight HTTPS connection to huggingface.co and captures
    the exact error if it fails.  This is used on server startup to detect
    SSL problems early and log actionable diagnostics.

    Returns:
        Dictionary with diagnostic results including:
        - ``success``: Whether the probe connected successfully
        - ``error``: Error message if failed, None if succeeded
        - ``error_type``: Exception class name if failed
        - ``ssl_config``: Current effective SSL configuration
        - ``python_version``: Python version string
        - ``openssl_version``: OpenSSL version string
        - ``env_vars``: Relevant SSL environment variables
    """
    import ssl
    import sys

    result: dict[str, object] = {
        "python_version": sys.version,
        "openssl_version": getattr(ssl, "OPENSSL_VERSION", "unknown"),
        "platform": platform.system(),
        "machine": platform.machine(),
    }

    # Capture relevant env vars (redact cert paths for privacy)
    env_vars: dict[str, str] = {}
    for key in [
        "ENYAL_SSL_VERIFY",
        "ENYAL_SSL_CERT_FILE",
        "ENYAL_SSL_TRUST_SYSTEM",
        "ENYAL_OFFLINE_MODE",
        "ENYAL_MODEL_PATH",
        "ENYAL_HF_ENDPOINT",
        "REQUESTS_CA_BUNDLE",
        "SSL_CERT_FILE",
        "CURL_CA_BUNDLE",
        "HF_HUB_DISABLE_XET",
        "HF_HUB_ENABLE_HF_TRANSFER",
        "PYTHONHTTPSVERIFY",
    ]:
        val = os.environ.get(key)
        if val is not None:
            env_vars[key] = val
    result["env_vars"] = env_vars

    # Capture SSL config
    config = get_ssl_config()
    result["ssl_config"] = {
        "verify": config.verify,
        "cert_file": config.cert_file,
        "offline_mode": config.offline_mode,
        "model_path": config.model_path,
    }

    # Skip probe in offline mode
    if config.offline_mode:
        result["success"] = True
        result["skipped"] = "offline mode"
        return result

    # Attempt HTTPS connection to HuggingFace Hub
    try:
        import urllib.request

        req = urllib.request.Request(
            "https://huggingface.co/api/whoami-v2",
            method="HEAD",
        )
        req.add_header("User-Agent", "enyal-ssl-probe/1.0")
        with urllib.request.urlopen(req, timeout=10):
            pass
        result["success"] = True
        result["error"] = None
        result["error_type"] = None
    except Exception as e:
        result["success"] = False
        result["error"] = str(e)
        result["error_type"] = type(e).__qualname__

        # Capture the full chain for debugging
        chain: list[str] = []
        current: BaseException | None = e
        while current is not None:
            chain.append(f"{type(current).__qualname__}: {current}")
            current = current.__cause__ or current.__context__
            if current is e:
                break  # prevent infinite loop
        result["error_chain"] = chain

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

    # Look up model config to get trust_remote_code and other settings
    from enyal.embeddings.models import MODEL_REGISTRY, ModelConfig

    kwargs: dict[str, object] = {}
    if cache_dir:
        kwargs["cache_folder"] = cache_dir

    if model_name in MODEL_REGISTRY:
        model_config = MODEL_REGISTRY[model_name]
    else:
        model_config = ModelConfig.from_env() if not model_name else None

    trust_remote_code = False
    if model_config and model_config.trust_remote_code:
        trust_remote_code = True
    # Allow env var override
    env_trust = os.environ.get("ENYAL_TRUST_REMOTE_CODE", "")
    if env_trust.lower() == "true":
        trust_remote_code = True

    if trust_remote_code:
        kwargs["trust_remote_code"] = True

    # Download the model (this triggers the actual download)
    model = SentenceTransformer(model_name, **kwargs)

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

        # Look up model config to get trust_remote_code
        from enyal.embeddings.models import MODEL_REGISTRY, ModelConfig

        kwargs: dict[str, object] = {}

        # Check registry by path or by model name
        model_config = MODEL_REGISTRY.get(path)
        if model_config is None:
            # If verifying a local path, check the default config
            model_config = ModelConfig.from_env()

        trust_remote_code = False
        if model_config and model_config.trust_remote_code:
            trust_remote_code = True
        # Allow env var override
        env_trust = os.environ.get("ENYAL_TRUST_REMOTE_CODE", "")
        if env_trust.lower() == "true":
            trust_remote_code = True

        if trust_remote_code:
            kwargs["trust_remote_code"] = True

        # Try to load the model
        model = SentenceTransformer(path, **kwargs)

        # Try a simple encode to verify it works
        _ = model.encode("test", convert_to_numpy=True)

        logger.info("Model verification successful")
        return True

    except Exception as e:
        logger.error(f"Model verification failed: {e}")
        return False
