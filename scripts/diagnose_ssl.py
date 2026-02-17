#!/usr/bin/env python3
"""SSL diagnostic script for corporate environments.

Run on your work MacBook to pinpoint exactly where SSL verification fails.

Usage:
    # Basic usage with your corporate CA cert:
    ENYAL_SSL_CERT_FILE=/path/to/your/corp-cert.pem python scripts/diagnose_ssl.py

    # With custom model name:
    ENYAL_MODEL_NAME=all-MiniLM-L6-v2 ENYAL_SSL_CERT_FILE=/path/to/cert.pem python scripts/diagnose_ssl.py

    # Inside the project venv:
    uv run --directory /path/to/enyal python scripts/diagnose_ssl.py
"""

import contextlib
import os
import ssl
import sys
import tempfile


def header(msg: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def check(label: str, result: bool, detail: str = "") -> None:
    status = "PASS" if result else "FAIL"
    print(f"  [{status}] {label}")
    if detail:
        for line in detail.split("\n"):
            print(f"         {line}")


def count_pem_certs(content: str) -> int:
    return content.count("-----BEGIN CERTIFICATE-----")


def main() -> None:
    print("Enyal SSL Diagnostic Tool")
    print(f"Python: {sys.version}")
    print(f"Platform: {sys.platform}")

    # ── 1. Environment variables ──
    header("1. Environment Variables")
    cert_file = os.environ.get("ENYAL_SSL_CERT_FILE", "")
    if cert_file:
        cert_file = os.path.expanduser(cert_file)
    check("ENYAL_SSL_CERT_FILE is set", bool(cert_file), cert_file or "(not set)")
    check("REQUESTS_CA_BUNDLE", bool(os.environ.get("REQUESTS_CA_BUNDLE")),
          os.environ.get("REQUESTS_CA_BUNDLE", "(not set)"))
    check("SSL_CERT_FILE", bool(os.environ.get("SSL_CERT_FILE")),
          os.environ.get("SSL_CERT_FILE", "(not set)"))

    model_name = os.environ.get("ENYAL_MODEL_NAME", "nomic-ai/nomic-embed-text-v1.5")
    print(f"\n  Model to test: {model_name}")
    print("  (Set ENYAL_MODEL_NAME to override)")

    if not cert_file:
        print("\n  ERROR: ENYAL_SSL_CERT_FILE is not set. Set it and re-run.")
        print("  Example: ENYAL_SSL_CERT_FILE=/path/to/cert.pem python scripts/diagnose_ssl.py")
        sys.exit(1)

    # ── 2. Cert file validation ──
    header("2. Certificate File Validation")
    check("File exists", os.path.isfile(cert_file), cert_file)

    if not os.path.isfile(cert_file):
        print(f"\n  ERROR: Cert file not found at: {cert_file}")
        sys.exit(1)

    file_size = os.path.getsize(cert_file)
    check("File is not empty", file_size > 0, f"{file_size} bytes")

    with open(cert_file, errors="replace") as f:
        user_cert_content = f.read()

    cert_count = count_pem_certs(user_cert_content)
    check("Contains PEM certificates", cert_count > 0, f"Found {cert_count} certificate(s)")

    # ── 3. Auto-combine with certifi ──
    header("3. Certificate Bundle Combination")
    combined_cert_file = cert_file  # default to original

    try:
        import certifi
        certifi_path = certifi.where()
        check("certifi available", True, f"v{certifi.__version__} at {certifi_path}")

        with open(certifi_path, errors="replace") as f:
            certifi_content = f.read()
        certifi_count = count_pem_certs(certifi_content)
        check("certifi bundle", True, f"{certifi_count} certificates")

        if cert_count < 5:
            print(f"\n  Your cert file has only {cert_count} certificate(s).")
            print("  Creating combined bundle (corp cert + certifi roots)...")

            if not certifi_content.endswith("\n"):
                certifi_content += "\n"
            combined = certifi_content + "\n# Corporate CA certificate(s)\n" + user_cert_content
            combined_count = count_pem_certs(combined)

            # Write to temp file for testing
            with tempfile.NamedTemporaryFile(mode="w", suffix=".pem", delete=False) as f:
                f.write(combined)
                combined_cert_file = f.name

            check("Combined bundle created", True,
                  f"{certifi_count} standard + {cert_count} corporate = {combined_count} total\n"
                  f"Written to: {combined_cert_file}")
        else:
            print(f"\n  Cert file has {cert_count} certificates - treating as complete bundle.")
    except ImportError:
        check("certifi available", False, "Not installed - cannot create combined bundle")

    # ── 4. Python ssl module ──
    header("4. Python SSL Module")
    for label, test_cert in [("original cert", cert_file), ("combined cert", combined_cert_file)]:
        if test_cert == cert_file and label == "combined cert":
            continue  # skip if no combined cert was created
        try:
            ctx = ssl.create_default_context(cafile=test_cert)
            check(f"ssl.create_default_context ({label})", True)
        except Exception as e:
            check(f"ssl.create_default_context ({label})", False, str(e))

        try:
            ctx = ssl.create_default_context()
            ctx.load_verify_locations(test_cert)
            check(f"ctx.load_verify_locations ({label})", True)
        except Exception as e:
            check(f"ctx.load_verify_locations ({label})", False, str(e))

    # ── 5. requests library ──
    header("5. requests Library")
    try:
        import requests
        check("requests importable", True, f"v{requests.__version__}")

        # Test with combined cert file (or original if no combined)
        for label, test_cert in [("original", cert_file), ("combined", combined_cert_file)]:
            if test_cert == cert_file and label == "combined":
                continue
            try:
                resp = requests.get(
                    "https://huggingface.co/api/models/sentence-transformers/all-MiniLM-L6-v2",
                    verify=test_cert, timeout=10)
                check(f"HuggingFace API ({label} cert)", resp.ok, f"HTTP {resp.status_code}")
            except requests.exceptions.SSLError as e:
                check(f"HuggingFace API ({label} cert)", False, f"SSL Error: {str(e)[:200]}")
            except Exception as e:
                check(f"HuggingFace API ({label} cert)", False, str(e)[:200])

        # Test with REQUESTS_CA_BUNDLE env var
        os.environ["REQUESTS_CA_BUNDLE"] = combined_cert_file
        os.environ["SSL_CERT_FILE"] = combined_cert_file
        os.environ["CURL_CA_BUNDLE"] = combined_cert_file
        try:
            resp = requests.get(
                "https://huggingface.co/api/models/sentence-transformers/all-MiniLM-L6-v2",
                timeout=10)
            check("HuggingFace API (via env vars)", resp.ok, f"HTTP {resp.status_code}")
        except requests.exceptions.SSLError as e:
            check("HuggingFace API (via env vars)", False, f"SSL Error: {str(e)[:200]}")
        except Exception as e:
            check("HuggingFace API (via env vars)", False, str(e)[:200])

    except ImportError:
        check("requests importable", False, "Not installed")

    # ── 6. huggingface_hub ──
    header("6. huggingface_hub")
    try:
        import huggingface_hub
        check("huggingface_hub importable", True, f"v{huggingface_hub.__version__}")

        # Configure the HTTP backend
        from huggingface_hub import configure_http_backend, get_session

        def factory():
            import requests as req
            s = req.Session()
            s.verify = combined_cert_file
            return s

        configure_http_backend(backend_factory=factory)

        session = get_session()
        check("Session verify setting", True, f"verify={session.verify}")

        # Test model info fetch
        try:
            info = huggingface_hub.model_info(model_name)
            check(f"Model info fetch ({model_name})", True, f"Model: {info.model_id}")
        except Exception as e:
            check(f"Model info fetch ({model_name})", False, str(e)[:200])

    except ImportError:
        check("huggingface_hub importable", False, "Not installed")

    # ── 7. Enyal SSL config module ──
    header("7. Enyal SSL Config Module")
    try:
        from enyal.core.ssl_config import (
            _ensure_combined_cert_bundle,
            configure_ssl_environment,
            get_ssl_config,
        )
        from enyal.core.ssl_config import (
            configure_http_backend as enyal_configure_http_backend,
        )

        # Reset env to test enyal's config
        os.environ["ENYAL_SSL_CERT_FILE"] = cert_file
        config = get_ssl_config()
        check("get_ssl_config()", True,
              f"cert_file={config.cert_file}\nverify={config.verify}\n"
              f"offline_mode={config.offline_mode}")

        # Test auto-combine
        combined = _ensure_combined_cert_bundle(cert_file)
        check("_ensure_combined_cert_bundle()", True,
              f"Result: {combined}\n(original: {cert_file})")

        # Test full configuration
        configure_ssl_environment(config)
        check("configure_ssl_environment()", True,
              f"REQUESTS_CA_BUNDLE={os.environ.get('REQUESTS_CA_BUNDLE', 'not set')}")

        enyal_configure_http_backend(config)
        check("configure_http_backend()", True)

    except ImportError:
        check("enyal.core.ssl_config importable", False,
              "Not installed - run from within the enyal project")
    except Exception as e:
        check("Enyal SSL config", False, str(e)[:300])

    # ── 8. SentenceTransformers model load ──
    header("8. SentenceTransformers Model Load")
    try:
        from sentence_transformers import SentenceTransformer
        check("sentence_transformers importable", True)

        # Determine if model needs trust_remote_code
        trust_remote = model_name in (
            "nomic-ai/nomic-embed-text-v1.5",
        )
        if trust_remote:
            print("  Model requires trust_remote_code=True")

        print("  Attempting model load (this may download the model)...")
        kwargs = {}
        if trust_remote:
            kwargs["trust_remote_code"] = True
        model = SentenceTransformer(model_name, **kwargs)
        check("Model loaded", True)

        embedding = model.encode("test", convert_to_numpy=True)
        check("Encoding works", True, f"dim={len(embedding)}")

    except Exception as e:
        check("Model load/encode", False, str(e)[:300])
        import traceback
        print("\n  Full traceback:")
        traceback.print_exc()

    # ── 9. MCP Server Environment Check ──
    header("9. MCP Server Environment Simulation")
    print("  Checking if ENYAL_SSL_CERT_FILE would be available to MCP server...")

    # Show what the Claude Code MCP config should look like
    print(f"""
  Your Claude Code MCP settings (~/.claude/settings.json) should include
  ENYAL_SSL_CERT_FILE in the 'env' section:

  "enyal": {{
    "command": "uv",
    "args": ["run", "--directory", "/path/to/enyal", "python", "-m", "enyal.mcp"],
    "env": {{
      "ENYAL_DB_PATH": "~/.enyal/context.db",
      "ENYAL_SSL_CERT_FILE": "{cert_file}"
    }}
  }}

  Without this, the MCP server process won't have access to your cert file
  even if it's set in your shell profile.
""")

    # ── Summary ──
    header("Summary")
    print("  If step 5 fails with ORIGINAL cert but passes with COMBINED cert:")
    print("    -> Your cert only has the corporate CA. Enyal now auto-combines")
    print("       with certifi roots. Just update to the latest version.")
    print()
    print("  If step 5 fails with BOTH original and combined cert:")
    print("    -> Your cert may be invalid, wrong format, or the proxy isn't")
    print("       intercepting HuggingFace traffic. Check cert format (must be PEM).")
    print()
    print("  If step 5 passes but step 6 fails:")
    print("    -> huggingface_hub backend config issue. Check version.")
    print()
    print("  If step 6 passes but step 8 fails:")
    print("    -> sentence_transformers specific issue (possibly trust_remote_code).")
    print()
    print("  If all pass but enyal MCP still fails:")
    print("    -> ENYAL_SSL_CERT_FILE may not be reaching the MCP server process.")
    print("       Add it to the 'env' section in ~/.claude/settings.json (see step 9).")

    # Cleanup temp file
    if combined_cert_file != cert_file and os.path.exists(combined_cert_file):
        with contextlib.suppress(OSError):
            os.unlink(combined_cert_file)


if __name__ == "__main__":
    main()
