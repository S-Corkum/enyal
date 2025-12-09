# SSL Troubleshooting Guide for Corporate Networks

This guide helps users in corporate environments with SSL inspection (e.g., Zscaler, BlueCoat, corporate proxies) resolve SSL certificate errors when Enyal downloads its embedding model.

## Understanding the Problem

When you first run Enyal, it downloads the `all-MiniLM-L6-v2` embedding model from Hugging Face Hub. Corporate networks often use **SSL inspection** (also called TLS interception) to monitor HTTPS traffic, which injects enterprise CA certificates into the SSL chain.

Python's `requests` library (used by Hugging Face libraries) doesn't recognize these enterprise certificates by default, resulting in errors like:

```
SSLError: HTTPSConnectionPool(host='huggingface.co', port=443):
Max retries exceeded with url: /api/models/sentence-transformers/all-MiniLM-L6-v2
(Caused by SSLError(SSLCertVerificationError(1,
'[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed:
self signed certificate in certificate chain')))
```

## Quick Diagnosis

Run the following command to check your SSL configuration status:

```bash
enyal model status
```

This shows:
- Whether SSL verification is enabled
- Configured certificate paths
- System CA bundle location
- Offline mode status
- Library versions

## Solution Decision Tree

```
Is the model already cached?
├── Yes → Set ENYAL_OFFLINE_MODE=true (see Approach 4)
└── No → Continue...
    │
    Can you obtain your corporate CA certificate bundle?
    ├── Yes → Use ENYAL_SSL_CERT_FILE (see Approach 1 - Recommended)
    └── No → Continue...
        │
        Can you download the model on a different network?
        ├── Yes → Pre-download and copy (see Approach 3)
        └── No → Use ENYAL_SSL_VERIFY=false (see Approach 2 - Last Resort)
```

## Approach 1: Corporate CA Bundle (Recommended)

The most secure approach is to configure Enyal to use your corporate CA certificate bundle.

### Step 1: Find Your Corporate CA Bundle

**Ask your IT department** for the corporate CA certificate bundle. Common locations:

| Platform | Common Locations |
|----------|-----------------|
| macOS | `/etc/ssl/cert.pem`, Keychain Access → System Roots |
| Linux | `/etc/ssl/certs/ca-certificates.crt`, `/etc/pki/tls/certs/ca-bundle.crt` |
| Windows | `certmgr.msc` → Trusted Root Certification Authorities |

**Export from browser** (Chrome/Firefox):
1. Visit `https://huggingface.co`
2. Click the lock icon → Certificate
3. Export the certificate chain (including root CA)
4. Save as PEM format

### Step 2: Configure Enyal

**Option A: Environment Variable (Recommended)**

```bash
# Add to your shell profile (~/.bashrc, ~/.zshrc, etc.)
export ENYAL_SSL_CERT_FILE=/path/to/corporate-ca-bundle.crt

# Then run Enyal
enyal serve
```

**Option B: MCP Configuration**

```json
{
  "mcpServers": {
    "enyal": {
      "command": "uvx",
      "args": ["enyal", "serve"],
      "env": {
        "ENYAL_SSL_CERT_FILE": "/path/to/corporate-ca-bundle.crt"
      }
    }
  }
}
```

### Step 3: Verify Configuration

```bash
# Check configuration
enyal model status

# Download and verify model
enyal model download
enyal model verify
```

## Approach 2: Disable SSL Verification (Last Resort)

**WARNING: This is insecure and should only be used when you cannot obtain the CA bundle and understand the security implications.**

```bash
# Disable SSL verification (INSECURE)
export ENYAL_SSL_VERIFY=false

# Download the model
enyal model download

# IMPORTANT: Re-enable SSL and switch to offline mode
unset ENYAL_SSL_VERIFY
export ENYAL_OFFLINE_MODE=true
```

**For MCP configuration:**

```json
{
  "mcpServers": {
    "enyal": {
      "command": "uvx",
      "args": ["enyal", "serve"],
      "env": {
        "ENYAL_SSL_VERIFY": "false"
      }
    }
  }
}
```

## Approach 3: Pre-download Model (Air-gapped/Highly Restricted)

For air-gapped environments or when you need to download on a different network:

### On a machine with unrestricted network access:

```bash
# Download the model
enyal model download --cache-dir /path/to/export/location

# Or use Python directly
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
model.save('/path/to/export/my-model')
"
```

### Transfer to the restricted machine:

Copy the model directory to the target machine.

### Configure Enyal to use the local model:

```bash
export ENYAL_MODEL_PATH=/path/to/my-model
export ENYAL_OFFLINE_MODE=true
enyal serve
```

**MCP configuration:**

```json
{
  "mcpServers": {
    "enyal": {
      "command": "uvx",
      "args": ["enyal", "serve"],
      "env": {
        "ENYAL_MODEL_PATH": "/path/to/my-model",
        "ENYAL_OFFLINE_MODE": "true"
      }
    }
  }
}
```

## Approach 4: Use Cached Model (Offline Mode)

If the model is already cached (from a previous successful download), you can run in offline mode:

```bash
# Check if model is cached
ls ~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2

# If it exists, enable offline mode
export ENYAL_OFFLINE_MODE=true
enyal serve
```

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `ENYAL_SSL_CERT_FILE` | (none) | Path to CA certificate bundle |
| `ENYAL_SSL_VERIFY` | `true` | Enable/disable SSL verification |
| `ENYAL_MODEL_PATH` | (none) | Path to local model directory |
| `ENYAL_OFFLINE_MODE` | `false` | Prevent all network calls |
| `HF_HOME` | `~/.cache/huggingface` | Hugging Face cache directory |
| `REQUESTS_CA_BUNDLE` | (system) | Fallback CA bundle (standard Python) |
| `SSL_CERT_FILE` | (system) | Fallback CA bundle (standard Python) |

**Priority**: `ENYAL_SSL_CERT_FILE` > `REQUESTS_CA_BUNDLE` > `SSL_CERT_FILE` > system default

## CLI Commands Reference

```bash
# Check SSL/network configuration status
enyal model status

# Download model (respects SSL configuration)
enyal model download

# Download specific model
enyal model download --model all-MiniLM-L6-v2

# Download to custom cache directory
enyal model download --cache-dir /custom/path

# Verify model can be loaded
enyal model verify

# Verify specific model/path
enyal model verify --model /path/to/local/model
```

## Platform-Specific Notes

### macOS

macOS stores certificates in the Keychain, not as files. You need to export them first.

#### Option 1: Export Corporate CA from Keychain (Recommended)

```bash
# Step 1: Find your corporate CA certificate name
# Open Keychain Access → System → Certificates
# Look for certificates with your company name (e.g., "Zscaler Root CA", "ACME Corp CA")

# Step 2: Export via command line (replace "Your Corporate CA Name" with actual name)
security find-certificate -c "Your Corporate CA Name" -p > ~/corporate-ca.pem

# Step 3: Verify the export worked
cat ~/corporate-ca.pem | head -5
# Should show: -----BEGIN CERTIFICATE-----

# Step 4: Use with Enyal
export ENYAL_SSL_CERT_FILE=~/corporate-ca.pem
```

#### Option 2: Export All System Certificates

```bash
# Export ALL certificates from System Keychain (includes corporate CAs)
security find-certificate -a -p /Library/Keychains/System.keychain > ~/all-system-certs.pem

# Also include the system roots
security find-certificate -a -p /System/Library/Keychains/SystemRootCertificates.keychain >> ~/all-system-certs.pem

# Use the combined bundle
export ENYAL_SSL_CERT_FILE=~/all-system-certs.pem
```

#### Option 3: Export via Keychain Access GUI

1. Open **Keychain Access** (Applications → Utilities → Keychain Access)
2. Select **System** keychain in the sidebar
3. Click **Certificates** category
4. Find your corporate CA certificate (often named after your company or proxy vendor)
5. Right-click → **Export "Certificate Name"...**
6. Save as `.pem` format (Privacy Enhanced Mail)
7. Use the exported file:
   ```bash
   export ENYAL_SSL_CERT_FILE=/path/to/exported-cert.pem
   ```

#### Option 4: Combine with System Bundle

If you need both the corporate CA and standard CAs:

```bash
# Start with system CAs
cp /etc/ssl/cert.pem ~/combined-ca-bundle.pem

# Add your corporate CA
security find-certificate -c "Your Corporate CA Name" -p >> ~/combined-ca-bundle.pem

# Use combined bundle
export ENYAL_SSL_CERT_FILE=~/combined-ca-bundle.pem
```

#### Finding Your Corporate CA Name

Not sure what your corporate CA is called? Try:

```bash
# List all certificates in System keychain
security find-certificate -a /Library/Keychains/System.keychain | grep "alis" | head -20

# Or check what's intercepting HTTPS (requires curl with verbose)
curl -v https://huggingface.co 2>&1 | grep -i "issuer"
```

#### System CA Bundle Locations

```bash
# Default macOS system bundle (may not include corporate CAs)
/etc/ssl/cert.pem

# Homebrew OpenSSL (if installed)
/usr/local/etc/openssl/cert.pem
/usr/local/etc/openssl@1.1/cert.pem
/opt/homebrew/etc/openssl@3/cert.pem  # Apple Silicon
```

### Linux (Debian/Ubuntu)

```bash
# System CA bundle
/etc/ssl/certs/ca-certificates.crt

# Update CA certificates after adding corporate cert
sudo cp corporate-ca.crt /usr/local/share/ca-certificates/
sudo update-ca-certificates
```

### Linux (RHEL/CentOS/Fedora)

```bash
# System CA bundle
/etc/pki/tls/certs/ca-bundle.crt

# Update CA certificates
sudo cp corporate-ca.crt /etc/pki/ca-trust/source/anchors/
sudo update-ca-trust
```

### Windows

```bash
# Export from Certificate Manager
certmgr.msc → Trusted Root Certification Authorities → Certificates
Right-click → All Tasks → Export → Base-64 encoded X.509 (.CER)

# Or use PowerShell
Get-ChildItem -Path Cert:\LocalMachine\Root | Export-Certificate -FilePath C:\certs\corporate-ca.cer
```

## Troubleshooting Common Issues

### "SSL: CERTIFICATE_VERIFY_FAILED"

1. Verify your CA bundle file exists: `ls -la /path/to/cert.pem`
2. Check file is readable: `cat /path/to/cert.pem | head`
3. Ensure it's in PEM format (starts with `-----BEGIN CERTIFICATE-----`)
4. Try the system CA bundle: `export ENYAL_SSL_CERT_FILE=/etc/ssl/cert.pem`

### "Model not found" in Offline Mode

1. Check cache exists: `ls ~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2`
2. If not, download first: `enyal model download`
3. Or copy from another machine to this location

### "Permission denied" errors

1. Check file permissions: `chmod 644 /path/to/cert.pem`
2. Check directory permissions: `chmod 755 ~/.enyal`

### MCP server won't start

1. Test CLI directly: `uvx enyal model status`
2. Check logs: Set `ENYAL_LOG_LEVEL=DEBUG` in MCP config
3. Verify environment variables are passed correctly in MCP config

## Getting Help

If you continue to experience issues:

1. Run `enyal model status --json` and include the output
2. Include the full error message
3. Describe your network environment (corporate proxy, VPN, etc.)
4. Open an issue at: https://github.com/seancorkum/enyal/issues

## Security Best Practices

1. **Never commit CA bundles** to version control
2. **Prefer CA bundle over disabling verification** - `ENYAL_SSL_VERIFY=false` should be temporary
3. **Use offline mode in production** - Download once, then set `ENYAL_OFFLINE_MODE=true`
4. **Regularly update CA bundles** - Corporate certificates can expire
5. **Audit your SSL configuration** - Run `enyal model status` periodically
