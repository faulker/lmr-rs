//! TLS material for the listener: PEM files from the config, or a self-signed pair generated
//! into the config directory on first run so `tls.enabled = true` works with nothing else set.

use std::fs;
use std::net::IpAddr;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use axum_server::tls_rustls::RustlsConfig;
use sha2::{Digest, Sha256};

use crate::settings::TlsSettings;

const CERT_FILE: &str = "cert.pem";
const KEY_FILE: &str = "key.pem";

/// Resolved certificate and key paths plus the leaf certificate's SHA-256 fingerprint.
#[derive(Debug)]
pub struct TlsMaterial {
    pub cert: PathBuf,
    pub key: PathBuf,
    pub fingerprint: String,
    /// True when this call created the files.
    pub generated: bool,
}

/// Find or create the PEM pair. `bind` is added to the self-signed cert's alt names so clients
/// connecting by IP can pin it.
pub fn prepare(cfg: &TlsSettings, config_dir: &Path, bind: IpAddr) -> Result<TlsMaterial> {
    let (cert, key, generated) = if cfg.cert.is_empty() {
        let cert = config_dir.join(CERT_FILE);
        let key = config_dir.join(KEY_FILE);
        let generated = !(cert.is_file() && key.is_file());
        if generated {
            generate(&cert, &key, bind)?;
        }
        (cert, key, generated)
    } else {
        (PathBuf::from(&cfg.cert), PathBuf::from(&cfg.key), false)
    };
    let fingerprint = fingerprint(&cert)?;
    Ok(TlsMaterial {
        cert,
        key,
        fingerprint,
        generated,
    })
}

/// Load the pair into a rustls server config. Must run inside a tokio runtime.
pub async fn load(material: &TlsMaterial) -> Result<RustlsConfig> {
    RustlsConfig::from_pem_file(&material.cert, &material.key)
        .await
        .with_context(|| {
            format!(
                "loading TLS files {} / {}",
                material.cert.display(),
                material.key.display()
            )
        })
}

/// Write a fresh self-signed certificate valid for `localhost` and `bind`; the key is 0600.
fn generate(cert_path: &Path, key_path: &Path, bind: IpAddr) -> Result<()> {
    let mut names = vec!["localhost".to_string(), bind.to_string()];
    if bind.is_unspecified() {
        names.push("127.0.0.1".into());
    }
    let ck =
        rcgen::generate_simple_self_signed(names).context("generating self-signed certificate")?;
    if let Some(dir) = cert_path.parent() {
        fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    fs::write(cert_path, ck.cert.pem())
        .with_context(|| format!("writing {}", cert_path.display()))?;
    fs::write(key_path, ck.signing_key.serialize_pem())
        .with_context(|| format!("writing {}", key_path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(key_path, fs::Permissions::from_mode(0o600))
            .with_context(|| format!("chmod 600 {}", key_path.display()))?;
    }
    Ok(())
}

/// `AA:BB:..` SHA-256 of the first certificate in the PEM file, as browsers and `openssl`
/// print it.
fn fingerprint(cert_path: &Path) -> Result<String> {
    let pem = fs::read_to_string(cert_path)
        .with_context(|| format!("reading {}", cert_path.display()))?;
    let der = first_der_block(&pem)
        .ok_or_else(|| anyhow!("no CERTIFICATE block in {}", cert_path.display()))?;
    let digest = Sha256::digest(der);
    Ok(digest
        .iter()
        .map(|b| format!("{b:02X}"))
        .collect::<Vec<_>>()
        .join(":"))
}

/// Decode the base64 body between the first BEGIN/END CERTIFICATE lines.
fn first_der_block(pem: &str) -> Option<Vec<u8>> {
    let mut body = String::new();
    let mut inside = false;
    for line in pem.lines() {
        let line = line.trim();
        if line == "-----BEGIN CERTIFICATE-----" {
            inside = true;
        } else if line == "-----END CERTIFICATE-----" {
            break;
        } else if inside {
            body.push_str(line);
        }
    }
    if body.is_empty() {
        return None;
    }
    base64_decode(&body)
}

/// Standard base64 with padding; enough for PEM, avoids another dependency.
fn base64_decode(s: &str) -> Option<Vec<u8>> {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let mut buf = 0u32;
    let mut bits = 0;
    for c in s.bytes() {
        if c == b'=' {
            break;
        }
        let v = T.iter().position(|&t| t == c)? as u32;
        buf = (buf << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((buf >> bits) as u8);
            buf &= (1 << bits) - 1;
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_once_then_reuses() {
        let dir = std::env::temp_dir().join(format!("lmr-rs-tls-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let cfg = TlsSettings {
            enabled: true,
            ..Default::default()
        };
        let first = prepare(&cfg, &dir, "0.0.0.0".parse().unwrap()).unwrap();
        assert!(first.generated);
        assert_eq!(first.fingerprint.len(), 32 * 3 - 1);
        let second = prepare(&cfg, &dir, "0.0.0.0".parse().unwrap()).unwrap();
        assert!(!second.generated);
        assert_eq!(first.fingerprint, second.fingerprint);
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(load(&second)).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(&second.key).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn explicit_paths_are_used_verbatim() {
        let cfg = TlsSettings {
            enabled: true,
            cert: "/nonexistent/c.pem".into(),
            key: "/nonexistent/k.pem".into(),
        };
        let err = prepare(&cfg, Path::new("/tmp"), "127.0.0.1".parse().unwrap()).unwrap_err();
        assert!(err.to_string().contains("/nonexistent/c.pem"));
    }

    #[test]
    fn base64_roundtrip() {
        assert_eq!(base64_decode("aGVsbG8=").unwrap(), b"hello");
        assert_eq!(base64_decode("aGk=").unwrap(), b"hi");
        assert!(base64_decode("a$").is_none());
    }
}
