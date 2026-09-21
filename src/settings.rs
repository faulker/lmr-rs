//! The TOML config file: where to listen, whether non-loopback binding is allowed, the API key,
//! TLS, which Laya checkpoint to serve, and where a daemonized process keeps its pid and log.
//! Every key has a default that reproduces the original loopback-only behaviour, so a missing
//! file is fine. `validate` enforces the safety rules before anything binds.

use std::fs;
use std::net::IpAddr;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::hub::DEFAULT_MODEL;

/// Environment variable naming an alternative config path.
pub const CONFIG_ENV: &str = "LAYA_RS_CONFIG";

/// One of the checkpoints published in the Laya repo.
pub struct Variant {
    pub name: &'static str,
    /// Subfolder inside `DEFAULT_MODEL`; `None` is the repo root.
    pub subfolder: Option<&'static str>,
    pub description: &'static str,
}

/// Known variants, in the order `laya-rs models` prints them.
pub const VARIANTS: &[Variant] = &[
    Variant {
        name: "english",
        subfolder: None,
        description: "ModernBERT-large, English (default, ~843 MB)",
    },
    Variant {
        name: "multilingual",
        subfolder: Some("multilingual"),
        description: "mmBERT-base, 100+ languages",
    },
    Variant {
        name: "typed-decisions",
        subfolder: Some("typed-decisions"),
        description: "Tuned for typed decision questions",
    },
];

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct Settings {
    pub server: ServerSettings,
    pub tls: TlsSettings,
    pub model: ModelSettings,
    pub daemon: DaemonSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct ServerSettings {
    pub bind: IpAddr,
    pub port: u16,
    /// Must be true to bind anything other than a loopback address.
    pub public: bool,
    pub api_key: String,
    pub api_key_env: String,
    pub api_key_file: String,
    /// When true `GET /health` needs the key too.
    pub health_requires_key: bool,
}

impl Default for ServerSettings {
    fn default() -> Self {
        Self {
            bind: IpAddr::from([127, 0, 0, 1]),
            port: 8321,
            public: false,
            api_key: String::new(),
            api_key_env: String::new(),
            api_key_file: String::new(),
            health_requires_key: false,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct TlsSettings {
    pub enabled: bool,
    /// PEM certificate path. Empty with `key` empty means "generate a self-signed pair".
    pub cert: String,
    pub key: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct ModelSettings {
    /// One of `VARIANTS` by name.
    pub variant: String,
    /// Raw HF repo id or local directory; when set it wins over `variant`.
    pub repo: String,
    pub subfolder: String,
    pub device: String,
}

impl Default for ModelSettings {
    fn default() -> Self {
        Self {
            variant: "english".into(),
            repo: String::new(),
            subfolder: String::new(),
            device: "auto".into(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct DaemonSettings {
    pub pid_file: String,
    pub log_file: String,
}

/// The commented config that `laya-rs config init` writes.
pub const TEMPLATE: &str = r#"# laya-rs configuration. Every key is optional; these are the defaults.

[server]
bind = "127.0.0.1"      # IP to listen on; "0.0.0.0" or "::" for all interfaces
port = 8321
public = false          # must be true to bind anything other than loopback
api_key = ""            # or api_key_env = "LAYA_API_KEY" / api_key_file = "/path/to/key"
health_requires_key = false

[tls]
enabled = false
cert = ""               # PEM paths; when both are empty and enabled = true, a
key = ""                # self-signed pair is generated next to this file on first run

[model]
variant = "english"     # english | multilingual | typed-decisions
# repo = "convaiinnovations/laya"   # raw HF id or local dir; overrides variant
# subfolder = ""
device = "auto"         # auto | cpu | metal | cuda

[daemon]
pid_file = ""           # defaults to <config dir>/laya-rs.pid when --daemonize is used
log_file = ""           # defaults to <config dir>/laya-rs.log
"#;

/// Default config location: `$LAYA_RS_CONFIG`, else `$XDG_CONFIG_HOME/laya-rs/config.toml`,
/// else `~/.config/laya-rs/config.toml`.
pub fn default_path() -> PathBuf {
    if let Ok(p) = std::env::var(CONFIG_ENV) {
        if !p.is_empty() {
            return PathBuf::from(p);
        }
    }
    let base = std::env::var("XDG_CONFIG_HOME")
        .ok()
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
            PathBuf::from(home).join(".config")
        });
    base.join("laya-rs").join("config.toml")
}

impl Settings {
    /// Parse `path`; a missing file yields the defaults.
    pub fn load(path: &Path) -> Result<Self> {
        match fs::read_to_string(path) {
            Ok(text) => Self::parse(&text).with_context(|| format!("parsing {}", path.display())),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(e).with_context(|| format!("reading {}", path.display())),
        }
    }

    pub fn parse(text: &str) -> Result<Self> {
        Ok(toml::from_str(text)?)
    }

    /// Enforce the rules that make a public listener safe to start.
    pub fn validate(&self) -> Result<()> {
        let s = &self.server;
        if !s.bind.is_loopback() && !s.public {
            bail!(
                "bind = {} is not a loopback address; set server.public = true (and an api key) \
                 to expose laya-rs beyond this machine",
                s.bind
            );
        }
        let sources = [&s.api_key, &s.api_key_env, &s.api_key_file]
            .iter()
            .filter(|v| !v.is_empty())
            .count();
        if sources > 1 {
            bail!("set only one of server.api_key, server.api_key_env, server.api_key_file");
        }
        if s.public && sources == 0 {
            bail!("server.public = true requires an api key (server.api_key, api_key_env, or api_key_file)");
        }
        if self.tls.cert.is_empty() != self.tls.key.is_empty() {
            bail!("tls.cert and tls.key must be set together");
        }
        if self.model.repo.is_empty() && variant(&self.model.variant).is_none() {
            bail!(
                "unknown model.variant {:?}; expected one of {}",
                self.model.variant,
                VARIANTS.iter().map(|v| v.name).collect::<Vec<_>>().join(", ")
            );
        }
        if !self.model.repo.is_empty() && self.model.variant != ModelSettings::default().variant {
            bail!("set model.repo or model.variant, not both");
        }
        Ok(())
    }

    /// The API key from whichever source is configured, or `None` when auth is off.
    pub fn resolve_api_key(&self) -> Result<Option<String>> {
        let s = &self.server;
        let key = if !s.api_key.is_empty() {
            s.api_key.clone()
        } else if !s.api_key_env.is_empty() {
            std::env::var(&s.api_key_env)
                .with_context(|| format!("reading api key from ${}", s.api_key_env))?
        } else if !s.api_key_file.is_empty() {
            fs::read_to_string(&s.api_key_file)
                .with_context(|| format!("reading api key file {}", s.api_key_file))?
        } else {
            return Ok(None);
        };
        let key = key.trim().to_string();
        if key.is_empty() {
            bail!("the configured api key is empty");
        }
        Ok(Some(key))
    }

    /// `(repo or directory, subfolder)` to hand to `hub::fetch`.
    pub fn resolve_model(&self) -> (String, Option<String>) {
        let m = &self.model;
        if !m.repo.is_empty() {
            let sub = (!m.subfolder.is_empty()).then(|| m.subfolder.clone());
            return (m.repo.clone(), sub);
        }
        let sub = variant(&m.variant)
            .and_then(|v| v.subfolder)
            .map(str::to_string);
        (DEFAULT_MODEL.to_string(), sub)
    }

    /// Path for the pid file, defaulting next to the config file.
    pub fn pid_file(&self, config_path: &Path) -> PathBuf {
        or_beside(&self.daemon.pid_file, config_path, "laya-rs.pid")
    }

    /// Path for the daemon log, defaulting next to the config file.
    pub fn log_file(&self, config_path: &Path) -> PathBuf {
        or_beside(&self.daemon.log_file, config_path, "laya-rs.log")
    }
}

/// `value` when set, else `name` in the config file's directory.
fn or_beside(value: &str, config_path: &Path, name: &str) -> PathBuf {
    if !value.is_empty() {
        return PathBuf::from(value);
    }
    config_path.parent().unwrap_or(Path::new(".")).join(name)
}

/// Look a variant up by name.
pub fn variant(name: &str) -> Option<&'static Variant> {
    VARIANTS.iter().find(|v| v.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn template_parses_to_defaults() {
        assert_eq!(Settings::parse(TEMPLATE).unwrap(), Settings::default());
        assert!(Settings::default().validate().is_ok());
    }

    #[test]
    fn unknown_keys_are_rejected() {
        assert!(Settings::parse("[server]\nprot = 1\n").is_err());
    }

    #[test]
    fn public_bind_needs_flag_and_key() {
        let mut s = Settings::parse("[server]\nbind = \"0.0.0.0\"\n").unwrap();
        assert!(s.validate().unwrap_err().to_string().contains("server.public"));
        s.server.public = true;
        assert!(s.validate().unwrap_err().to_string().contains("api key"));
        s.server.api_key = "k".into();
        assert!(s.validate().is_ok());
        s.server.api_key_env = "X".into();
        assert!(s.validate().unwrap_err().to_string().contains("only one"));
    }

    #[test]
    fn tls_paths_come_in_pairs() {
        let s = Settings::parse("[tls]\nenabled = true\ncert = \"a.pem\"\n").unwrap();
        assert!(s.validate().unwrap_err().to_string().contains("together"));
    }

    #[test]
    fn model_resolution() {
        let s = Settings::parse("[model]\nvariant = \"multilingual\"\n").unwrap();
        assert_eq!(s.resolve_model(), (DEFAULT_MODEL.to_string(), Some("multilingual".into())));
        let s = Settings::parse("[model]\nrepo = \"/tmp/ckpt\"\nsubfolder = \"x\"\n").unwrap();
        assert_eq!(s.resolve_model(), ("/tmp/ckpt".to_string(), Some("x".into())));
        assert!(s.validate().is_ok());
        let s = Settings::parse("[model]\nvariant = \"nope\"\n").unwrap();
        assert!(s.validate().unwrap_err().to_string().contains("unknown model.variant"));
        let s = Settings::parse("[model]\nvariant = \"multilingual\"\nrepo = \"a/b\"\n").unwrap();
        assert!(s.validate().unwrap_err().to_string().contains("not both"));
    }

    #[test]
    fn api_key_sources() {
        let s = Settings::parse("[server]\napi_key = \" abc \"\n").unwrap();
        assert_eq!(s.resolve_api_key().unwrap().as_deref(), Some("abc"));
        assert_eq!(Settings::default().resolve_api_key().unwrap(), None);
        let s = Settings::parse("[server]\napi_key_file = \"/nonexistent/key\"\n").unwrap();
        assert!(s.resolve_api_key().is_err());
    }

    #[test]
    fn daemon_paths_default_beside_config() {
        let s = Settings::default();
        let cfg = Path::new("/etc/laya-rs/config.toml");
        assert_eq!(s.pid_file(cfg), PathBuf::from("/etc/laya-rs/laya-rs.pid"));
        let s = Settings::parse("[daemon]\nlog_file = \"/var/log/l.log\"\n").unwrap();
        assert_eq!(s.log_file(cfg), PathBuf::from("/var/log/l.log"));
    }
}
