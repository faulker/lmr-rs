//! The TOML config file: where to listen, whether non-loopback binding is allowed, the API key,
//! TLS, which checkpoint to serve, the browser UI, and where a daemonized process keeps its pid
//! and log. Every key has a default that reproduces the original loopback-only behaviour, so a
//! missing file is fine. `validate` enforces the safety rules before anything binds.

use std::fs;
use std::net::IpAddr;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::decide::DecidePolicy;
use crate::hub::DEFAULT_MODEL;

/// Environment variable naming an alternative config path.
pub const CONFIG_ENV: &str = "LMR_RS_CONFIG";

/// Which runtime a catalog entry uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Engine {
    /// Laya System One (ModernBERT + decision head).
    Laya,
    /// Causal GGUF chat model (candle quantized Qwen3 or Llama).
    Gguf,
}

/// One named checkpoint this binary can download and run.
pub struct Variant {
    pub name: &'static str,
    pub engine: Engine,
    /// HF repo id (or the Laya default repo).
    pub repo: &'static str,
    /// Subfolder inside `repo`; `None` is the repo root.
    pub subfolder: Option<&'static str>,
    /// GGUF filename inside `repo`.
    pub filename: Option<&'static str>,
    /// HF repo that holds `tokenizer.json` for a GGUF checkpoint.
    pub tokenizer_repo: Option<&'static str>,
    pub description: &'static str,
}

impl Variant {
    /// The hub spec `download` / `delete` / `update` use for this catalog entry.
    pub fn resolve(&self) -> ResolvedModel {
        match self.engine {
            Engine::Gguf => gguf_from_variant(self, ""),
            Engine::Laya => ResolvedModel::Laya {
                repo: self.repo.to_string(),
                subfolder: self.subfolder.map(str::to_string),
            },
        }
    }
}

/// Known variants, in the order `lmr-rs models` prints them.
pub const VARIANTS: &[Variant] = &[
    Variant {
        name: "minicpm5-2b",
        engine: Engine::Gguf,
        repo: "openbmb/MiniCPM5-2B-GGUF",
        subfolder: None,
        filename: Some("MiniCPM5-2B-Q4_K_M.gguf"),
        tokenizer_repo: Some("openbmb/MiniCPM5-2B"),
        description: "MiniCPM5 2B GGUF Q4_K_M, chat (default, ~1.5 GB)",
    },
    Variant {
        name: "qwen3-0.6b",
        engine: Engine::Gguf,
        repo: "Qwen/Qwen3-0.6B-GGUF",
        subfolder: None,
        filename: Some("Qwen3-0.6B-Q8_0.gguf"),
        tokenizer_repo: Some("Qwen/Qwen3-0.6B"),
        description: "Qwen3 0.6B GGUF Q8_0, chat (~640 MB)",
    },
    Variant {
        name: "english",
        engine: Engine::Laya,
        repo: DEFAULT_MODEL,
        subfolder: None,
        filename: None,
        tokenizer_repo: None,
        description: "ModernBERT-large, English (~843 MB)",
    },
    Variant {
        name: "multilingual",
        engine: Engine::Laya,
        repo: DEFAULT_MODEL,
        subfolder: Some("multilingual"),
        filename: None,
        tokenizer_repo: None,
        description: "mmBERT-base, 100+ languages (~678 MB)",
    },
    Variant {
        name: "typed-decisions",
        engine: Engine::Laya,
        repo: DEFAULT_MODEL,
        subfolder: Some("typed-decisions"),
        filename: None,
        tokenizer_repo: None,
        description: "Tuned for typed decision questions (~846 MB)",
    },
];

/// What `hub` should fetch after config and flags are applied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolvedModel {
    Laya {
        repo: String,
        subfolder: Option<String>,
    },
    Gguf {
        repo: String,
        filename: String,
        tokenizer_repo: String,
    },
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct Settings {
    pub server: ServerSettings,
    pub tls: TlsSettings,
    pub model: ModelSettings,
    pub daemon: DaemonSettings,
    pub web: WebSettings,
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
    /// GGUF filename inside `repo`. Empty uses the variant default.
    pub filename: String,
    pub device: String,
    /// Spend unused `max_len` tokens on option texts when the state is short.
    pub pack_head: bool,
    /// Split large `choice` questions into a group question, then a member question.
    pub tournament: bool,
    /// Start the tournament above this many options.
    pub tournament_after: usize,
    /// Floor for a flat `choice` with more than 10 options. `0` keeps the checkpoint
    /// temperature (~0.1 for `choice:11+`).
    pub choice_min_temperature: f32,
}

impl Default for ModelSettings {
    fn default() -> Self {
        Self {
            variant: String::new(),
            repo: String::new(),
            subfolder: String::new(),
            filename: String::new(),
            device: "auto".into(),
            pack_head: true,
            tournament: true,
            tournament_after: 10,
            choice_min_temperature: 1.0,
        }
    }
}

impl ModelSettings {
    /// Runtime decide policy for `Agent::load_with_policy`.
    pub fn policy(&self) -> DecidePolicy {
        DecidePolicy {
            pack_head: self.pack_head,
            tournament: self.tournament,
            tournament_after: self.tournament_after,
            choice_min_temperature: self.choice_min_temperature,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct DaemonSettings {
    pub pid_file: String,
    pub log_file: String,
}

/// Browser UI. On by default; set `enabled = false` (or `lmr-rs serve --no-web`) to hide it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct WebSettings {
    pub enabled: bool,
    /// Empty means the UI is open to anyone who can reach the listener.
    pub password: String,
}

impl Default for WebSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            password: String::new(),
        }
    }
}

/// The commented config that `lmr-rs config init` writes.
pub const TEMPLATE: &str = r#"# lmr-rs configuration. Every key is optional; these are the defaults.

[server]
bind = "127.0.0.1"      # IP to listen on; "0.0.0.0" or "::" for all interfaces
port = 8321
public = false          # must be true to bind anything other than loopback
api_key = ""            # or api_key_env = "LMR_API_KEY" / api_key_file = "/path/to/key"
health_requires_key = false

[tls]
enabled = false
cert = ""               # PEM paths; when both are empty and enabled = true, a
key = ""                # self-signed pair is generated next to this file on first run

[model]
# variant = "minicpm5-2b"   # omit to use a downloaded checkpoint; prompt if several are cached
                            # minicpm5-2b | qwen3-0.6b | english | multilingual | typed-decisions
# repo = "convaiinnovations/laya"   # raw HF id or local dir; overrides variant
# subfolder = ""
# filename = ""         # GGUF file; overrides the variant default (e.g. MiniCPM5-2B-Q8_0.gguf)
device = "auto"         # auto | cpu | metal | cuda
pack_head = true        # spend unused max_len tokens on option texts
tournament = true       # split choice questions larger than tournament_after
tournament_after = 10   # stay out of the published choice:11+ bucket
choice_min_temperature = 1.0  # floor for a flat 11+ choice; 0 = checkpoint (~0.1)

[daemon]
pid_file = ""           # defaults to <config dir>/lmr-rs.pid when --daemonize is used
log_file = ""           # defaults to <config dir>/lmr-rs.log

[web]
enabled = true          # JSON console at GET /; false or --no-web to hide it
password = ""           # empty = open; set a password to gate the UI
"#;

/// Default config location: `$LMR_RS_CONFIG`, else `$XDG_CONFIG_HOME/lmr-rs/config.toml`,
/// else `~/.config/lmr-rs/config.toml`.
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
    base.join("lmr-rs").join("config.toml")
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
                 to expose lmr-rs beyond this machine",
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
        if self.model.repo.is_empty()
            && !self.model.variant.is_empty()
            && variant(&self.model.variant).is_none()
        {
            bail!(
                "unknown model.variant {:?}; expected one of {}",
                self.model.variant,
                VARIANTS
                    .iter()
                    .map(|v| v.name)
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        if !self.model.repo.is_empty() && !self.model.variant.is_empty() {
            bail!("set model.repo or model.variant, not both");
        }
        if !self.model.filename.is_empty() && !self.uses_gguf() {
            bail!("model.filename is only used for GGUF models (qwen3-0.6b, minicpm5-2b, or repo + filename)");
        }
        if self.model.tournament_after < 2 {
            bail!("model.tournament_after must be at least 2");
        }
        if self.model.choice_min_temperature < 0.0 {
            bail!("model.choice_min_temperature must be >= 0");
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

    /// True when config or flags named a checkpoint (`variant` or `repo`).
    pub fn model_specified(&self) -> bool {
        !self.model.repo.is_empty() || !self.model.variant.is_empty()
    }

    /// True when this config loads a GGUF chat model rather than Laya.
    pub fn uses_gguf(&self) -> bool {
        if !self.model.repo.is_empty() {
            return !self.model.filename.is_empty()
                || gguf_variant_for_repo(&self.model.repo).is_some();
        }
        variant(&self.model.variant).is_some_and(|v| v.engine == Engine::Gguf)
    }

    /// What `hub` should download (or open on disk).
    pub fn resolve_model(&self) -> ResolvedModel {
        let m = &self.model;
        if !m.repo.is_empty() {
            if let Some(spec) = resolve_gguf(&m.repo, &m.filename) {
                return spec;
            }
            let sub = (!m.subfolder.is_empty()).then(|| m.subfolder.clone());
            return ResolvedModel::Laya {
                repo: m.repo.clone(),
                subfolder: sub,
            };
        }
        match variant(&m.variant) {
            Some(v) if v.engine == Engine::Gguf => gguf_from_variant(v, &m.filename),
            Some(v) => ResolvedModel::Laya {
                repo: v.repo.to_string(),
                subfolder: v.subfolder.map(str::to_string),
            },
            None => ResolvedModel::Laya {
                repo: DEFAULT_MODEL.to_string(),
                subfolder: None,
            },
        }
    }

    /// Path for the pid file, defaulting next to the config file.
    pub fn pid_file(&self, config_path: &Path) -> PathBuf {
        or_beside(&self.daemon.pid_file, config_path, "lmr-rs.pid")
    }

    /// Path for the daemon log, defaulting next to the config file.
    pub fn log_file(&self, config_path: &Path) -> PathBuf {
        or_beside(&self.daemon.log_file, config_path, "lmr-rs.log")
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

/// Look a GGUF catalog entry up by its weights repo id.
fn gguf_variant_for_repo(repo: &str) -> Option<&'static Variant> {
    VARIANTS
        .iter()
        .find(|v| v.engine == Engine::Gguf && v.repo == repo)
}

/// Build a GGUF spec from a catalog entry, honoring an optional filename override.
fn gguf_from_variant(v: &Variant, filename: &str) -> ResolvedModel {
    ResolvedModel::Gguf {
        repo: v.repo.to_string(),
        filename: if filename.is_empty() {
            v.filename.unwrap_or("").to_string()
        } else {
            filename.to_string()
        },
        tokenizer_repo: v.tokenizer_repo.unwrap_or(v.repo).to_string(),
    }
}

/// Resolve a raw repo (and optional filename) to a GGUF spec when we can.
fn resolve_gguf(repo: &str, filename: &str) -> Option<ResolvedModel> {
    if let Some(v) = gguf_variant_for_repo(repo) {
        return Some(gguf_from_variant(v, filename));
    }
    if filename.is_empty() {
        return None;
    }
    Some(ResolvedModel::Gguf {
        repo: repo.to_string(),
        filename: filename.to_string(),
        tokenizer_repo: repo.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn product_paths_use_lmr_rs() {
        assert_eq!(CONFIG_ENV, "LMR_RS_CONFIG");
        assert!(TEMPLATE.contains("# lmr-rs configuration"));
        assert!(TEMPLATE.contains("lmr-rs.pid"));
        assert!(TEMPLATE.contains("LMR_API_KEY"));
    }

    #[test]
    fn template_parses_to_defaults() {
        assert_eq!(Settings::parse(TEMPLATE).unwrap(), Settings::default());
        assert!(Settings::default().validate().is_ok());
    }

    #[test]
    fn minicpm5_is_the_default_and_listed_first() {
        assert_eq!(VARIANTS[0].name, "minicpm5-2b");
        assert!(VARIANTS[0].description.contains("default"));
        assert_eq!(
            VARIANTS
                .iter()
                .filter(|v| v.description.contains("default"))
                .count(),
            1
        );
        assert!(TEMPLATE.contains("# variant = \"minicpm5-2b\""));
    }

    #[test]
    fn catalog_disk_sizes_match_published_checkpoints() {
        assert!(variant("english").unwrap().description.contains("~843 MB"));
        assert!(variant("multilingual")
            .unwrap()
            .description
            .contains("~678 MB"));
        assert!(variant("typed-decisions")
            .unwrap()
            .description
            .contains("~846 MB"));
    }

    #[test]
    fn unknown_keys_are_rejected() {
        assert!(Settings::parse("[server]\nprot = 1\n").is_err());
    }

    #[test]
    fn public_bind_needs_flag_and_key() {
        let mut s = Settings::parse("[server]\nbind = \"0.0.0.0\"\n").unwrap();
        assert!(s
            .validate()
            .unwrap_err()
            .to_string()
            .contains("server.public"));
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
        assert_eq!(
            s.resolve_model(),
            ResolvedModel::Laya {
                repo: DEFAULT_MODEL.to_string(),
                subfolder: Some("multilingual".into()),
            }
        );
        assert_eq!(
            variant("multilingual").unwrap().resolve(),
            s.resolve_model()
        );
        let s = Settings::parse("[model]\nrepo = \"/tmp/ckpt\"\nsubfolder = \"x\"\n").unwrap();
        assert_eq!(
            s.resolve_model(),
            ResolvedModel::Laya {
                repo: "/tmp/ckpt".to_string(),
                subfolder: Some("x".into()),
            }
        );
        assert!(s.validate().is_ok());
        let s = Settings::parse("[model]\nvariant = \"nope\"\n").unwrap();
        assert!(s
            .validate()
            .unwrap_err()
            .to_string()
            .contains("unknown model.variant"));
        let s = Settings::parse("[model]\nvariant = \"multilingual\"\nrepo = \"a/b\"\n").unwrap();
        assert!(s.validate().unwrap_err().to_string().contains("not both"));
        let s = Settings::parse("[model]\nrepo = \"a/b\"\nvariant = \"english\"\n").unwrap();
        assert!(s.validate().unwrap_err().to_string().contains("not both"));
        assert!(!Settings::default().model_specified());
        assert!(Settings::default().validate().is_ok());
        assert!(Settings::parse("[model]\nvariant = \"english\"\n")
            .unwrap()
            .model_specified());
    }

    #[test]
    fn gguf_variants_resolve_and_override_filename() {
        let s = Settings::parse("[model]\nvariant = \"qwen3-0.6b\"\n").unwrap();
        assert!(s.validate().is_ok());
        assert!(s.uses_gguf());
        assert_eq!(
            s.resolve_model(),
            ResolvedModel::Gguf {
                repo: "Qwen/Qwen3-0.6B-GGUF".into(),
                filename: "Qwen3-0.6B-Q8_0.gguf".into(),
                tokenizer_repo: "Qwen/Qwen3-0.6B".into(),
            }
        );
        assert_eq!(variant("qwen3-0.6b").unwrap().resolve(), s.resolve_model());
        let s = Settings::parse(
            "[model]\nvariant = \"minicpm5-2b\"\nfilename = \"MiniCPM5-2B-Q8_0.gguf\"\n",
        )
        .unwrap();
        assert!(s.validate().is_ok());
        assert_eq!(
            s.resolve_model(),
            ResolvedModel::Gguf {
                repo: "openbmb/MiniCPM5-2B-GGUF".into(),
                filename: "MiniCPM5-2B-Q8_0.gguf".into(),
                tokenizer_repo: "openbmb/MiniCPM5-2B".into(),
            }
        );
        let s = Settings::parse("[model]\nvariant = \"english\"\nfilename = \"x.gguf\"\n").unwrap();
        assert!(s
            .validate()
            .unwrap_err()
            .to_string()
            .contains("filename is only used"));
        let s = Settings::parse("[model]\nrepo = \"/tmp/weights\"\nfilename = \"local.gguf\"\n")
            .unwrap();
        assert!(s.validate().is_ok());
        assert_eq!(
            s.resolve_model(),
            ResolvedModel::Gguf {
                repo: "/tmp/weights".into(),
                filename: "local.gguf".into(),
                tokenizer_repo: "/tmp/weights".into(),
            }
        );
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
        let cfg = Path::new("/etc/lmr-rs/config.toml");
        assert_eq!(s.pid_file(cfg), PathBuf::from("/etc/lmr-rs/lmr-rs.pid"));
        let s = Settings::parse("[daemon]\nlog_file = \"/var/log/l.log\"\n").unwrap();
        assert_eq!(s.log_file(cfg), PathBuf::from("/var/log/l.log"));
    }

    #[test]
    fn decide_keys_parse_and_validate() {
        assert_eq!(Settings::default().model.policy(), DecidePolicy::default());
        let s = Settings::parse(
            r#"
            [model]
            pack_head = false
            tournament = false
            tournament_after = 8
            choice_min_temperature = 0.0
            "#,
        )
        .unwrap();
        assert!(!s.model.pack_head);
        assert!(!s.model.tournament);
        assert_eq!(s.model.tournament_after, 8);
        assert_eq!(s.model.choice_min_temperature, 0.0);
        assert!(s.validate().is_ok());
        let s = Settings::parse("[model]\ntournament_after = 1\n").unwrap();
        assert!(s
            .validate()
            .unwrap_err()
            .to_string()
            .contains("tournament_after"));
        let s = Settings::parse("[model]\nchoice_min_temperature = -1\n").unwrap();
        assert!(s
            .validate()
            .unwrap_err()
            .to_string()
            .contains("choice_min_temperature"));
    }

    #[test]
    fn web_defaults_on_and_parses() {
        assert!(Settings::default().web.enabled);
        assert!(Settings::default().web.password.is_empty());
        let s = Settings::parse("[web]\nenabled = true\npassword = \"desk\"\n").unwrap();
        assert!(s.web.enabled);
        assert_eq!(s.web.password, "desk");
        assert!(s.validate().is_ok());
        let off = Settings::parse("[web]\nenabled = false\n").unwrap();
        assert!(!off.web.enabled);
    }
}
