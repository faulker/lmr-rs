//! `lmr-rs`: download a checkpoint, serve it (foreground or daemonized, plain or TLS,
//! with an optional API key), ask it one question, or manage its config and service files.

use std::fs;
use std::io::{self, IsTerminal};
use std::net::{IpAddr, SocketAddr, TcpListener};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use candle_core::Device;
use clap::{Args, Parser, Subcommand};
use lmr_rs::settings::{self, ResolvedModel, Settings, Variant, VARIANTS};
use lmr_rs::{
    hub, tls, Agent, Auth, ChatMessage, ChatOpts, Decider, GgufEngine, HttpServer, ModelFiles,
    WebConfig,
};
use serde_json::Value;

#[derive(Parser)]
#[command(
    name = "lmr-rs",
    version,
    about = "Native Rust inference for local models"
)]
struct Cli {
    /// Config file (default: $LMR_RS_CONFIG or ~/.config/lmr-rs/config.toml).
    #[arg(long, global = true)]
    config: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}

/// Model flags shared by `download`, `serve`, and `ask`; each overrides the `[model]` section.
#[derive(Args, Clone, Default)]
struct ModelArgs {
    /// Hugging Face repo id or a local checkpoint directory.
    #[arg(long)]
    model: Option<String>,
    /// Checkpoint subfolder inside the repo, e.g. `multilingual` or `typed-decisions`.
    #[arg(long)]
    subfolder: Option<String>,
    /// Named checkpoint: minicpm5-2b, qwen3-0.6b, english, multilingual, typed-decisions.
    #[arg(long, conflicts_with = "model")]
    variant: Option<String>,
    /// GGUF filename inside the repo; overrides the variant default.
    #[arg(long)]
    filename: Option<String>,
    /// Where to run: auto, cpu, metal, or cuda.
    #[arg(long)]
    device: Option<String>,
}

#[derive(Subcommand)]
enum Command {
    /// Fetch the checkpoint into the Hugging Face cache and exit.
    Download {
        #[command(flatten)]
        model: ModelArgs,
    },
    /// Load the model and serve the HTTP API plus the browser UI at `/`.
    Serve {
        #[command(flatten)]
        model: ModelArgs,
        /// IP to listen on; anything but loopback also needs --public.
        #[arg(long)]
        bind: Option<IpAddr>,
        #[arg(long)]
        port: Option<u16>,
        /// Allow a non-loopback bind address. Requires an API key.
        #[arg(long)]
        public: bool,
        /// Require this key as `Authorization: Bearer <key>` or `X-API-Key: <key>`.
        #[arg(long, alias = "token")]
        api_key: Option<String>,
        /// Gate GET /health behind the key as well.
        #[arg(long)]
        health_requires_key: bool,
        /// Serve HTTPS. Without --cert/--key a self-signed pair is generated next to the config.
        #[arg(long)]
        tls: bool,
        /// PEM certificate (implies --tls).
        #[arg(long, requires = "key")]
        cert: Option<PathBuf>,
        /// PEM private key (implies --tls).
        #[arg(long, requires = "cert")]
        key: Option<PathBuf>,
        /// Fork into the background after binding; writes a pid file and a log file.
        #[arg(long)]
        daemonize: bool,
        /// Serve the browser UI (overrides web.enabled).
        #[arg(long)]
        web: bool,
        /// Do not serve the browser UI (overrides web.enabled).
        #[arg(long, conflicts_with = "web")]
        no_web: bool,
        /// Password for the web UI. Empty in config means the UI is open.
        #[arg(long)]
        web_password: Option<String>,
    },
    /// Answer one System One question (any engine), or one GGUF chat turn, and print the JSON.
    Ask {
        #[command(flatten)]
        model: ModelArgs,
        /// State as text or JSON.
        #[arg(long)]
        state: Option<String>,
        /// One question definition as JSON.
        #[arg(long)]
        question: Option<String>,
        /// User message for a GGUF chat model (OpenAI `messages`). Extra; System One is portable.
        #[arg(long, conflicts_with_all = ["state", "question"])]
        prompt: Option<String>,
        /// `max_tokens` for `--prompt` (default 256).
        #[arg(long)]
        max_tokens: Option<usize>,
    },
    /// List cached checkpoints, or download / delete / update one.
    Models {
        #[command(subcommand)]
        command: Option<ModelsCommand>,
    },
    /// Manage the config file.
    Config {
        #[command(subcommand)]
        command: ConfigCommand,
    },
    /// Install a launchd or systemd unit that runs `lmr-rs serve` in the foreground.
    Service {
        #[command(subcommand)]
        command: ServiceCommand,
    },
}

/// Flags shared by `models download`, `models delete`, and `models update`.
#[derive(Args, Clone, Default)]
struct CacheModelArgs {
    /// Hugging Face repo id or a local checkpoint directory.
    #[arg(long)]
    model: Option<String>,
    /// Checkpoint subfolder inside the repo, e.g. `multilingual`.
    #[arg(long)]
    subfolder: Option<String>,
    /// Named checkpoint: minicpm5-2b, qwen3-0.6b, english, multilingual, typed-decisions.
    #[arg(long, conflicts_with = "model")]
    variant: Option<String>,
    /// GGUF filename inside the repo; overrides the variant default.
    #[arg(long)]
    filename: Option<String>,
}

#[derive(Subcommand)]
enum ModelsCommand {
    /// Fetch a checkpoint into the Hugging Face cache.
    Download {
        /// Catalog name, e.g. `minicpm5-2b` or `english`. Omit to pick from a list.
        #[arg(value_name = "VARIANT", conflicts_with_all = ["model", "variant"])]
        name: Option<String>,
        #[command(flatten)]
        model: CacheModelArgs,
    },
    /// Remove a downloaded checkpoint from the Hugging Face cache.
    Delete {
        /// Catalog name, e.g. `minicpm5-2b` or `english`. Omit to pick from downloaded models.
        #[arg(value_name = "VARIANT", conflicts_with_all = ["model", "variant"])]
        name: Option<String>,
        #[command(flatten)]
        model: CacheModelArgs,
    },
    /// Re-download a checkpoint, replacing the cached copy.
    Update {
        /// Catalog name, e.g. `minicpm5-2b` or `english`. Omit to pick from downloaded models.
        #[arg(value_name = "VARIANT", conflicts_with_all = ["model", "variant"])]
        name: Option<String>,
        #[command(flatten)]
        model: CacheModelArgs,
    },
}

#[derive(Subcommand)]
enum ConfigCommand {
    /// Write a commented default config (refuses to overwrite).
    Init,
    /// Print the resolved config path and the effective settings.
    Show,
}

#[derive(Subcommand)]
enum ServiceCommand {
    /// Write the unit file for this platform (launchd on macOS, systemd on Linux).
    Install {
        /// Install system-wide (/Library/LaunchDaemons or /etc/systemd/system) instead of per user.
        #[arg(long)]
        system: bool,
    },
}

/// Pick a device by name. `auto` prefers Metal, then CUDA, then CPU, but only when the
/// binary was built with that backend.
fn pick_device(name: &str) -> Result<Device> {
    match name {
        "cpu" => Ok(Device::Cpu),
        "metal" => Ok(Device::new_metal(0)?),
        "cuda" => Ok(Device::new_cuda(0)?),
        "auto" => {
            #[cfg(feature = "metal")]
            if let Ok(d) = Device::new_metal(0) {
                return Ok(d);
            }
            #[cfg(feature = "cuda")]
            if let Ok(d) = Device::new_cuda(0) {
                return Ok(d);
            }
            Ok(Device::Cpu)
        }
        other => bail!("unknown device {other:?}; use auto, cpu, metal, or cuda"),
    }
}

/// Resolve the config path and read it, then lay the model flags over the `[model]` section.
fn load_settings(config: &Option<PathBuf>, model: &ModelArgs) -> Result<(PathBuf, Settings)> {
    let path = config.clone().unwrap_or_else(settings::default_path);
    let mut s = Settings::load(&path)?;
    overlay_model(
        &mut s,
        model.model.as_deref(),
        model.variant.as_deref(),
        model.subfolder.as_deref(),
        model.filename.as_deref(),
        model.device.as_deref(),
    );
    Ok((path, s))
}

/// Which command is asking for a variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PromptKind {
    Download,
    Delete,
    Update,
    /// `serve` / `ask` with no model named in config or flags.
    Use,
}

impl PromptKind {
    fn verb(self) -> &'static str {
        match self {
            Self::Download => "download",
            Self::Delete => "delete",
            Self::Update => "update",
            Self::Use => "serve",
        }
    }

    fn prompt(self) -> &'static str {
        match self {
            Self::Download => "Download which model?",
            Self::Delete => "Delete which downloaded model?",
            Self::Update => "Update which downloaded model?",
            Self::Use => "Use which downloaded model?",
        }
    }

    fn empty_message(self) -> &'static str {
        match self {
            Self::Download => "no catalog entries",
            Self::Delete => "no downloaded checkpoints to delete",
            Self::Update => "no downloaded checkpoints to update",
            Self::Use => "no downloaded checkpoint; run `lmr-rs models download` to fetch one",
        }
    }

    fn downloaded_only(self) -> bool {
        matches!(self, Self::Delete | Self::Update | Self::Use)
    }

    fn need_name_message(self) -> String {
        match self {
            Self::Use => {
                "multiple downloaded models; set model.variant or pass --variant, e.g. lmr-rs serve --variant minicpm5-2b"
                    .into()
            }
            other => format!(
                "pass a catalog name, e.g. lmr-rs models {} minicpm5-2b",
                other.verb()
            ),
        }
    }
}

/// How to resolve a run when no checkpoint was named.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CachedPick {
    Empty,
    Use(&'static str),
    Prompt,
}

/// One cached checkpoint is used as-is; several need a picker; none is an error.
fn pick_cached(names: &[&'static str]) -> CachedPick {
    match names {
        [] => CachedPick::Empty,
        [name] => CachedPick::Use(*name),
        _ => CachedPick::Prompt,
    }
}

struct Choice {
    name: &'static str,
    label: String,
}

/// True when the user did not name a checkpoint on the command line.
fn needs_prompt(name: &Option<String>, model: &CacheModelArgs) -> bool {
    name.is_none() && model.model.is_none() && model.variant.is_none() && model.filename.is_none()
}

/// Catalog rows for a picker. `download` lists everything; `delete` / `update` only what is cached.
fn collect_choices(kind: PromptKind) -> Result<Vec<Choice>> {
    let mut out = Vec::new();
    for v in VARIANTS {
        let st = hub::cache_status(&v.resolve())?;
        if kind.downloaded_only() && !st.downloaded {
            continue;
        }
        out.push(Choice {
            name: v.name,
            label: choice_label(v, &st),
        });
    }
    Ok(out)
}

fn choice_label(v: &Variant, st: &hub::CacheStatus) -> String {
    let engine = match v.engine {
        settings::Engine::Laya => "laya",
        settings::Engine::Gguf => "gguf",
    };
    let cached = if st.downloaded {
        format_bytes(st.bytes)
    } else {
        "-".into()
    };
    format!("{:<16} {engine:<5} {cached:<8} {}", v.name, v.description)
}

/// Arrow-key picker. Scripts without a TTY still need an explicit catalog name.
fn prompt_variant(kind: PromptKind) -> Result<String> {
    if !io::stdin().is_terminal() || !io::stdout().is_terminal() {
        bail!("{}", kind.need_name_message());
    }
    let rows = collect_choices(kind)?;
    if rows.is_empty() {
        bail!("{}", kind.empty_message());
    }
    let labels: Vec<&str> = rows.iter().map(|c| c.label.as_str()).collect();
    let idx = dialoguer::Select::new()
        .with_prompt(kind.prompt())
        .items(&labels)
        .default(0)
        .interact()
        .context("model selection cancelled")?;
    Ok(rows[idx].name.to_string())
}

/// Fill in a missing catalog name from a TTY picker.
fn ensure_variant(
    kind: PromptKind,
    name: Option<String>,
    model: &CacheModelArgs,
) -> Result<Option<String>> {
    if !needs_prompt(&name, model) {
        return Ok(name);
    }
    Ok(Some(prompt_variant(kind)?))
}

/// Same as `load_settings`, but for `models download` / `delete` / `update`.
fn load_cache_settings(
    config: &Option<PathBuf>,
    name: Option<String>,
    model: &CacheModelArgs,
) -> Result<(PathBuf, Settings)> {
    if needs_prompt(&name, model) {
        bail!("pass a catalog name, e.g. lmr-rs models download minicpm5-2b");
    }
    let path = config.clone().unwrap_or_else(settings::default_path);
    let mut s = Settings::load(&path)?;
    let variant = name.or_else(|| model.variant.clone());
    overlay_model(
        &mut s,
        model.model.as_deref(),
        variant.as_deref(),
        model.subfolder.as_deref(),
        model.filename.as_deref(),
        None,
    );
    Ok((path, s))
}

/// Apply optional model flags onto loaded settings. `repo` wins over `variant` the same way
/// `--model` does on `download` / `serve` / `ask`.
fn overlay_model(
    s: &mut Settings,
    repo: Option<&str>,
    variant: Option<&str>,
    subfolder: Option<&str>,
    filename: Option<&str>,
    device: Option<&str>,
) {
    if let Some(repo) = repo {
        s.model.repo = repo.to_string();
        s.model.variant.clear();
    }
    if let Some(v) = variant {
        s.model.variant = v.to_string();
        s.model.repo.clear();
    }
    if let Some(sub) = subfolder {
        s.model.subfolder = sub.to_string();
    }
    if let Some(f) = filename {
        s.model.filename = f.to_string();
    }
    if let Some(d) = device {
        s.model.device = d.to_string();
    }
}

/// Fill `s.model.variant` from the Hugging Face cache when config and flags named none.
fn apply_downloaded_model(s: &mut Settings) -> Result<()> {
    if s.model_specified() {
        return Ok(());
    }
    let rows = collect_choices(PromptKind::Use)?;
    let names: Vec<&'static str> = rows.iter().map(|c| c.name).collect();
    match pick_cached(&names) {
        CachedPick::Empty => bail!("{}", PromptKind::Use.empty_message()),
        CachedPick::Use(name) => {
            eprintln!("using downloaded {name}");
            s.model.variant = name.to_string();
        }
        CachedPick::Prompt => {
            s.model.variant = prompt_variant(PromptKind::Use)?;
        }
    }
    Ok(())
}

/// Ask which catalog entry to fetch when `download` was given no checkpoint.
fn apply_download_target(s: &mut Settings) -> Result<()> {
    if s.model_specified() {
        return Ok(());
    }
    s.model.variant = prompt_variant(PromptKind::Download)?;
    Ok(())
}

/// Download (or locate) the configured checkpoint.
fn fetch(s: &Settings) -> Result<hub::ModelFiles> {
    if !s.model_specified() {
        bail!("{}", PromptKind::Use.empty_message());
    }
    hub::fetch_model(&s.resolve_model())
}

fn load_runtime(files: &hub::ModelFiles, s: &Settings) -> Result<Box<dyn Decider>> {
    let device = pick_device(&s.model.device)?;
    eprintln!("loading {} ...", files.id());
    let started = std::time::Instant::now();
    let runtime: Box<dyn Decider> = match files {
        ModelFiles::Laya(files) => {
            Box::new(Agent::load_with_policy(files, &device, s.model.policy())?)
        }
        ModelFiles::Gguf(files) => Box::new(GgufEngine::load(files, &device)?),
    };
    let device_name = runtime
        .info()
        .get("device")
        .and_then(Value::as_str)
        .unwrap_or(&s.model.device)
        .to_string();
    eprintln!(
        "ready on {device_name} in {:.1}s",
        started.elapsed().as_secs_f32()
    );
    Ok(runtime)
}

#[allow(clippy::too_many_arguments)]
fn serve(
    config_path: PathBuf,
    mut s: Settings,
    bind: Option<IpAddr>,
    port: Option<u16>,
    public: bool,
    api_key: Option<String>,
    health_requires_key: bool,
    tls_flag: bool,
    cert: Option<PathBuf>,
    key: Option<PathBuf>,
    daemonize: bool,
    web: bool,
    no_web: bool,
    web_password: Option<String>,
) -> Result<()> {
    if let Some(b) = bind {
        s.server.bind = b;
    }
    if let Some(p) = port {
        s.server.port = p;
    }
    s.server.public |= public;
    s.server.health_requires_key |= health_requires_key;
    if let Some(k) = api_key {
        s.server.api_key = k;
        s.server.api_key_env.clear();
        s.server.api_key_file.clear();
    }
    s.tls.enabled |= tls_flag || cert.is_some();
    if let (Some(c), Some(k)) = (cert, key) {
        s.tls.cert = c.display().to_string();
        s.tls.key = k.display().to_string();
    }
    if no_web {
        s.web.enabled = false;
    } else {
        s.web.enabled |= web;
    }
    if let Some(p) = web_password {
        s.web.password = p;
    }
    s.validate()?;

    let auth = Auth {
        key: s.resolve_api_key()?,
        health_requires_key: s.server.health_requires_key,
    };
    if s.server.public && !s.tls.enabled {
        eprintln!(
            "WARNING: serving on {} without TLS; the api key crosses the network in clear text. \
             Set tls.enabled = true or put a TLS-terminating proxy in front.",
            s.server.bind
        );
    }
    if s.web.enabled
        && s.web.password.trim().is_empty()
        && (s.server.public || !s.server.bind.is_loopback())
    {
        eprintln!(
            "WARNING: web UI is enabled without a password on {}; anyone who can reach it can run \
             inference. Set web.password or bind loopback.",
            s.server.bind
        );
    }
    let config_dir = config_path.parent().unwrap_or(Path::new(".")).to_path_buf();
    let tls_material = if s.tls.enabled {
        let m = tls::prepare(&s.tls, &config_dir, s.server.bind)?;
        if m.generated {
            eprintln!(
                "generated self-signed certificate {} (key {})",
                m.cert.display(),
                m.key.display()
            );
        }
        eprintln!("TLS certificate SHA-256 fingerprint: {}", m.fingerprint);
        Some(m)
    } else {
        None
    };

    // Bind before anything slow so a taken port fails immediately, and before forking so the
    // parent can report it.
    let addr = SocketAddr::new(s.server.bind, s.server.port);
    let listener = TcpListener::bind(addr).with_context(|| format!("binding {addr}"))?;
    let bound = listener.local_addr()?;
    let scheme = if s.tls.enabled { "https" } else { "http" };

    // Fork before the download and the model load: neither the HTTP client's threads nor a GPU
    // context survive fork(), and macOS aborts a child that touches Foundation afterwards.
    if daemonize {
        let pid = s.pid_file(&config_path);
        let log = s.log_file(&config_path);
        daemonize_into(&pid, &log)?;
        eprintln!(
            "lmr-rs daemon pid {} will listen on {scheme}://{bound}",
            std::process::id()
        );
    }
    let files = fetch(&s)?;
    let runtime = load_runtime(&files, &s)?;
    let web = WebConfig::new(s.web.enabled, Some(s.web.password.clone()), s.tls.enabled);
    let server = HttpServer::from_listener(listener, runtime, auth, tls_material, web);
    eprintln!(
        "listening on {scheme}://{bound}  (POST /v1/systemone, POST /v1/chat/completions, GET /v1/models, GET /health)"
    );
    if s.web.enabled {
        eprintln!("web ui: {scheme}://{bound}/");
    }
    server.run()
}

/// Fork to the background. The parent prints where the pid and log went and exits.
#[cfg(unix)]
fn daemonize_into(pid_file: &Path, log_file: &Path) -> Result<()> {
    if let Some(dir) = log_file.parent() {
        fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    let open = || {
        fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_file)
            .with_context(|| format!("opening {}", log_file.display()))
    };
    let (out, err) = (open()?, open()?);
    eprintln!(
        "daemonizing: pid file {}, log {}",
        pid_file.display(),
        log_file.display()
    );
    daemonize::Daemonize::new()
        .pid_file(pid_file)
        .working_directory(std::env::current_dir()?)
        .stdout(out)
        .stderr(err)
        .start()
        .context("daemonizing")?;
    Ok(())
}

#[cfg(not(unix))]
fn daemonize_into(_pid_file: &Path, _log_file: &Path) -> Result<()> {
    bail!("--daemonize is only supported on Unix; run in the foreground under a service manager")
}

/// Write `path` unless it already exists.
fn write_new(path: &Path, contents: &str) -> Result<()> {
    if path.exists() {
        bail!("{} already exists; remove it first", path.display());
    }
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    fs::write(path, contents).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

fn home() -> Result<PathBuf> {
    std::env::var("HOME")
        .map(PathBuf::from)
        .context("HOME is not set")
}

/// Fill the unit template for this platform and write it where the service manager looks.
fn service_install(config_path: &Path, s: &Settings, system: bool) -> Result<()> {
    let bin = std::env::current_exe().context("locating this binary")?;
    let config_abs = fs::canonicalize(config_path).with_context(|| {
        format!(
            "{} does not exist; run `lmr-rs config init` first",
            config_path.display()
        )
    })?;
    let log = s.log_file(&config_abs);
    let (template, target, enable) = if cfg!(target_os = "macos") {
        let target = if system {
            PathBuf::from("/Library/LaunchDaemons/lmr-rs.plist")
        } else {
            home()?.join("Library/LaunchAgents/lmr-rs.plist")
        };
        let enable = format!("launchctl load -w {}", target.display());
        (include_str!("../contrib/lmr-rs.plist"), target, enable)
    } else {
        let target = if system {
            PathBuf::from("/etc/systemd/system/lmr-rs.service")
        } else {
            home()?.join(".config/systemd/user/lmr-rs.service")
        };
        let flag = if system { "" } else { "--user " };
        let enable =
            format!("systemctl {flag}daemon-reload && systemctl {flag}enable --now lmr-rs");
        (include_str!("../contrib/lmr-rs.service"), target, enable)
    };
    let unit = template
        .replace("{{BIN}}", &bin.display().to_string())
        .replace("{{CONFIG}}", &config_abs.display().to_string())
        .replace("{{LOG}}", &log.display().to_string());
    write_new(&target, &unit)?;
    println!("wrote {}", target.display());
    println!("enable it with:\n  {enable}");
    Ok(())
}

/// Print the catalog with Hugging Face cache status, then the configured spec.
fn print_models(s: &Settings) -> Result<()> {
    for v in VARIANTS {
        let selected = s.model.repo.is_empty() && s.model.variant == v.name;
        let mark = if selected { "*" } else { " " };
        let st = hub::cache_status(&v.resolve())?;
        println!("{mark} {}", choice_label(v, &st));
    }
    if s.model_specified() {
        let spec = s.resolve_model();
        let extra = match hub::cache_status(&spec) {
            Ok(st) if st.downloaded && st.source == hub::CacheSource::Local => "  (local)",
            Ok(st) if st.downloaded => "",
            _ => "  (not downloaded)",
        };
        println!("\nconfigured: {}{extra}", spec_label(&spec));
    } else {
        println!("\nconfigured: (none; serve uses a downloaded checkpoint)");
    }
    Ok(())
}

/// One-line id used in `models` output and delete confirmations.
fn spec_label(spec: &ResolvedModel) -> String {
    match spec {
        ResolvedModel::Laya { repo, subfolder } => match subfolder {
            Some(sub) => format!("{repo}/{sub}"),
            None => repo.clone(),
        },
        ResolvedModel::Gguf {
            repo,
            filename,
            tokenizer_repo,
        } => format!("{repo}/{filename}  tokenizer {tokenizer_repo}"),
    }
}

/// Compact SI size for the cache column (`843 MB`, `1.5 GB`).
fn format_bytes(n: u64) -> String {
    const K: f64 = 1000.0;
    let x = n as f64;
    if x >= K * K * K {
        format!("{:.1} GB", x / (K * K * K))
    } else if x >= K * K {
        format!("{:.0} MB", x / (K * K))
    } else if x >= K {
        format!("{:.0} KB", x / K)
    } else {
        format!("{n} B")
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Download { model } => {
            let (_, mut s) = load_settings(&cli.config, &model)?;
            apply_download_target(&mut s)?;
            s.validate()?;
            let files = fetch(&s)?;
            println!("{}", files.weights().display());
        }
        Command::Serve {
            model,
            bind,
            port,
            public,
            api_key,
            health_requires_key,
            tls,
            cert,
            key,
            daemonize,
            web,
            no_web,
            web_password,
        } => {
            let (path, mut s) = load_settings(&cli.config, &model)?;
            apply_downloaded_model(&mut s)?;
            serve(
                path,
                s,
                bind,
                port,
                public,
                api_key,
                health_requires_key,
                tls,
                cert,
                key,
                daemonize,
                web,
                no_web,
                web_password,
            )?;
        }
        Command::Ask {
            model,
            state,
            question,
            prompt,
            max_tokens,
        } => {
            let (_, mut s) = load_settings(&cli.config, &model)?;
            apply_downloaded_model(&mut s)?;
            s.validate()?;
            let mut runtime = load_runtime(&fetch(&s)?, &s)?;
            let answer = if let Some(prompt) = prompt {
                let chat = runtime.as_chat().ok_or_else(|| {
                    anyhow::anyhow!(
                        "this checkpoint is a System One model; use --state and --question"
                    )
                })?;
                chat.chat(
                    &[ChatMessage {
                        role: "user".into(),
                        content: prompt,
                    }],
                    &ChatOpts {
                        max_tokens,
                        ..ChatOpts::default()
                    },
                )?
            } else {
                let (state, question) = match (state, question) {
                    (Some(state), Some(question)) => (state, question),
                    _ => bail!("ask needs --prompt, or both --state and --question"),
                };
                let state: Value = serde_json::from_str(&state).unwrap_or(Value::String(state));
                let question: Value = serde_json::from_str(&question)?;
                let mut questions = serde_json::Map::new();
                questions.insert("q".into(), question);
                runtime.system_one(&state, &questions)?
            };
            println!("{}", serde_json::to_string_pretty(&answer)?);
        }
        Command::Models { command } => match command {
            None => {
                let (_, s) = load_settings(&cli.config, &ModelArgs::default())?;
                print_models(&s)?;
            }
            Some(ModelsCommand::Download { name, model }) => {
                let name = ensure_variant(PromptKind::Download, name, &model)?;
                let (_, s) = load_cache_settings(&cli.config, name, &model)?;
                s.validate()?;
                let files = fetch(&s)?;
                println!("{}", files.weights().display());
            }
            Some(ModelsCommand::Delete { name, model }) => {
                let name = ensure_variant(PromptKind::Delete, name, &model)?;
                let (_, s) = load_cache_settings(&cli.config, name, &model)?;
                s.validate()?;
                let spec = s.resolve_model();
                let report = hub::delete_model(&spec)?;
                println!(
                    "removed {} file{} ({})  {}",
                    report.files,
                    if report.files == 1 { "" } else { "s" },
                    format_bytes(report.bytes),
                    spec_label(&spec)
                );
            }
            Some(ModelsCommand::Update { name, model }) => {
                let name = ensure_variant(PromptKind::Update, name, &model)?;
                let (_, s) = load_cache_settings(&cli.config, name, &model)?;
                s.validate()?;
                let files = hub::update_model(&s.resolve_model())?;
                println!("{}", files.weights().display());
            }
        },
        Command::Config { command } => {
            let path = cli.config.unwrap_or_else(settings::default_path);
            match command {
                ConfigCommand::Init => {
                    write_new(&path, settings::TEMPLATE)?;
                    println!("wrote {}", path.display());
                }
                ConfigCommand::Show => {
                    let s = Settings::load(&path)?;
                    println!("# {}", path.display());
                    print!("{}", toml::to_string_pretty(&s)?);
                }
            }
        }
        Command::Service { command } => {
            let path = cli.config.unwrap_or_else(settings::default_path);
            let s = Settings::load(&path)?;
            match command {
                ServiceCommand::Install { system } => service_install(&path, &s, system)?,
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Cli {
        Cli::try_parse_from(args).unwrap()
    }

    #[test]
    fn models_download_takes_a_catalog_name() {
        let cli = parse(&["lmr-rs", "models", "download", "english"]);
        match cli.command {
            Command::Models {
                command: Some(ModelsCommand::Download { name, .. }),
            } => assert_eq!(name.as_deref(), Some("english")),
            _ => panic!("expected models download"),
        }
    }

    #[test]
    fn models_download_without_a_name_wants_a_picker() {
        let cli = parse(&["lmr-rs", "models", "download"]);
        match cli.command {
            Command::Models {
                command: Some(ModelsCommand::Download { name, model }),
            } => assert!(needs_prompt(&name, &model)),
            _ => panic!("expected models download"),
        }
        let named = parse(&["lmr-rs", "models", "download", "english"]);
        match named.command {
            Command::Models {
                command: Some(ModelsCommand::Download { name, model }),
            } => assert!(!needs_prompt(&name, &model)),
            _ => panic!("expected named download"),
        }
    }

    #[test]
    fn download_lists_the_whole_catalog() {
        let names: Vec<_> = VARIANTS.iter().map(|v| v.name).collect();
        assert_eq!(names_for(PromptKind::Download, &[]), names);
    }

    #[test]
    fn delete_and_update_list_downloaded_only() {
        assert_eq!(
            names_for(PromptKind::Delete, &["english", "qwen3-0.6b"]),
            ["qwen3-0.6b", "english"]
        );
        assert!(names_for(PromptKind::Update, &[]).is_empty());
        assert!(PromptKind::Delete.empty_message().contains("delete"));
    }

    #[test]
    fn serve_lists_downloaded_only() {
        assert_eq!(names_for(PromptKind::Use, &["qwen3-0.6b"]), ["qwen3-0.6b"]);
        assert!(names_for(PromptKind::Use, &[]).is_empty());
        assert!(PromptKind::Use.empty_message().contains("models download"));
        assert!(PromptKind::Use.need_name_message().contains("--variant"));
    }

    #[test]
    fn one_cached_model_is_used_without_a_prompt() {
        assert_eq!(pick_cached(&["qwen3-0.6b"]), CachedPick::Use("qwen3-0.6b"));
        assert_eq!(pick_cached(&[]), CachedPick::Empty);
        assert_eq!(pick_cached(&["english", "qwen3-0.6b"]), CachedPick::Prompt);
    }

    #[test]
    fn overlay_model_flag_clears_a_configured_variant() {
        let mut s = Settings::parse("[model]\nvariant = \"multilingual\"\n").unwrap();
        overlay_model(&mut s, Some("a/b"), None, None, None, None);
        assert_eq!(s.model.repo, "a/b");
        assert!(s.model.variant.is_empty());
        assert!(s.model_specified());
        assert!(s.validate().is_ok());
    }

    fn names_for(kind: PromptKind, downloaded: &[&str]) -> Vec<&'static str> {
        VARIANTS
            .iter()
            .filter(|v| !kind.downloaded_only() || downloaded.contains(&v.name))
            .map(|v| v.name)
            .collect()
    }

    #[test]
    fn choice_label_shows_engine_and_size() {
        let v = settings::variant("english").unwrap();
        let st = hub::CacheStatus {
            downloaded: true,
            bytes: 843_000_000,
            source: hub::CacheSource::Hub,
        };
        let label = choice_label(v, &st);
        assert!(label.contains("english"));
        assert!(label.contains("laya"));
        assert!(label.contains("843 MB"));
    }
}
