//! `laya-rs`: download a Laya checkpoint, serve it (foreground or daemonized, plain or TLS,
//! with an optional API key), ask it one question, or manage its config and service files.

use std::fs;
use std::net::{IpAddr, SocketAddr, TcpListener};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use candle_core::Device;
use clap::{Args, Parser, Subcommand};
use laya_rs::settings::{self, Settings, VARIANTS};
use laya_rs::{hub, tls, Agent, Auth, Decider, HttpServer};
use serde_json::Value;

#[derive(Parser)]
#[command(name = "laya-rs", version, about = "Native Rust inference for the Laya decision model")]
struct Cli {
    /// Config file (default: $LAYA_RS_CONFIG or ~/.config/laya-rs/config.toml).
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
    /// Named checkpoint: english, multilingual, or typed-decisions (see `laya-rs models`).
    #[arg(long, conflicts_with = "model")]
    variant: Option<String>,
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
    /// Load the model and serve `POST /v1/systemone`.
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
    },
    /// Answer one question from the command line and print the JSON.
    Ask {
        #[command(flatten)]
        model: ModelArgs,
        /// State as text or JSON.
        #[arg(long)]
        state: String,
        /// One question definition as JSON, e.g. '{"type":"noul","instructions":"Is it spam?"}'.
        #[arg(long)]
        question: String,
    },
    /// List the Laya checkpoints this binary can serve.
    Models,
    /// Manage the config file.
    Config {
        #[command(subcommand)]
        command: ConfigCommand,
    },
    /// Install a launchd or systemd unit that runs `laya-rs serve` in the foreground.
    Service {
        #[command(subcommand)]
        command: ServiceCommand,
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
    if let Some(repo) = &model.model {
        s.model.repo = repo.clone();
        s.model.variant = "english".into();
    }
    if let Some(v) = &model.variant {
        s.model.variant = v.clone();
        s.model.repo.clear();
    }
    if let Some(sub) = &model.subfolder {
        s.model.subfolder = sub.clone();
    }
    if let Some(d) = &model.device {
        s.model.device = d.clone();
    }
    Ok((path, s))
}

/// Download (or locate) the configured checkpoint.
fn fetch(s: &Settings) -> Result<hub::CheckpointFiles> {
    let (repo, subfolder) = s.resolve_model();
    hub::fetch(&repo, subfolder.as_deref())
}

fn load_agent(files: &hub::CheckpointFiles, device: &str) -> Result<Agent> {
    let device = pick_device(device)?;
    eprintln!("loading {} ...", files.id);
    let started = std::time::Instant::now();
    let agent = Agent::load(files, &device)?;
    eprintln!("ready on {} in {:.1}s", agent.device_name(), started.elapsed().as_secs_f32());
    Ok(agent)
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
    let config_dir = config_path.parent().unwrap_or(Path::new(".")).to_path_buf();
    let tls_material = if s.tls.enabled {
        let m = tls::prepare(&s.tls, &config_dir, s.server.bind)?;
        if m.generated {
            eprintln!("generated self-signed certificate {} (key {})", m.cert.display(), m.key.display());
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
        eprintln!("laya-rs daemon pid {} will listen on {scheme}://{bound}", std::process::id());
    }
    let files = fetch(&s)?;
    let agent = load_agent(&files, &s.model.device)?;
    let server = HttpServer::from_listener(listener, Box::new(agent), auth, tls_material);
    eprintln!("listening on {scheme}://{bound}  (POST /v1/systemone, GET /health)");
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
    eprintln!("daemonizing: pid file {}, log {}", pid_file.display(), log_file.display());
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
    std::env::var("HOME").map(PathBuf::from).context("HOME is not set")
}

/// Fill the unit template for this platform and write it where the service manager looks.
fn service_install(config_path: &Path, s: &Settings, system: bool) -> Result<()> {
    let bin = std::env::current_exe().context("locating this binary")?;
    let config_abs = fs::canonicalize(config_path)
        .with_context(|| format!("{} does not exist; run `laya-rs config init` first", config_path.display()))?;
    let log = s.log_file(&config_abs);
    let (template, target, enable) = if cfg!(target_os = "macos") {
        let target = if system {
            PathBuf::from("/Library/LaunchDaemons/laya-rs.plist")
        } else {
            home()?.join("Library/LaunchAgents/laya-rs.plist")
        };
        let enable = format!("launchctl load -w {}", target.display());
        (include_str!("../contrib/laya-rs.plist"), target, enable)
    } else {
        let target = if system {
            PathBuf::from("/etc/systemd/system/laya-rs.service")
        } else {
            home()?.join(".config/systemd/user/laya-rs.service")
        };
        let flag = if system { "" } else { "--user " };
        let enable = format!("systemctl {flag}daemon-reload && systemctl {flag}enable --now laya-rs");
        (include_str!("../contrib/laya-rs.service"), target, enable)
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

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Download { model } => {
            let (_, s) = load_settings(&cli.config, &model)?;
            s.validate()?;
            let files = fetch(&s)?;
            println!("{}", files.weights.display());
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
        } => {
            let (path, s) = load_settings(&cli.config, &model)?;
            serve(path, s, bind, port, public, api_key, health_requires_key, tls, cert, key, daemonize)?;
        }
        Command::Ask { model, state, question } => {
            let (_, s) = load_settings(&cli.config, &model)?;
            s.validate()?;
            let agent = load_agent(&fetch(&s)?, &s.model.device)?;
            let state: Value = serde_json::from_str(&state).unwrap_or(Value::String(state));
            let question: Value = serde_json::from_str(&question)?;
            let mut questions = serde_json::Map::new();
            questions.insert("q".into(), question);
            let answer = agent.system_one(&state, &questions)?;
            println!("{}", serde_json::to_string_pretty(&answer)?);
        }
        Command::Models => {
            let (_, s) = load_settings(&cli.config, &ModelArgs::default())?;
            let (repo, sub) = s.resolve_model();
            for v in VARIANTS {
                let selected = s.model.repo.is_empty() && s.model.variant == v.name;
                let mark = if selected { "*" } else { " " };
                println!("{mark} {:<16} {}", v.name, v.description);
            }
            match sub {
                Some(sub) => println!("\nconfigured: {repo} / {sub}"),
                None => println!("\nconfigured: {repo}"),
            }
        }
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
