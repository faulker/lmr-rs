//! Locating a checkpoint: a local directory, or the Hugging Face cache (downloading on first
//! use). Tokens come from the environment through `hf-hub`; nothing here reads or prints them.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{anyhow, bail, Context, Result};
use hf_hub::progress::{DownloadEvent, ProgressEvent, ProgressHandler};
use hf_hub::{split_id, HFClientSync, HFError};

/// Default checkpoint (English, ModernBERT-large).
pub const DEFAULT_MODEL: &str = "convaiinnovations/laya";

const CONFIG: &str = "rl_agent_config.json";
const ENCODER_CONFIG: &str = "encoder/config.json";
const TOKENIZER: &str = "tokenizer/tokenizer.json";
const TOKENIZER_CONFIG: &str = "tokenizer/tokenizer_config.json";
const WEIGHTS: &str = "model.safetensors";

/// Local paths of everything `Agent::load` reads.
#[derive(Debug, Clone)]
pub struct CheckpointFiles {
    /// Human readable id, e.g. `convaiinnovations/laya/multilingual` or a directory path.
    pub id: String,
    pub config: PathBuf,
    pub encoder_config: PathBuf,
    pub tokenizer: PathBuf,
    pub tokenizer_config: Option<PathBuf>,
    pub weights: PathBuf,
}

/// Resolve `model` (an HF repo id or a local directory) and an optional subfolder, downloading
/// missing files into the HF cache. Prints download progress to stderr.
pub fn fetch(model: &str, subfolder: Option<&str>) -> Result<CheckpointFiles> {
    let local = Path::new(model);
    if local.is_dir() {
        return from_dir(local, subfolder);
    }
    let (owner, name) = split_id(model);
    if owner.is_empty() || name.is_empty() {
        bail!("{model:?} is neither a directory nor an owner/name Hugging Face id");
    }
    let client = HFClientSync::new().context("creating Hugging Face client")?;
    let repo = client.model(owner, name);
    let rel = |file: &str| match subfolder {
        Some(sub) => format!("{sub}/{file}"),
        None => file.to_string(),
    };
    let get = |file: &str| -> Result<PathBuf> {
        let path = rel(file);
        repo.download_file()
            .filename(path.clone())
            .progress(StderrProgress::default())
            .send()
            .map_err(|e| anyhow!("fetching {path} from {model}: {e}"))
    };
    let config = get(CONFIG)?;
    let encoder_config = get(ENCODER_CONFIG)?;
    let tokenizer = get(TOKENIZER)?;
    let tokenizer_config = match repo
        .download_file()
        .filename(rel(TOKENIZER_CONFIG))
        .send()
    {
        Ok(p) => Some(p),
        Err(HFError::EntryNotFound { .. }) => None,
        Err(e) => return Err(anyhow!("fetching {} from {model}: {e}", rel(TOKENIZER_CONFIG))),
    };
    let weights = get(WEIGHTS)?;
    let id = match subfolder {
        Some(sub) => format!("{model}/{sub}"),
        None => model.to_string(),
    };
    Ok(CheckpointFiles {
        id,
        config,
        encoder_config,
        tokenizer,
        tokenizer_config,
        weights,
    })
}

/// Use a checkpoint that is already on disk, laid out like the HF repo.
fn from_dir(dir: &Path, subfolder: Option<&str>) -> Result<CheckpointFiles> {
    let dir = match subfolder {
        Some(sub) => dir.join(sub),
        None => dir.to_path_buf(),
    };
    let need = |file: &str| -> Result<PathBuf> {
        let p = dir.join(file);
        if p.is_file() {
            Ok(p)
        } else {
            bail!("missing {}", p.display())
        }
    };
    let tokenizer_config = dir.join(TOKENIZER_CONFIG);
    Ok(CheckpointFiles {
        id: dir.display().to_string(),
        config: need(CONFIG)?,
        encoder_config: need(ENCODER_CONFIG)?,
        tokenizer: need(TOKENIZER)?,
        tokenizer_config: tokenizer_config.is_file().then_some(tokenizer_config),
        weights: need(WEIGHTS)?,
    })
}

/// Prints one line per 5% of a file so an 800 MB first download does not look hung.
#[derive(Default)]
struct StderrProgress {
    last_pct: Mutex<u64>,
}

impl ProgressHandler for StderrProgress {
    fn on_progress(&self, event: &ProgressEvent) {
        let ProgressEvent::Download(DownloadEvent::Progress { files }) = event else {
            return;
        };
        for f in files {
            if f.total_bytes == 0 {
                continue;
            }
            let pct = f.bytes_completed * 100 / f.total_bytes;
            let mut last = self.last_pct.lock().unwrap_or_else(|e| e.into_inner());
            if pct >= *last + 5 || (pct == 100 && *last != 100) {
                *last = pct;
                eprintln!("downloading {}: {pct}%", f.filename);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_dir_needs_all_files() {
        let dir = std::env::temp_dir().join(format!("laya-rs-test-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("encoder")).unwrap();
        std::fs::create_dir_all(dir.join("tokenizer")).unwrap();
        assert!(fetch(dir.to_str().unwrap(), None).is_err());
        for f in [CONFIG, ENCODER_CONFIG, TOKENIZER, WEIGHTS] {
            std::fs::write(dir.join(f), b"{}").unwrap();
        }
        let files = fetch(dir.to_str().unwrap(), None).unwrap();
        assert!(files.tokenizer_config.is_none());
        assert!(files.weights.ends_with(WEIGHTS));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
