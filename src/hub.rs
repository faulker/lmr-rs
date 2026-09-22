//! Locating a checkpoint: a local directory, or the Hugging Face cache (downloading on first
//! use). Tokens come from the environment through `hf-hub`; nothing here reads or prints them.
//! Cached copies can be inspected, deleted, or force-updated without touching local directories.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{anyhow, bail, Context, Result};
use hf_hub::progress::{DownloadEvent, ProgressEvent, ProgressHandler};
use hf_hub::{split_id, HFClientSync, HFError};

use crate::settings::ResolvedModel;

/// Default checkpoint (English, ModernBERT-large).
pub const DEFAULT_MODEL: &str = "convaiinnovations/laya";

const CONFIG: &str = "rl_agent_config.json";
const ENCODER_CONFIG: &str = "encoder/config.json";
const TOKENIZER: &str = "tokenizer/tokenizer.json";
const TOKENIZER_CONFIG: &str = "tokenizer/tokenizer_config.json";
const WEIGHTS: &str = "model.safetensors";
const GGUF_TOKENIZER: &str = "tokenizer.json";
const GGUF_TOKENIZER_CONFIG: &str = "tokenizer_config.json";
const RERANK_CONFIG: &str = "config.json";

/// Local paths of a GGUF chat checkpoint plus its Hugging Face tokenizer.
#[derive(Debug, Clone)]
pub struct GgufFiles {
    /// Human readable id, e.g. `Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf`.
    pub id: String,
    pub weights: PathBuf,
    pub tokenizer: PathBuf,
    pub tokenizer_config: Option<PathBuf>,
}

/// Local paths of a cross-encoder reranker (HF `config.json` + safetensors).
#[derive(Debug, Clone)]
pub struct RerankFiles {
    /// Human readable id, e.g. `BAAI/bge-reranker-v2-m3` or a directory path.
    pub id: String,
    pub config: PathBuf,
    pub tokenizer: PathBuf,
    pub tokenizer_config: Option<PathBuf>,
    pub weights: PathBuf,
}

/// A Laya decision checkpoint, a GGUF chat model, or a reranker.
#[derive(Debug, Clone)]
pub enum ModelFiles {
    Laya(CheckpointFiles),
    Gguf(GgufFiles),
    Rerank(RerankFiles),
}

impl ModelFiles {
    /// Human readable id for logs and `/health`.
    pub fn id(&self) -> &str {
        match self {
            Self::Laya(f) => &f.id,
            Self::Gguf(f) => &f.id,
            Self::Rerank(f) => &f.id,
        }
    }

    /// Path printed by `lmr-rs download`.
    pub fn weights(&self) -> &Path {
        match self {
            Self::Laya(f) => &f.weights,
            Self::Gguf(f) => &f.weights,
            Self::Rerank(f) => &f.weights,
        }
    }
}

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

/// Whether a checkpoint is on disk, and how much unique blob space it uses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheStatus {
    pub downloaded: bool,
    pub bytes: u64,
    /// `hub` for Hugging Face cache, `local` for a directory or file the user pointed at.
    pub source: CacheSource,
}

/// Where `CacheStatus` found the files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheSource {
    Hub,
    Local,
}

/// What `delete_model` removed from the Hugging Face cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeleteReport {
    pub files: usize,
    pub bytes: u64,
}

/// Fetch whatever `Settings::resolve_model` selected.
pub fn fetch_model(spec: &ResolvedModel) -> Result<ModelFiles> {
    fetch_model_inner(spec, false)
}

/// Re-download from Hugging Face, replacing the cached copy. Local paths are refused.
pub fn update_model(spec: &ResolvedModel) -> Result<ModelFiles> {
    if spec_is_local(spec) {
        bail!("local checkpoints are not updated from Hugging Face");
    }
    fetch_model_inner(spec, true)
}

fn fetch_model_inner(spec: &ResolvedModel, force: bool) -> Result<ModelFiles> {
    match spec {
        ResolvedModel::Laya { repo, subfolder } => Ok(ModelFiles::Laya(fetch_laya(
            repo,
            subfolder.as_deref(),
            force,
        )?)),
        ResolvedModel::Gguf {
            repo,
            filename,
            tokenizer_repo,
        } => Ok(ModelFiles::Gguf(fetch_gguf_inner(
            repo,
            filename,
            tokenizer_repo,
            force,
        )?)),
        ResolvedModel::Rerank { repo } => Ok(ModelFiles::Rerank(fetch_rerank_inner(repo, force)?)),
    }
}

/// Download (or open) a reranker: `config.json`, `tokenizer.json`, `model.safetensors`.
pub fn fetch_rerank(repo: &str) -> Result<RerankFiles> {
    fetch_rerank_inner(repo, false)
}

fn fetch_rerank_inner(model: &str, force: bool) -> Result<RerankFiles> {
    let local = Path::new(model);
    if local.is_dir() {
        if force {
            bail!("local checkpoints are not updated from Hugging Face");
        }
        return rerank_from_dir(local);
    }
    let repo = hf_model(model)?;
    let config = download(&repo, RERANK_CONFIG, model, force, true)?;
    let tokenizer = download(&repo, GGUF_TOKENIZER, model, force, true)?;
    let tokenizer_config = download_optional(&repo, GGUF_TOKENIZER_CONFIG, model, force)?;
    let weights = download(&repo, WEIGHTS, model, force, true)?;
    Ok(RerankFiles {
        id: model.to_string(),
        config,
        tokenizer,
        tokenizer_config,
        weights,
    })
}

/// Use a reranker that is already on disk, laid out like the HF repo.
fn rerank_from_dir(dir: &Path) -> Result<RerankFiles> {
    let need = |file: &str| -> Result<PathBuf> {
        let p = dir.join(file);
        if p.is_file() {
            Ok(p)
        } else {
            bail!("missing {}", p.display())
        }
    };
    let tokenizer_config = dir.join(GGUF_TOKENIZER_CONFIG);
    Ok(RerankFiles {
        id: dir.display().to_string(),
        config: need(RERANK_CONFIG)?,
        tokenizer: need(GGUF_TOKENIZER)?,
        tokenizer_config: tokenizer_config.is_file().then_some(tokenizer_config),
        weights: need(WEIGHTS)?,
    })
}

/// Resolve `model` (an HF repo id or a local directory) and an optional subfolder, downloading
/// missing files into the HF cache. Prints download progress to stderr.
pub fn fetch(model: &str, subfolder: Option<&str>) -> Result<CheckpointFiles> {
    fetch_laya(model, subfolder, false)
}

fn fetch_laya(model: &str, subfolder: Option<&str>, force: bool) -> Result<CheckpointFiles> {
    let local = Path::new(model);
    if local.is_dir() {
        if force {
            bail!("local checkpoints are not updated from Hugging Face");
        }
        return from_dir(local, subfolder);
    }
    let repo = hf_model(model)?;
    let rel = |file: &str| match subfolder {
        Some(sub) => format!("{sub}/{file}"),
        None => file.to_string(),
    };
    let get = |file: &str| download(&repo, &rel(file), model, force, true);
    let config = get(CONFIG)?;
    let encoder_config = get(ENCODER_CONFIG)?;
    let tokenizer = get(TOKENIZER)?;
    let tokenizer_config = download_optional(&repo, &rel(TOKENIZER_CONFIG), model, force)?;
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

/// Download (or open) a GGUF file and the tokenizer that goes with it.
pub fn fetch_gguf(repo: &str, filename: &str, tokenizer_repo: &str) -> Result<GgufFiles> {
    fetch_gguf_inner(repo, filename, tokenizer_repo, false)
}

fn fetch_gguf_inner(
    repo: &str,
    filename: &str,
    tokenizer_repo: &str,
    force: bool,
) -> Result<GgufFiles> {
    if filename.is_empty() {
        bail!("GGUF fetch needs a filename");
    }
    let weights = locate_gguf(repo, filename, force)?;
    let (tokenizer, tokenizer_config) = locate_tokenizer(tokenizer_repo, force)?;
    let id = if Path::new(repo).is_file() {
        repo.to_string()
    } else {
        format!("{repo}/{filename}")
    };
    Ok(GgufFiles {
        id,
        weights,
        tokenizer,
        tokenizer_config,
    })
}

/// Open a local GGUF file, or a file inside a local directory, or download it from HF.
fn locate_gguf(repo: &str, filename: &str, force: bool) -> Result<PathBuf> {
    let path = Path::new(repo);
    if path.is_file() {
        if force {
            bail!("local checkpoints are not updated from Hugging Face");
        }
        return Ok(path.to_path_buf());
    }
    if path.is_dir() {
        if force {
            bail!("local checkpoints are not updated from Hugging Face");
        }
        let p = path.join(filename);
        if p.is_file() {
            return Ok(p);
        }
        bail!("missing {}", p.display());
    }
    download_hf_file(repo, filename, force)
}

/// Open `tokenizer.json` next to a local GGUF, or download it from the tokenizer repo.
fn locate_tokenizer(tokenizer_repo: &str, force: bool) -> Result<(PathBuf, Option<PathBuf>)> {
    let path = Path::new(tokenizer_repo);
    let dir = if path.is_file() {
        path.parent().unwrap_or(path)
    } else if path.is_dir() {
        path
    } else {
        let repo = hf_model(tokenizer_repo)?;
        let tokenizer = download(&repo, GGUF_TOKENIZER, tokenizer_repo, force, true)?;
        let tokenizer_config =
            download_optional(&repo, GGUF_TOKENIZER_CONFIG, tokenizer_repo, force)?;
        return Ok((tokenizer, tokenizer_config));
    };
    if force {
        bail!("local checkpoints are not updated from Hugging Face");
    }
    let tokenizer = dir.join(GGUF_TOKENIZER);
    if !tokenizer.is_file() {
        bail!(
            "missing {} (GGUF chat needs a Hugging Face tokenizer.json)",
            tokenizer.display()
        );
    }
    let cfg = dir.join(GGUF_TOKENIZER_CONFIG);
    Ok((tokenizer, cfg.is_file().then_some(cfg)))
}

fn hf_model(model: &str) -> Result<hf_hub::HFRepositorySync<hf_hub::RepoTypeModel>> {
    let (owner, name) = split_id(model);
    if owner.is_empty() || name.is_empty() {
        bail!("{model:?} is neither a directory nor an owner/name Hugging Face id");
    }
    let client = HFClientSync::new().context("creating Hugging Face client")?;
    Ok(client.model(owner, name))
}

fn download(
    repo: &hf_hub::HFRepositorySync<hf_hub::RepoTypeModel>,
    filename: &str,
    model: &str,
    force: bool,
    progress: bool,
) -> Result<PathBuf> {
    let req = repo
        .download_file()
        .filename(filename.to_string())
        .force_download(force);
    let result = if progress {
        req.progress(StderrProgress::default()).send()
    } else {
        req.send()
    };
    result.map_err(|e| anyhow!("fetching {filename} from {model}: {e}"))
}

fn download_optional(
    repo: &hf_hub::HFRepositorySync<hf_hub::RepoTypeModel>,
    filename: &str,
    model: &str,
    force: bool,
) -> Result<Option<PathBuf>> {
    match repo
        .download_file()
        .filename(filename.to_string())
        .force_download(force)
        .send()
    {
        Ok(p) => Ok(Some(p)),
        Err(HFError::EntryNotFound { .. }) => Ok(None),
        Err(e) => Err(anyhow!("fetching {filename} from {model}: {e}")),
    }
}

/// Download one file from an `owner/name` Hugging Face repo into the shared cache.
fn download_hf_file(model: &str, filename: &str, force: bool) -> Result<PathBuf> {
    let repo = hf_model(model)?;
    download(&repo, filename, model, force, true)
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

/// Look a resolved spec up on disk without downloading.
pub fn cache_status(spec: &ResolvedModel) -> Result<CacheStatus> {
    cache_status_in(&hf_hub::resolve_cache_dir(), spec)
}

fn cache_status_in(cache: &Path, spec: &ResolvedModel) -> Result<CacheStatus> {
    if spec_is_local(spec) {
        return local_status(spec);
    }
    let files = planned_files(spec)?;
    let Some(snap) = snapshot_with(&cache_root(cache, &files.repo), &files.required) else {
        return Ok(CacheStatus {
            downloaded: false,
            bytes: 0,
            source: CacheSource::Hub,
        });
    };
    let mut bytes = unique_size(&snap, &files.required);
    if let ResolvedModel::Gguf { tokenizer_repo, .. } = spec {
        let tok_files = vec![GGUF_TOKENIZER.to_string()];
        let Some(tok) = snapshot_with(&cache_root(cache, tokenizer_repo), &tok_files) else {
            return Ok(CacheStatus {
                downloaded: false,
                bytes: 0,
                source: CacheSource::Hub,
            });
        };
        if tokenizer_repo != &files.repo {
            bytes += unique_size(&tok, &tok_files);
        }
    }
    Ok(CacheStatus {
        downloaded: true,
        bytes,
        source: CacheSource::Hub,
    })
}

/// Remove this checkpoint's files from the Hugging Face cache. Local paths are refused.
pub fn delete_model(spec: &ResolvedModel) -> Result<DeleteReport> {
    delete_model_in(&hf_hub::resolve_cache_dir(), spec)
}

fn delete_model_in(cache: &Path, spec: &ResolvedModel) -> Result<DeleteReport> {
    if spec_is_local(spec) {
        bail!("refusing to delete a local path; only Hugging Face cache entries are removed");
    }
    let files = planned_files(spec)?;
    let mut names: Vec<String> = files.required.clone();
    names.extend(files.optional);
    names.sort();
    names.dedup();
    let report = delete_repo_files(cache, &files.repo, &names)?;
    let tok = match spec {
        ResolvedModel::Gguf { tokenizer_repo, .. } if tokenizer_repo != &files.repo => {
            delete_repo_files(
                cache,
                tokenizer_repo,
                &[GGUF_TOKENIZER.into(), GGUF_TOKENIZER_CONFIG.into()],
            )?
        }
        _ => DeleteReport { files: 0, bytes: 0 },
    };
    let total = DeleteReport {
        files: report.files + tok.files,
        bytes: report.bytes + tok.bytes,
    };
    if total.files == 0 {
        bail!("not in the Hugging Face cache");
    }
    Ok(total)
}

struct PlannedFiles {
    repo: String,
    required: Vec<String>,
    optional: Vec<String>,
}

fn planned_files(spec: &ResolvedModel) -> Result<PlannedFiles> {
    match spec {
        ResolvedModel::Laya { repo, subfolder } => {
            let rel = |file: &str| match subfolder {
                Some(sub) => format!("{sub}/{file}"),
                None => file.to_string(),
            };
            Ok(PlannedFiles {
                repo: repo.clone(),
                required: vec![
                    rel(CONFIG),
                    rel(ENCODER_CONFIG),
                    rel(TOKENIZER),
                    rel(WEIGHTS),
                ],
                optional: vec![rel(TOKENIZER_CONFIG)],
            })
        }
        ResolvedModel::Gguf { repo, filename, .. } => {
            if filename.is_empty() {
                bail!("GGUF delete/update needs a filename");
            }
            Ok(PlannedFiles {
                repo: repo.clone(),
                required: vec![filename.clone()],
                optional: Vec::new(),
            })
        }
        ResolvedModel::Rerank { repo } => Ok(PlannedFiles {
            repo: repo.clone(),
            required: vec![RERANK_CONFIG.into(), GGUF_TOKENIZER.into(), WEIGHTS.into()],
            optional: vec![GGUF_TOKENIZER_CONFIG.into()],
        }),
    }
}

fn spec_is_local(spec: &ResolvedModel) -> bool {
    match spec {
        ResolvedModel::Laya { repo, .. } => Path::new(repo).is_dir(),
        ResolvedModel::Gguf { repo, .. } => {
            let p = Path::new(repo);
            p.is_file() || p.is_dir()
        }
        ResolvedModel::Rerank { repo } => Path::new(repo).is_dir(),
    }
}

fn local_status(spec: &ResolvedModel) -> Result<CacheStatus> {
    let missing = match spec {
        ResolvedModel::Laya { repo, subfolder } => {
            from_dir(Path::new(repo), subfolder.as_deref()).err()
        }
        ResolvedModel::Gguf {
            repo,
            filename,
            tokenizer_repo,
        } => fetch_gguf(repo, filename, tokenizer_repo).err(),
        ResolvedModel::Rerank { repo } => rerank_from_dir(Path::new(repo)).err(),
    };
    if missing.is_some() {
        return Ok(CacheStatus {
            downloaded: false,
            bytes: 0,
            source: CacheSource::Local,
        });
    }
    let bytes = match spec {
        ResolvedModel::Laya { repo, subfolder } => {
            let files = planned_files(spec)?;
            let dir = match subfolder {
                Some(sub) => Path::new(repo).join(sub),
                None => PathBuf::from(repo),
            };
            unique_size(&dir, &files.required)
        }
        ResolvedModel::Gguf {
            repo,
            filename,
            tokenizer_repo,
        } => {
            let weights = locate_gguf(repo, filename, false)?;
            let (tokenizer, tokenizer_config) = locate_tokenizer(tokenizer_repo, false)?;
            let mut paths = vec![weights, tokenizer];
            if let Some(cfg) = tokenizer_config {
                paths.push(cfg);
            }
            unique_paths_size(&paths)
        }
        ResolvedModel::Rerank { repo } => {
            let files = planned_files(spec)?;
            unique_size(Path::new(repo), &files.required)
        }
    };
    Ok(CacheStatus {
        downloaded: true,
        bytes,
        source: CacheSource::Local,
    })
}

fn cache_root(cache: &Path, repo_id: &str) -> PathBuf {
    cache.join(format!("models--{}", repo_id.replace('/', "--")))
}

fn snapshot_with(root: &Path, required: &[String]) -> Option<PathBuf> {
    let snaps = root.join("snapshots");
    let revs = fs::read_dir(&snaps).ok()?;
    for rev in revs.flatten() {
        let path = rev.path();
        if path.is_dir() && required.iter().all(|f| path.join(f).is_file()) {
            return Some(path);
        }
    }
    None
}

fn unique_size(dir: &Path, rels: &[String]) -> u64 {
    unique_paths_size(&rels.iter().map(|f| dir.join(f)).collect::<Vec<_>>())
}

fn unique_paths_size(paths: &[PathBuf]) -> u64 {
    let mut seen = HashSet::new();
    let mut bytes = 0u64;
    for path in paths {
        let Ok(meta) = fs::metadata(path) else {
            continue;
        };
        let key = fs::canonicalize(path).unwrap_or_else(|_| path.clone());
        if seen.insert(key) {
            bytes += meta.len();
        }
    }
    bytes
}

fn delete_repo_files(cache: &Path, repo_id: &str, names: &[String]) -> Result<DeleteReport> {
    let root = cache_root(cache, repo_id);
    let snapshots = root.join("snapshots");
    if !snapshots.is_dir() {
        return Ok(DeleteReport { files: 0, bytes: 0 });
    }
    let mut removed = HashSet::new();
    let mut files = 0usize;
    let mut bytes = 0u64;
    for rev in fs::read_dir(&snapshots).context("reading Hugging Face snapshots")? {
        let snap = rev?.path();
        if !snap.is_dir() {
            continue;
        }
        for name in names {
            let pointer = snap.join(name);
            if fs::symlink_metadata(&pointer).is_err() {
                continue;
            }
            let size = fs::metadata(&pointer).map(|m| m.len()).unwrap_or(0);
            let blob = fs::canonicalize(&pointer).unwrap_or_else(|_| pointer.clone());
            fs::remove_file(&pointer).with_context(|| format!("removing {}", pointer.display()))?;
            files += 1;
            if removed.insert(blob) {
                bytes += size;
            }
            prune_empty_parents(&pointer, &snap);
        }
    }
    for blob in &removed {
        if blob.exists() && !blob_referenced(&snapshots, blob) {
            let _ = fs::remove_file(blob);
        }
    }
    prune_empty_dir(&snapshots);
    prune_empty_dir(&root.join("blobs"));
    if !has_any_file(&snapshots) {
        let _ = fs::remove_dir_all(&root);
    }
    Ok(DeleteReport { files, bytes })
}

fn blob_referenced(snapshots: &Path, blob: &Path) -> bool {
    let Ok(target) = fs::canonicalize(blob) else {
        return false;
    };
    fn walk(dir: &Path, target: &Path) -> bool {
        let Ok(entries) = fs::read_dir(dir) else {
            return false;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if walk(&path, target) {
                    return true;
                }
            } else if fs::canonicalize(&path).ok().as_deref() == Some(target) {
                return true;
            }
        }
        false
    }
    walk(snapshots, &target)
}

fn has_any_file(dir: &Path) -> bool {
    let Ok(entries) = fs::read_dir(dir) else {
        return false;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if has_any_file(&path) {
                return true;
            }
        } else {
            return true;
        }
    }
    false
}

fn prune_empty_parents(file: &Path, stop: &Path) {
    let mut parent = file.parent();
    while let Some(dir) = parent {
        if dir == stop {
            break;
        }
        if fs::remove_dir(dir).is_err() {
            break;
        }
        parent = dir.parent();
    }
}

fn prune_empty_dir(dir: &Path) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            prune_empty_dir(&path);
            let _ = fs::remove_dir(&path);
        }
    }
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

    fn tmp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("lmr-rs-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn plant(cache: &Path, repo: &str, commit: &str, rel: &str, bytes: &[u8]) {
        let root = cache_root(cache, repo);
        let blob = root.join("blobs").join(format!(
            "{:x}",
            bytes.len() as u64 + rel.len() as u64 + commit.len() as u64
        ));
        fs::create_dir_all(blob.parent().unwrap()).unwrap();
        fs::write(&blob, bytes).unwrap();
        let pointer = root.join("snapshots").join(commit).join(rel);
        fs::create_dir_all(pointer.parent().unwrap()).unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(&blob, &pointer).unwrap();
        #[cfg(not(unix))]
        fs::copy(&blob, &pointer).unwrap();
        let refs = root.join("refs");
        fs::create_dir_all(&refs).unwrap();
        fs::write(refs.join("main"), commit).unwrap();
    }

    fn plant_laya(cache: &Path, repo: &str, commit: &str, sub: Option<&str>) {
        let rel = |file: &str| match sub {
            Some(s) => format!("{s}/{file}"),
            None => file.to_string(),
        };
        for (file, body) in [
            (CONFIG, b"cfg" as &[u8]),
            (ENCODER_CONFIG, b"enc"),
            (TOKENIZER, b"tok"),
            (WEIGHTS, b"wwwwwwww"),
        ] {
            plant(cache, repo, commit, &rel(file), body);
        }
    }

    #[test]
    fn local_dir_needs_all_files() {
        let dir = tmp("local-laya");
        fs::create_dir_all(dir.join("encoder")).unwrap();
        fs::create_dir_all(dir.join("tokenizer")).unwrap();
        assert!(fetch(dir.to_str().unwrap(), None).is_err());
        for f in [CONFIG, ENCODER_CONFIG, TOKENIZER, WEIGHTS] {
            fs::write(dir.join(f), b"{}").unwrap();
        }
        let files = fetch(dir.to_str().unwrap(), None).unwrap();
        assert!(files.tokenizer_config.is_none());
        assert!(files.weights.ends_with(WEIGHTS));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn local_gguf_needs_weights_and_tokenizer() {
        let dir = tmp("local-gguf");
        assert!(fetch_gguf(dir.to_str().unwrap(), "m.gguf", dir.to_str().unwrap()).is_err());
        fs::write(dir.join("m.gguf"), b"gguf").unwrap();
        assert!(fetch_gguf(dir.to_str().unwrap(), "m.gguf", dir.to_str().unwrap()).is_err());
        fs::write(dir.join("tokenizer.json"), b"{}").unwrap();
        let files = fetch_gguf(dir.to_str().unwrap(), "m.gguf", dir.to_str().unwrap()).unwrap();
        assert!(files.weights.ends_with("m.gguf"));
        assert!(files.tokenizer.ends_with("tokenizer.json"));
        assert!(files.tokenizer_config.is_none());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn local_rerank_dir_needs_config_tokenizer_and_weights() {
        let dir = tmp("local-rerank");
        assert!(fetch_rerank(dir.to_str().unwrap()).is_err());
        fs::write(dir.join(RERANK_CONFIG), b"{}").unwrap();
        fs::write(dir.join(GGUF_TOKENIZER), b"{}").unwrap();
        assert!(fetch_rerank(dir.to_str().unwrap()).is_err());
        fs::write(dir.join(WEIGHTS), b"w").unwrap();
        let files = fetch_rerank(dir.to_str().unwrap()).unwrap();
        assert!(files.weights.ends_with(WEIGHTS));
        assert!(files.tokenizer_config.is_none());
        let spec = ResolvedModel::Rerank {
            repo: dir.to_str().unwrap().into(),
        };
        let st = cache_status_in(&dir, &spec).unwrap();
        assert!(st.downloaded);
        assert_eq!(st.source, CacheSource::Local);
        assert!(delete_model_in(&dir, &spec).is_err());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn cached_rerank_is_found_and_deleted() {
        let cache = tmp("hf-rerank");
        let repo = "BAAI/bge-reranker-v2-m3";
        for (file, body) in [
            (RERANK_CONFIG, b"cfg" as &[u8]),
            (GGUF_TOKENIZER, b"tok"),
            (GGUF_TOKENIZER_CONFIG, b"{}"),
            (WEIGHTS, b"wwwwwwww"),
        ] {
            plant(&cache, repo, "ddd", file, body);
        }
        let spec = ResolvedModel::Rerank { repo: repo.into() };
        let st = cache_status_in(&cache, &spec).unwrap();
        assert!(st.downloaded);
        assert_eq!(st.source, CacheSource::Hub);
        let report = delete_model_in(&cache, &spec).unwrap();
        assert_eq!(report.files, 4);
        assert!(!cache_status_in(&cache, &spec).unwrap().downloaded);
        fs::remove_dir_all(&cache).unwrap();
    }

    #[test]
    fn cache_status_and_delete_keep_sibling_subfolder() {
        let cache = tmp("hf-cache");
        plant_laya(&cache, DEFAULT_MODEL, "aaa", None);
        plant_laya(&cache, DEFAULT_MODEL, "aaa", Some("multilingual"));
        let english = ResolvedModel::Laya {
            repo: DEFAULT_MODEL.into(),
            subfolder: None,
        };
        let multi = ResolvedModel::Laya {
            repo: DEFAULT_MODEL.into(),
            subfolder: Some("multilingual".into()),
        };
        let st = cache_status_in(&cache, &english).unwrap();
        assert!(st.downloaded);
        assert_eq!(st.source, CacheSource::Hub);
        assert!(st.bytes > 0);
        assert!(cache_status_in(&cache, &multi).unwrap().downloaded);

        let report = delete_model_in(&cache, &english).unwrap();
        assert_eq!(report.files, 4);
        assert!(!cache_status_in(&cache, &english).unwrap().downloaded);
        assert!(cache_status_in(&cache, &multi).unwrap().downloaded);
        fs::remove_dir_all(&cache).unwrap();
    }

    #[test]
    fn delete_missing_is_an_error() {
        let cache = tmp("hf-empty");
        let spec = ResolvedModel::Gguf {
            repo: "Qwen/Qwen3-0.6B-GGUF".into(),
            filename: "Qwen3-0.6B-Q8_0.gguf".into(),
            tokenizer_repo: "Qwen/Qwen3-0.6B".into(),
        };
        let err = delete_model_in(&cache, &spec).unwrap_err().to_string();
        assert!(err.contains("not in the Hugging Face cache"));
        fs::remove_dir_all(&cache).unwrap();
    }

    #[test]
    fn delete_gguf_drops_weights_and_tokenizer() {
        let cache = tmp("hf-gguf");
        plant(
            &cache,
            "Qwen/Qwen3-0.6B-GGUF",
            "bbb",
            "Qwen3-0.6B-Q8_0.gguf",
            b"weights-here",
        );
        plant(
            &cache,
            "Qwen/Qwen3-0.6B",
            "ccc",
            GGUF_TOKENIZER,
            b"tokenizer",
        );
        plant(
            &cache,
            "Qwen/Qwen3-0.6B",
            "ccc",
            GGUF_TOKENIZER_CONFIG,
            b"{}",
        );
        let spec = ResolvedModel::Gguf {
            repo: "Qwen/Qwen3-0.6B-GGUF".into(),
            filename: "Qwen3-0.6B-Q8_0.gguf".into(),
            tokenizer_repo: "Qwen/Qwen3-0.6B".into(),
        };
        assert!(cache_status_in(&cache, &spec).unwrap().downloaded);
        let report = delete_model_in(&cache, &spec).unwrap();
        assert_eq!(report.files, 3);
        assert!(!cache_status_in(&cache, &spec).unwrap().downloaded);
        assert!(!cache_root(&cache, "Qwen/Qwen3-0.6B-GGUF").exists());
        fs::remove_dir_all(&cache).unwrap();
    }

    #[test]
    fn update_and_delete_refuse_local_paths() {
        let dir = tmp("local-refuse");
        for f in [CONFIG, ENCODER_CONFIG, TOKENIZER, WEIGHTS] {
            if let Some(parent) = dir.join(f).parent() {
                fs::create_dir_all(parent).unwrap();
            }
            fs::write(dir.join(f), b"{}").unwrap();
        }
        let spec = ResolvedModel::Laya {
            repo: dir.to_str().unwrap().into(),
            subfolder: None,
        };
        assert_eq!(
            cache_status_in(&dir, &spec).unwrap().source,
            CacheSource::Local
        );
        let del = delete_model_in(&dir, &spec).unwrap_err().to_string();
        assert!(del.contains("local path"));
        let upd = update_model(&spec).unwrap_err().to_string();
        assert!(upd.contains("local checkpoints"));
        fs::remove_dir_all(&dir).unwrap();
    }
}
