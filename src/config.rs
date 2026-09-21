//! Checkpoint configuration: Laya's `rl_agent_config.json` and the encoder's HF `config.json`.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use candle_transformers::models::modernbert;
use serde::Deserialize;

use crate::sequence::{temp_bucket, QType};

/// The head's own settings. Field names follow `rl_agent_config.json`; defaults mirror what
/// `laya.Agent` falls back to when a key is missing.
#[derive(Debug, Clone, Deserialize)]
pub struct LayaConfig {
    /// HF id of the encoder the head was trained on, e.g. `answerdotai/ModernBERT-large`.
    pub encoder: String,
    #[serde(default = "default_head_layers")]
    pub head_layers: usize,
    #[serde(default = "default_max_len")]
    pub max_len: usize,
    #[serde(default = "default_head_max_len")]
    pub head_max_len: usize,
    #[serde(default)]
    pub model_name: Option<String>,
    /// Per question type calibration temperature, indexed by `QType`.
    #[serde(default = "default_temperature")]
    pub temperature: Vec<f32>,
    /// Finer calibration keyed by `temp_bucket`, e.g. `choice:6-10`.
    #[serde(default)]
    pub temperature_by_options: HashMap<String, f32>,
}

fn default_head_layers() -> usize {
    2
}
fn default_max_len() -> usize {
    512
}
fn default_head_max_len() -> usize {
    192
}
fn default_temperature() -> Vec<f32> {
    vec![1.0, 1.0, 1.0]
}

impl LayaConfig {
    /// Read and parse `rl_agent_config.json`.
    pub fn load(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path)
            .with_context(|| format!("reading {}", path.display()))?;
        serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))
    }

    /// Calibration temperature for a question with `k` options, mirroring `Agent.system_one`:
    /// the option-count bucket wins, then the per type value, then 1.0.
    pub fn temperature_for(&self, qtype: QType, k: usize) -> f32 {
        if let Some(t) = self.temperature_by_options.get(&temp_bucket(qtype, k)) {
            return *t;
        }
        self.temperature.get(qtype as usize).copied().unwrap_or(1.0)
    }
}

/// Parse the encoder's `config.json` into candle's ModernBERT config. `pad_token_id` is filled
/// from the tokenizer when the file leaves it out, and the rope thetas are lifted out of the
/// `rope_parameters` block that transformers 5 writes instead of the flat fields candle reads.
pub fn load_encoder_config(path: &Path, pad_token_id: u32) -> Result<modernbert::Config> {
    let text = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let value: serde_json::Value =
        serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
    let value = normalize_encoder_config(value, pad_token_id);
    serde_json::from_value(value).with_context(|| format!("encoder config {}", path.display()))
}

/// ModernBERT's published defaults, used when neither config layout names a theta.
const DEFAULT_GLOBAL_ROPE_THETA: f64 = 160_000.0;
const DEFAULT_LOCAL_ROPE_THETA: f64 = 10_000.0;

fn normalize_encoder_config(mut value: serde_json::Value, pad_token_id: u32) -> serde_json::Value {
    let Some(obj) = value.as_object_mut() else {
        return value;
    };
    obj.entry("pad_token_id")
        .or_insert_with(|| serde_json::Value::from(pad_token_id));
    let nested = |kind: &str| -> Option<f64> {
        obj.get("rope_parameters")?
            .get(kind)?
            .get("rope_theta")?
            .as_f64()
    };
    let global = nested("full_attention").unwrap_or(DEFAULT_GLOBAL_ROPE_THETA);
    let local = nested("sliding_attention").unwrap_or(DEFAULT_LOCAL_ROPE_THETA);
    obj.entry("global_rope_theta")
        .or_insert_with(|| serde_json::Value::from(global));
    obj.entry("local_rope_theta")
        .or_insert_with(|| serde_json::Value::from(local));
    value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temperature_prefers_bucket_then_type() {
        let cfg: LayaConfig = serde_json::from_str(
            r#"{"encoder":"x","temperature":[1.5,1.2,1.9],
                "temperature_by_options":{"choice:6-10":1.0}}"#,
        )
        .unwrap();
        assert_eq!(cfg.temperature_for(QType::Choice, 7), 1.0);
        assert_eq!(cfg.temperature_for(QType::Choice, 3), 1.5);
        assert_eq!(cfg.temperature_for(QType::Noul, 2), 1.9);
        assert_eq!(cfg.head_layers, 2);
        assert_eq!(cfg.max_len, 512);
    }

    #[test]
    fn encoder_config_accepts_transformers_5_rope_layout() {
        let v: serde_json::Value = serde_json::from_str(
            r#"{"rope_parameters":{"full_attention":{"rope_theta":160000.0},
                "sliding_attention":{"rope_theta":10000.0}}}"#,
        )
        .unwrap();
        let out = normalize_encoder_config(v, 7);
        assert_eq!(out["global_rope_theta"], 160000.0);
        assert_eq!(out["local_rope_theta"], 10000.0);
        assert_eq!(out["pad_token_id"], 7);
        // Flat fields win when present and an explicit pad id is kept.
        let v: serde_json::Value =
            serde_json::from_str(r#"{"global_rope_theta":1.0,"local_rope_theta":2.0,"pad_token_id":3}"#).unwrap();
        let out = normalize_encoder_config(v, 7);
        assert_eq!(out["global_rope_theta"], 1.0);
        assert_eq!(out["pad_token_id"], 3);
    }
}
