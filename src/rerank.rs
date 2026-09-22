//! Cross-encoder rerankers such as [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3):
//! XLM-RoBERTa with a one-logit classification head. Each `(query, document)` pair is one
//! forward pass (`<s> query </s></s> document </s>`) and the sigmoid of the logit is the
//! relevance score, the same number FlagEmbedding's `compute_score(normalize=True)` returns.
//! Pairs run one at a time, like the rest of this crate; no padding, no batching.

use std::fs;

use anyhow::{anyhow, bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::xlm_roberta::{Config, XLMRobertaForSequenceClassification};
use serde_json::{json, Map, Value};
use tokenizers::{Tokenizer, TruncationParams, TruncationStrategy};

use crate::decide::{Decider, LmrError, Ranked, RerankEngine, Reranked};
use crate::hub::RerankFiles;

/// Tokenizer cap when `tokenizer_config.json` does not name `model_max_length`.
const DEFAULT_MAX_LENGTH: usize = 512;

/// A loaded reranker.
pub struct Reranker {
    model: XLMRobertaForSequenceClassification,
    tokenizer: Tokenizer,
    checkpoint: String,
    device: Device,
    /// Longest `(query, document)` pair in tokens; longer pairs are truncated.
    max_length: usize,
}

impl Reranker {
    /// Memory-map the safetensors weights in F32 and load the tokenizer beside them.
    pub fn load(files: &RerankFiles, device: &Device) -> Result<Self> {
        let cfg_text = fs::read_to_string(&files.config)
            .with_context(|| format!("reading {}", files.config.display()))?;
        let raw: Value = serde_json::from_str(&cfg_text)
            .with_context(|| format!("parsing {}", files.config.display()))?;
        check_architecture(&raw).with_context(|| files.config.display().to_string())?;
        let cfg: Config = serde_json::from_value(raw.clone())
            .with_context(|| format!("reranker config {}", files.config.display()))?;
        let num_labels = raw
            .get("id2label")
            .and_then(Value::as_object)
            .map_or(1, Map::len);
        if num_labels != 1 {
            bail!("reranker must have one label, this checkpoint has {num_labels}");
        }
        // Safety: the file is memory mapped read only and not modified while loaded.
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[&files.weights], DType::F32, device)? };
        let model = XLMRobertaForSequenceClassification::new(1, &cfg, vb)
            .with_context(|| format!("loading weights from {}", files.weights.display()))?;
        let mut tokenizer = Tokenizer::from_file(&files.tokenizer)
            .map_err(|e| anyhow!("loading tokenizer {}: {e}", files.tokenizer.display()))?;
        let configured = match &files.tokenizer_config {
            Some(p) => {
                let text =
                    fs::read_to_string(p).with_context(|| format!("reading {}", p.display()))?;
                let v: Value = serde_json::from_str(&text)
                    .with_context(|| format!("parsing {}", p.display()))?;
                v.get("model_max_length")
                    .and_then(Value::as_u64)
                    .map(|n| n as usize)
            }
            None => None,
        };
        let max_length = max_pair_length(configured, cfg.max_position_embeddings);
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length,
                strategy: TruncationStrategy::LongestFirst,
                ..Default::default()
            }))
            .map_err(|e| anyhow!("configuring truncation: {e}"))?;
        Ok(Self {
            model,
            tokenizer,
            checkpoint: files.id.clone(),
            device: device.clone(),
            max_length,
        })
    }

    /// Where the model lives, for logs and `/health`.
    pub fn device_name(&self) -> &'static str {
        match self.device {
            Device::Cpu => "cpu",
            Device::Cuda(_) => "cuda",
            Device::Metal(_) => "metal",
        }
    }

    /// One forward pass for a pair. Returns the raw logit and the token count.
    fn score_pair(&self, query: &str, document: &str) -> Result<(f32, usize)> {
        let enc = self
            .tokenizer
            .encode((query, document), true)
            .map_err(|e| anyhow!("tokenizing: {e}"))?;
        let ids = enc.get_ids();
        let n = ids.len();
        let input = Tensor::new(ids, &self.device)?.unsqueeze(0)?;
        let mask = Tensor::ones((1, n), DType::F32, &self.device)?;
        let types = Tensor::zeros((1, n), DType::U32, &self.device)?;
        let logits = self.model.forward(&input, &mask, &types)?;
        let logit = logits.flatten_all()?.to_vec1::<f32>()?[0];
        Ok((logit, n))
    }
}

/// Refuse checkpoints that are not a sequence classifier before touching the weights.
fn check_architecture(cfg: &Value) -> Result<()> {
    let archs = cfg
        .get("architectures")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let ok = archs
        .iter()
        .filter_map(Value::as_str)
        .any(|a| a.ends_with("ForSequenceClassification"));
    if !ok {
        bail!(
            "not a reranker: architectures = {}; expected XLMRobertaForSequenceClassification",
            serde_json::to_string(&archs).unwrap_or_default()
        );
    }
    let model_type = cfg.get("model_type").and_then(Value::as_str).unwrap_or("");
    if !matches!(model_type, "xlm-roberta" | "roberta") {
        bail!("unsupported reranker model_type {model_type:?}; expected xlm-roberta");
    }
    Ok(())
}

/// The tokenizer's own cap, clamped to what the position table can hold. RoBERTa position
/// ids start at `padding_idx + 1`, so two slots of `max_position_embeddings` are unusable.
fn max_pair_length(configured: Option<usize>, max_position_embeddings: usize) -> usize {
    let table = max_position_embeddings.saturating_sub(2).max(1);
    configured
        .filter(|&n| n > 0 && n < 1 << 20)
        .unwrap_or(DEFAULT_MAX_LENGTH)
        .min(table)
}

/// `1 / (1 + e^-x)` in f64 so tiny logits keep their order after rounding.
fn sigmoid(x: f32) -> f64 {
    1.0 / (1.0 + (-f64::from(x)).exp())
}

/// Order best first; ties keep request order so results are deterministic.
fn sort_ranked(results: &mut [Ranked]) {
    results.sort_by(|a, b| {
        b.relevance_score
            .partial_cmp(&a.relevance_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.index.cmp(&b.index))
    });
}

impl RerankEngine for Reranker {
    fn rerank(&mut self, query: &str, documents: &[String]) -> Result<Reranked, LmrError> {
        if query.trim().is_empty() {
            return Err(LmrError::Invalid("query must not be empty".into()));
        }
        if documents.is_empty() {
            return Err(LmrError::Invalid("documents must not be empty".into()));
        }
        let mut results = Vec::with_capacity(documents.len());
        let mut tokens = 0;
        for (index, doc) in documents.iter().enumerate() {
            let (logit, n) = self.score_pair(query, doc)?;
            tokens += n;
            results.push(Ranked {
                index,
                relevance_score: sigmoid(logit),
            });
        }
        sort_ranked(&mut results);
        Ok(Reranked { results, tokens })
    }
}

impl Decider for Reranker {
    /// A reranker has no decision head; point System One callers at `/v1/rerank`.
    fn system_one(
        &mut self,
        _state: &Value,
        _questions: &Map<String, Value>,
    ) -> Result<Value, LmrError> {
        Err(LmrError::Invalid(
            "this checkpoint is a reranker; use POST /v1/rerank".into(),
        ))
    }

    fn info(&self) -> Value {
        json!({
            "status": "ok",
            "model": "lmr-rs",
            "engine": "rerank",
            "checkpoint": self.checkpoint,
            "device": self.device_name(),
            "max_length": self.max_length,
        })
    }

    fn as_rerank(&mut self) -> Option<&mut dyn RerankEngine> {
        Some(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn architecture_check_wants_a_sequence_classifier() {
        let ok = json!({
            "architectures": ["XLMRobertaForSequenceClassification"],
            "model_type": "xlm-roberta"
        });
        assert!(check_architecture(&ok).is_ok());
        let masked = json!({
            "architectures": ["XLMRobertaForMaskedLM"],
            "model_type": "xlm-roberta"
        });
        assert!(check_architecture(&masked)
            .unwrap_err()
            .to_string()
            .contains("not a reranker"));
        let bert = json!({
            "architectures": ["BertForSequenceClassification"],
            "model_type": "bert"
        });
        assert!(check_architecture(&bert)
            .unwrap_err()
            .to_string()
            .contains("model_type"));
        assert!(check_architecture(&json!({})).is_err());
    }

    #[test]
    fn pair_length_is_capped_by_the_position_table() {
        assert_eq!(max_pair_length(Some(8192), 8194), 8192);
        assert_eq!(max_pair_length(Some(8192), 514), 512);
        assert_eq!(max_pair_length(None, 8194), DEFAULT_MAX_LENGTH);
        // transformers writes a huge sentinel when the length is unset.
        assert_eq!(
            max_pair_length(Some(1_000_000_000_000), 8194),
            DEFAULT_MAX_LENGTH
        );
    }

    #[test]
    fn sigmoid_and_sort_order() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-12);
        assert!(sigmoid(10.0) > 0.9999);
        assert!(sigmoid(-10.0) < 0.0001);
        let mut r = vec![
            Ranked {
                index: 0,
                relevance_score: 0.2,
            },
            Ranked {
                index: 1,
                relevance_score: 0.9,
            },
            Ranked {
                index: 2,
                relevance_score: 0.2,
            },
        ];
        sort_ranked(&mut r);
        assert_eq!(r.iter().map(|x| x.index).collect::<Vec<_>>(), vec![1, 0, 2]);
    }
}
