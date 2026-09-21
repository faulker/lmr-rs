//! `Agent`: the end to end System One call, shaping answers exactly like `laya.Agent.system_one`
//! (and typesafe.ai's `/v1/systemone`), so clients written for either work unchanged.

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use serde_json::{json, Map, Value};
use thiserror::Error;

use crate::config::{load_encoder_config, LayaConfig};
use crate::hub::CheckpointFiles;
use crate::model::DecisionModel;
use crate::sequence::{
    build_sequence, calibrated_softmax, confidence_from_probs, render_options, round4, QType,
    Question, Tok,
};

/// Why a request could not be answered. `Invalid` is the caller's fault (HTTP 422); `Model`
/// is ours (HTTP 500).
#[derive(Debug, Error)]
pub enum LayaError {
    #[error("{0}")]
    Invalid(String),
    #[error("model error: {0}")]
    Model(#[from] anyhow::Error),
}

/// Anything the server can ask questions of. `Agent` is the real one; tests use a stub.
pub trait Decider: Send + Sync {
    /// Evaluate `questions` against `state`, returning the System One response document.
    fn system_one(&self, state: &Value, questions: &Map<String, Value>) -> Result<Value, LayaError>;
    /// Static facts for `GET /health`.
    fn info(&self) -> Value;
}

pub struct Agent {
    model: DecisionModel,
    tok: Tok,
    cfg: LayaConfig,
    checkpoint: String,
}

impl Agent {
    /// Load a checkpoint onto `device` in F32.
    pub fn load(files: &CheckpointFiles, device: &Device) -> Result<Self> {
        let cfg = LayaConfig::load(&files.config)?;
        let tok = Tok::load(&files.tokenizer, files.tokenizer_config.as_deref())?;
        let enc_cfg = load_encoder_config(&files.encoder_config, tok.pad_id)?;
        // Safety: the file is memory mapped read only and not modified while loaded.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&files.weights], DType::F32, device)? };
        let model = DecisionModel::load(vb, &enc_cfg, &cfg)
            .with_context(|| format!("loading weights from {}", files.weights.display()))?;
        Ok(Self {
            model,
            tok,
            cfg,
            checkpoint: files.id.clone(),
        })
    }

    /// Where the model lives, for logs and `/health`.
    pub fn device_name(&self) -> &'static str {
        match self.model.device() {
            Device::Cpu => "cpu",
            Device::Cuda(_) => "cuda",
            Device::Metal(_) => "metal",
        }
    }

    fn answer_one(&self, state: &Value, qid: &str, def: &Value) -> Result<(Value, usize), LayaError> {
        let obj = def
            .as_object()
            .ok_or_else(|| LayaError::Invalid(format!("question {qid:?} must be an object")))?;
        let kind = obj
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| LayaError::Invalid(format!("question {qid:?} needs a string type")))?;
        let qtype = QType::parse(kind).ok_or_else(|| {
            LayaError::Invalid(format!("question {qid:?} has unknown type {kind:?}"))
        })?;
        let instructions = obj
            .get("instructions")
            .ok_or_else(|| LayaError::Invalid(format!("question {qid:?} needs instructions")))?;
        let q = Question::from_def(qtype, instructions, obj.get("criteria"))
            .map_err(|e| LayaError::Invalid(format!("question {qid:?}: {e}")))?;
        let options = render_options(&q).map_err(|e| LayaError::Invalid(format!("question {qid:?}: {e}")))?;
        if options.is_empty() {
            return Err(LayaError::Invalid(format!("question {qid:?} has no options")));
        }
        let seq = build_sequence(&self.tok, state, &q, self.cfg.max_len, self.cfg.head_max_len)?;
        if seq.markers.len() != options.len() {
            return Err(LayaError::Invalid(format!(
                "question {qid:?} options exceed head_max_len={}",
                self.cfg.head_max_len
            )));
        }
        let scores = self.model.forward(&seq.ids, &seq.markers, qtype)?;
        let k = seq.markers.len();
        let p = calibrated_softmax(&scores.logits[..k], self.cfg.temperature_for(qtype, k));
        let confidence = round4(confidence_from_probs(&p, k));
        let action = json!({ "act_probability": round4(f64::from(scores.act_probability)) });
        let argmax = p
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let answer = match qtype {
            QType::Choice => {
                let keys = q.choice_keys();
                let probabilities: Map<String, Value> = keys
                    .iter()
                    .zip(&p)
                    .map(|(key, v)| (key.clone(), json!(round4(*v))))
                    .collect();
                json!({
                    "type": "choice",
                    "choice": keys[argmax],
                    "probabilities": probabilities,
                    "confidence": confidence,
                    "action": action,
                })
            }
            QType::Score => {
                let expected: f64 = p.iter().enumerate().map(|(i, v)| i as f64 * v).sum();
                let criteria = q.criteria.as_ref().and_then(Value::as_array).cloned().unwrap_or_default();
                let legend: Map<String, Value> = criteria
                    .iter()
                    .enumerate()
                    .map(|(i, c)| (i.to_string(), c.clone()))
                    .collect();
                let probabilities: Map<String, Value> = p
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (i.to_string(), json!(round4(*v))))
                    .collect();
                json!({
                    "type": "score",
                    "score": round4(expected),
                    "legend": legend,
                    "probabilities": probabilities,
                    "confidence": confidence,
                    "action": action,
                })
            }
            QType::Noul => {
                let yes = p.get(1).copied().unwrap_or(0.0);
                json!({
                    "type": "noul",
                    "noul": round4(yes),
                    "confidence": round4(yes.max(1.0 - yes)),
                    "action": action,
                })
            }
        };
        Ok((answer, seq.ids.len()))
    }
}

impl Decider for Agent {
    /// One forward pass per question. Questions are answered in the order given.
    fn system_one(&self, state: &Value, questions: &Map<String, Value>) -> Result<Value, LayaError> {
        if questions.is_empty() {
            return Err(LayaError::Invalid("questions must not be empty".into()));
        }
        let mut answers = Map::new();
        let mut tokens = 0;
        for (qid, def) in questions {
            let (answer, n) = self.answer_one(state, qid, def)?;
            answers.insert(qid.clone(), answer);
            tokens += n;
        }
        Ok(json!({
            "model": "laya-rs",
            "answers": answers,
            "usage": { "input_tokens": tokens, "output_tokens": 0 },
        }))
    }

    fn info(&self) -> Value {
        json!({
            "model": "laya-rs",
            "checkpoint": self.checkpoint,
            "encoder": self.cfg.encoder,
            "device": self.device_name(),
        })
    }
}
