//! `Agent`: the end to end System One call. Small questions match `laya.Agent.system_one`
//! (and typesafe.ai's `/v1/systemone`). Choice questions with more than ten options are
//! answered as a tournament so we never use the published `choice:11+` temperature of ~0.1.

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use serde_json::{json, Map, Value};
use thiserror::Error;

use crate::config::{load_encoder_config, LayaConfig};
use crate::hub::CheckpointFiles;
use crate::model::DecisionModel;
use crate::sequence::{
    build_sequence, calibrated_softmax, confidence_from_probs, packed_head_max_len, render_options,
    round4, QType, Question, Tok,
};
use crate::tournament::{groups_question, pack_indices, subset_question};

/// Why a request could not be answered. `Invalid` is the caller's fault (HTTP 422); `Model`
/// is ours (HTTP 500).
#[derive(Debug, Error)]
pub enum LmrError {
    #[error("{0}")]
    Invalid(String),
    #[error("model error: {0}")]
    Model(#[from] anyhow::Error),
}

/// One chat turn. Roles are the usual `system` / `user` / `assistant`.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Sampling knobs from an OpenAI / llama.cpp `chat.completions` body.
/// `None` keeps the model family default.
#[derive(Debug, Clone)]
pub struct ChatOpts {
    pub max_tokens: Option<usize>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub seed: Option<u64>,
    /// From llama.cpp `chat_template_kwargs.enable_thinking`.
    pub enable_thinking: bool,
}

impl Default for ChatOpts {
    fn default() -> Self {
        Self {
            max_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            seed: None,
            enable_thinking: false,
        }
    }
}

/// GGUF chat. Laya does not implement this.
pub trait ChatEngine: Send + Sync {
    /// Run one completion and return an OpenAI-shaped `chat.completion` document.
    fn chat(&mut self, messages: &[ChatMessage], opts: &ChatOpts) -> Result<Value, LmrError>;
}

/// One document's place in a rerank answer. `index` is its position in the request.
#[derive(Debug, Clone, PartialEq)]
pub struct Ranked {
    pub index: usize,
    /// Sigmoid of the cross-encoder logit, in `[0, 1]`.
    pub relevance_score: f64,
}

/// Cross-encoder reranking (`POST /v1/rerank`). Only reranker checkpoints implement this.
pub trait RerankEngine: Send + Sync {
    /// Score every document against `query`, best first. `tokens` is the summed input length.
    fn rerank(&mut self, query: &str, documents: &[String]) -> Result<Reranked, LmrError>;
}

/// Sorted results plus the token count spent producing them.
#[derive(Debug, Clone, PartialEq)]
pub struct Reranked {
    pub results: Vec<Ranked>,
    pub tokens: usize,
}

/// Anything the server can ask questions of. `Agent` is the real one; tests use a stub.
/// Every engine accepts and returns the same System One document, so a client can
/// switch checkpoints without changing request or response parsing.
pub trait Decider: Send + Sync {
    /// Evaluate `questions` against `state`, returning the System One response document.
    fn system_one(
        &mut self,
        state: &Value,
        questions: &Map<String, Value>,
    ) -> Result<Value, LmrError>;
    /// Static facts for `GET /health`.
    fn info(&self) -> Value;
    /// Id returned by `GET /v1/models`, matching llama.cpp's loaded-file id.
    fn openai_model_id(&self) -> String {
        self.info()
            .get("checkpoint")
            .and_then(Value::as_str)
            .unwrap_or("lmr-rs")
            .to_string()
    }
    /// GGUF models return `Some`; Laya and rerankers return `None`.
    fn as_chat(&mut self) -> Option<&mut dyn ChatEngine> {
        None
    }
    /// Rerankers return `Some`; Laya and GGUF return `None`.
    fn as_rerank(&mut self) -> Option<&mut dyn RerankEngine> {
        None
    }
}

/// How `Agent` packs sequences and answers large `choice` questions. Defaults match the
/// lmr-rs improvements (unused head room, tournament above 10 options, 11+ temperature
/// floored at 1.0). Set these from `[model]` in the TOML config.
#[derive(Debug, Clone, PartialEq)]
pub struct DecidePolicy {
    /// Spend unused `max_len` tokens on option texts when the state is short.
    pub pack_head: bool,
    /// Split `choice` questions larger than `tournament_after` into group then member.
    pub tournament: bool,
    /// Start the tournament above this many options (the `choice:11+` bucket begins at 11).
    pub tournament_after: usize,
    /// Floor for a flat `choice` with more than 10 options. `0` keeps the checkpoint value
    /// (~0.1), which makes a modest logit gap look like 100% confidence.
    pub choice_min_temperature: f32,
}

impl Default for DecidePolicy {
    fn default() -> Self {
        Self {
            pack_head: true,
            tournament: true,
            tournament_after: 10,
            choice_min_temperature: 1.0,
        }
    }
}

pub struct Agent {
    model: DecisionModel,
    tok: Tok,
    cfg: LayaConfig,
    checkpoint: String,
    policy: DecidePolicy,
}

impl Agent {
    /// Load a checkpoint onto `device` in F32 with the default decide policy.
    pub fn load(files: &CheckpointFiles, device: &Device) -> Result<Self> {
        Self::load_with_policy(files, device, DecidePolicy::default())
    }

    /// Load a checkpoint and apply `policy` from the TOML config.
    pub fn load_with_policy(
        files: &CheckpointFiles,
        device: &Device,
        policy: DecidePolicy,
    ) -> Result<Self> {
        let cfg = LayaConfig::load(&files.config)?;
        let tok = Tok::load(&files.tokenizer, files.tokenizer_config.as_deref())?;
        let enc_cfg = load_encoder_config(&files.encoder_config, tok.pad_id)?;
        // Safety: the file is memory mapped read only and not modified while loaded.
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[&files.weights], DType::F32, device)? };
        let model = DecisionModel::load(vb, &enc_cfg, &cfg)
            .with_context(|| format!("loading weights from {}", files.weights.display()))?;
        Ok(Self {
            model,
            tok,
            cfg,
            checkpoint: files.id.clone(),
            policy,
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

    fn answer_one(
        &self,
        state: &Value,
        qid: &str,
        def: &Value,
    ) -> Result<(Value, usize), LmrError> {
        let q = parse_question(qid, def)?;
        if q.qtype == QType::Choice
            && self.policy.tournament
            && q.choice_keys().len() > self.policy.tournament_after
        {
            return self.answer_choice_tree(state, &q);
        }
        let scored = self.score_question(state, &q)?;
        Ok((answer_json(&q, &scored.p, scored.act), scored.tokens))
    }

    /// One forward pass: logits at each option marker, calibrated to probabilities.
    fn score_question(&self, state: &Value, q: &Question) -> Result<Scored, LmrError> {
        let options = render_options(q)?;
        let state_tokens = self.tok.state_token_count(state)?;
        let head_max_len = if self.policy.pack_head {
            packed_head_max_len(self.cfg.max_len, self.cfg.head_max_len, state_tokens)
        } else {
            self.cfg.head_max_len
        };
        let seq = build_sequence(&self.tok, state, q, self.cfg.max_len, head_max_len)?;
        if seq.markers.len() != options.len() {
            return Err(LmrError::Invalid(format!(
                "options exceed head_max_len={}",
                self.cfg.head_max_len
            )));
        }
        let scores = self.model.forward(&seq.ids, &seq.markers, q.qtype)?;
        let k = seq.markers.len();
        let p = calibrated_softmax(&scores.logits[..k], self.temperature(q.qtype, k));
        Ok(Scored {
            p,
            tokens: seq.ids.len(),
            act: scores.act_probability,
        })
    }

    /// Calibration temperature, with the configured floor on `choice:11+`.
    fn temperature(&self, qtype: QType, k: usize) -> f32 {
        let t = self.cfg.temperature_for(qtype, k);
        if qtype == QType::Choice && k > 10 {
            t.max(self.policy.choice_min_temperature)
        } else {
            t
        }
    }

    /// Recursively pick a group, then pick inside it, until the question has
    /// `tournament_after` options or fewer. Probabilities are P(group) * P(member | group)
    /// for the winning path; losing groups share their group mass uniformly.
    fn answer_choice_tree(&self, state: &Value, q: &Question) -> Result<(Value, usize), LmrError> {
        let keys = q.choice_keys();
        let idxs: Vec<usize> = (0..keys.len()).collect();
        let pick = self.pick_choice(state, q, &keys, &idxs)?;
        Ok((
            shape_choice(&keys, &pick.p, pick.act, pick.winner),
            pick.tokens,
        ))
    }

    fn pick_choice(
        &self,
        state: &Value,
        q: &Question,
        keys: &[String],
        idxs: &[usize],
    ) -> Result<Pick, LmrError> {
        if idxs.len() <= self.policy.tournament_after {
            let sub = subset_question(q, keys, idxs);
            let scored = self.score_question(state, &sub)?;
            return Ok(Pick {
                winner: idxs[argmax(&scored.p)],
                p: scored.p,
                tokens: scored.tokens,
                act: scored.act,
            });
        }
        let groups = pack_indices(idxs, keys, self.policy.tournament_after);
        if groups.len() == 1 {
            let sub = subset_question(q, keys, idxs);
            let scored = self.score_question(state, &sub)?;
            return Ok(Pick {
                winner: idxs[argmax(&scored.p)],
                p: scored.p,
                tokens: scored.tokens,
                act: scored.act,
            });
        }
        let gq = groups_question(q, keys, &groups);
        let gscored = self.score_question(state, &gq)?;
        let gwin = argmax(&gscored.p);
        let inner = self.pick_choice(state, q, keys, &groups[gwin])?;
        let mut p = vec![0.0; idxs.len()];
        for (g_i, group) in groups.iter().enumerate() {
            if g_i == gwin {
                for (inner_i, &global) in group.iter().enumerate() {
                    let pos = idxs.iter().position(|&x| x == global).unwrap_or(0);
                    p[pos] = gscored.p[g_i] * inner.p.get(inner_i).copied().unwrap_or(0.0);
                }
            } else {
                let share = gscored.p[g_i] / group.len().max(1) as f64;
                for &global in group {
                    let pos = idxs.iter().position(|&x| x == global).unwrap_or(0);
                    p[pos] = share;
                }
            }
        }
        Ok(Pick {
            winner: inner.winner,
            p,
            tokens: gscored.tokens + inner.tokens,
            act: inner.act,
        })
    }
}

/// Calibrated probabilities and the `act_head` output for one forward pass.
struct Scored {
    p: Vec<f64>,
    tokens: usize,
    act: f32,
}

struct Pick {
    winner: usize,
    p: Vec<f64>,
    tokens: usize,
    act: f32,
}

pub(crate) fn argmax(p: &[f64]) -> usize {
    p.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Parse one question object from a System One request.
pub(crate) fn parse_question(qid: &str, def: &Value) -> Result<Question, LmrError> {
    let obj = def
        .as_object()
        .ok_or_else(|| LmrError::Invalid(format!("question {qid:?} must be an object")))?;
    let kind = obj
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| LmrError::Invalid(format!("question {qid:?} needs a string type")))?;
    let qtype = QType::parse(kind)
        .ok_or_else(|| LmrError::Invalid(format!("question {qid:?} has unknown type {kind:?}")))?;
    let instructions = obj
        .get("instructions")
        .ok_or_else(|| LmrError::Invalid(format!("question {qid:?} needs instructions")))?;
    let q = Question::from_def(qtype, instructions, obj.get("criteria"))
        .map_err(|e| LmrError::Invalid(format!("question {qid:?}: {e}")))?;
    let options =
        render_options(&q).map_err(|e| LmrError::Invalid(format!("question {qid:?}: {e}")))?;
    if options.is_empty() {
        return Err(LmrError::Invalid(format!(
            "question {qid:?} has no options"
        )));
    }
    Ok(q)
}

pub(crate) fn shape_choice(keys: &[String], p: &[f64], act: f32, winner: usize) -> Value {
    let probabilities: Map<String, Value> = keys
        .iter()
        .zip(p)
        .map(|(key, v)| (key.clone(), json!(round4(*v))))
        .collect();
    json!({
        "type": "choice",
        "choice": keys.get(winner).cloned().unwrap_or_default(),
        "probabilities": probabilities,
        "confidence": round4(confidence_from_probs(p, p.len())),
        "action": { "act_probability": round4(f64::from(act)) },
    })
}

pub(crate) fn shape_score(q: &Question, p: &[f64], act: f32) -> Value {
    let expected: f64 = p.iter().enumerate().map(|(i, v)| i as f64 * v).sum();
    let criteria = q
        .criteria
        .as_ref()
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
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
        "confidence": round4(confidence_from_probs(p, p.len())),
        "action": json!({ "act_probability": round4(f64::from(act)) }),
    })
}

pub(crate) fn shape_noul(p: &[f64], act: f32) -> Value {
    let yes = p.get(1).copied().unwrap_or(0.0);
    json!({
        "type": "noul",
        "noul": round4(yes),
        "confidence": round4(yes.max(1.0 - yes)),
        "action": json!({ "act_probability": round4(f64::from(act)) }),
    })
}

/// Shape one answer object. Every engine uses this so the wire document stays identical.
pub(crate) fn answer_json(q: &Question, p: &[f64], act: f32) -> Value {
    match q.qtype {
        QType::Choice => shape_choice(&q.choice_keys(), p, act, argmax(p)),
        QType::Score => shape_score(q, p, act),
        QType::Noul => shape_noul(p, act),
    }
}

/// Top-level System One response. Same keys for Laya and GGUF.
pub(crate) fn system_one_document(answers: Map<String, Value>, input_tokens: usize) -> Value {
    json!({
        "model": "lmr-rs",
        "answers": answers,
        "usage": { "input_tokens": input_tokens, "output_tokens": 0 },
    })
}

impl Decider for Agent {
    /// One forward pass per small question. Large `choice` questions run a tournament.
    fn system_one(
        &mut self,
        state: &Value,
        questions: &Map<String, Value>,
    ) -> Result<Value, LmrError> {
        if questions.is_empty() {
            return Err(LmrError::Invalid("questions must not be empty".into()));
        }
        let mut answers = Map::new();
        let mut tokens = 0;
        for (qid, def) in questions {
            let (answer, n) = self.answer_one(state, qid, def)?;
            answers.insert(qid.clone(), answer);
            tokens += n;
        }
        Ok(system_one_document(answers, tokens))
    }

    fn info(&self) -> Value {
        json!({
            "status": "ok",
            "model": "lmr-rs",
            "engine": "laya",
            "checkpoint": self.checkpoint,
            "encoder": self.cfg.encoder,
            "device": self.device_name(),
            "decide": {
                "pack_head": self.policy.pack_head,
                "tournament": self.policy.tournament,
                "tournament_after": self.policy.tournament_after,
                "choice_min_temperature": self.policy.choice_min_temperature,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn choice_q() -> Question {
        parse_question(
            "category",
            &json!({
                "type": "choice",
                "instructions": "Which category?",
                "criteria": {"Dining": "cafes", "Gas": null}
            }),
        )
        .unwrap()
    }

    #[test]
    fn parse_question_rejects_bad_defs() {
        assert!(parse_question("q", &json!("nope")).is_err());
        assert!(parse_question("q", &json!({ "type": "choice" })).is_err());
        assert!(parse_question("q", &json!({ "type": "mystery", "instructions": "?" })).is_err());
        assert!(parse_question(
            "q",
            &json!({ "type": "choice", "instructions": "?", "criteria": {} })
        )
        .is_err());
    }

    #[test]
    fn answer_json_has_the_same_keys_for_every_type() {
        let choice = answer_json(&choice_q(), &[0.75, 0.25], 0.8);
        assert_eq!(choice["type"], "choice");
        assert_eq!(choice["choice"], "Dining");
        assert_eq!(choice["probabilities"]["Dining"], 0.75);
        assert_eq!(choice["probabilities"]["Gas"], 0.25);
        assert!(choice["confidence"].as_f64().is_some());
        assert_eq!(choice["action"]["act_probability"], 0.8);

        let score_q = parse_question(
            "urgency",
            &json!({
                "type": "score",
                "instructions": "How urgent?",
                "criteria": ["low", "high"]
            }),
        )
        .unwrap();
        let score = answer_json(&score_q, &[0.25, 0.75], 0.6);
        assert_eq!(score["type"], "score");
        assert_eq!(score["score"], 0.75);
        assert_eq!(score["legend"]["0"], "low");
        assert_eq!(score["probabilities"]["1"], 0.75);
        assert_eq!(score["action"]["act_probability"], 0.6);

        let noul_q = parse_question(
            "spam",
            &json!({ "type": "noul", "instructions": "Is this spam?" }),
        )
        .unwrap();
        let noul = answer_json(&noul_q, &[0.2, 0.8], 0.7);
        assert_eq!(noul["type"], "noul");
        assert_eq!(noul["noul"], 0.8);
        assert_eq!(noul["confidence"], 0.8);
        assert_eq!(noul["action"]["act_probability"], 0.7);
    }

    #[test]
    fn system_one_document_keys_are_stable() {
        let mut answers = Map::new();
        answers.insert("category".into(), json!({ "type": "choice" }));
        let doc = system_one_document(answers, 12);
        assert_eq!(doc["model"], "lmr-rs");
        assert_eq!(doc["usage"]["input_tokens"], 12);
        assert_eq!(doc["usage"]["output_tokens"], 0);
        assert!(doc["answers"].get("category").is_some());
        assert!(doc.get("state").is_none());
    }
}
