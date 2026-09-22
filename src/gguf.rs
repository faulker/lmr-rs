//! Causal GGUF models: Qwen3 and Llama-architecture MiniCPM5, loaded through candle.
//! Chat completions stay available; System One is the portable request and response.

use std::collections::HashMap;
use std::fs::File;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, bail, Context, Result};
use candle_core::quantized::gguf_file;
use candle_core::{Device, IndexOp, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::quantized_llama;
use candle_transformers::models::quantized_llama::ModelWeights as Llama;
use candle_transformers::models::quantized_qwen3::ModelWeights as Qwen3;
use serde_json::{json, Map, Value};
use tokenizers::Tokenizer;

use crate::decide::{
    answer_json, argmax, parse_question, shape_choice, system_one_document, ChatEngine,
    ChatMessage, ChatOpts, Decider, LmrError,
};
use crate::hub::GgufFiles;
use crate::sequence::{
    calibrated_softmax, python_json_dumps, render_options, serialize_state, QType, Question,
};
use crate::tournament::{groups_question, pack_indices, subset_question};

/// Stay inside candle's quantized Llama RoPE table (`quantized_llama::MAX_SEQ_LEN`).
const MAX_CONTEXT: usize = quantized_llama::MAX_SEQ_LEN;
/// Most options one pass scores. Each option costs a prefix forward, so larger choice
/// questions run the tournament in `pick_choice`.
const MAX_FLAT: usize = 26;
/// How much of the content-free prior `score_question` divides out. 1.0 is full contextual
/// calibration, which over-corrects on merchants the model does not know (any option can
/// win) and never lets a high-prior `other` through; 0.5 kept the known cases right on both
/// catalog models while leaving `other` reachable.
const PRIOR_WEIGHT: f64 = 0.5;
/// Distinct questions whose content-free scores stay cached before the cache resets.
const PRIOR_CACHE_MAX: usize = 64;
const DECISION_SYSTEM: &str = concat!(
    "You are a decision model. Read the state and pick exactly one listed answer. ",
    "Reply with that answer's name exactly as written and nothing else."
);

/// How the prompt is wrapped. Detected from GGUF `general.architecture`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatFamily {
    Qwen3,
    MiniCpm5,
}

/// Default sampling from each model's published non-thinking recommendations.
#[derive(Debug, Clone, Copy)]
struct SampleDefaults {
    temperature: f64,
    top_p: f64,
    top_k: usize,
}

impl ChatFamily {
    fn from_arch(arch: &str) -> Result<Self> {
        match arch {
            "qwen3" => Ok(Self::Qwen3),
            "llama" => Ok(Self::MiniCpm5),
            other => bail!("unsupported GGUF architecture {other:?}; expected qwen3 or llama"),
        }
    }

    fn sample_defaults(self) -> SampleDefaults {
        match self {
            Self::Qwen3 => SampleDefaults {
                temperature: 0.7,
                top_p: 0.8,
                top_k: 20,
            },
            Self::MiniCpm5 => SampleDefaults {
                temperature: 1.0,
                top_p: 0.95,
                top_k: 0,
            },
        }
    }

    fn bos(self) -> &'static str {
        match self {
            Self::Qwen3 => "",
            Self::MiniCpm5 => "<s>",
        }
    }
}

enum Inner {
    Qwen3(Qwen3),
    Llama(Llama),
}

/// A loaded GGUF chat model.
pub struct GgufEngine {
    inner: Inner,
    tokenizer: Tokenizer,
    eos: Vec<u32>,
    family: ChatFamily,
    checkpoint: String,
    device: Device,
    /// Content-free option scores by question, see `prior_scores`.
    prior_cache: HashMap<String, Vec<f64>>,
}

impl GgufEngine {
    /// Memory-map the GGUF weights and load the Hugging Face tokenizer beside them.
    pub fn load(files: &GgufFiles, device: &Device) -> Result<Self> {
        let mut reader = File::open(&files.weights)
            .with_context(|| format!("opening {}", files.weights.display()))?;
        let ct = gguf_file::Content::read(&mut reader)
            .map_err(|e| anyhow!("reading {}: {e}", files.weights.display()))?;
        let arch = ct
            .metadata
            .get("general.architecture")
            .and_then(|v| v.to_string().ok())
            .map(String::as_str)
            .unwrap_or("")
            .to_string();
        let family = ChatFamily::from_arch(&arch)?;
        let inner = match family {
            ChatFamily::Qwen3 => Inner::Qwen3(
                Qwen3::from_gguf(ct, &mut reader, device)
                    .with_context(|| format!("loading Qwen3 GGUF {}", files.weights.display()))?,
            ),
            ChatFamily::MiniCpm5 => Inner::Llama(
                Llama::from_gguf(ct, &mut reader, device)
                    .with_context(|| format!("loading Llama GGUF {}", files.weights.display()))?,
            ),
        };
        let tokenizer = Tokenizer::from_file(&files.tokenizer)
            .map_err(|e| anyhow!("loading tokenizer {}: {e}", files.tokenizer.display()))?;
        Ok(Self {
            inner,
            eos: eos_ids(&tokenizer),
            tokenizer,
            family,
            checkpoint: files.id.clone(),
            device: device.clone(),
            prior_cache: HashMap::new(),
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

    fn clear_kv_cache(&mut self) {
        match &mut self.inner {
            Inner::Qwen3(m) => m.clear_kv_cache(),
            Inner::Llama(m) => m.clear_kv_cache(),
        }
    }

    fn forward(&mut self, tokens: &[u32], offset: usize) -> Result<Tensor> {
        let input = Tensor::new(tokens, &self.device)?.unsqueeze(0)?;
        let logits = match &mut self.inner {
            Inner::Qwen3(m) => m.forward(&input, offset)?,
            Inner::Llama(m) => m.forward(&input, offset)?,
        };
        logits_for_sample(logits.to_dtype(candle_core::DType::F32)?)
    }

    /// Generate a completion for an already-templated prompt.
    fn generate(
        &mut self,
        prompt: &str,
        opts: &ChatOpts,
    ) -> Result<(String, usize, usize, &'static str), LmrError> {
        self.clear_kv_cache();
        let enc = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| LmrError::Invalid(format!("tokenizing prompt: {e}")))?;
        let prompt_ids = enc.get_ids().to_vec();
        if prompt_ids.is_empty() {
            return Err(LmrError::Invalid("prompt tokenized to nothing".into()));
        }
        if prompt_ids.len() >= MAX_CONTEXT {
            return Err(LmrError::Invalid(format!(
                "prompt is {} tokens; max context is {MAX_CONTEXT}",
                prompt_ids.len()
            )));
        }
        let defaults = self.family.sample_defaults();
        let temperature = opts.temperature.unwrap_or(defaults.temperature);
        let top_p = opts.top_p.unwrap_or(defaults.top_p);
        let top_k = opts.top_k.unwrap_or(defaults.top_k);
        let sampling = if temperature < 1e-7 {
            Sampling::ArgMax
        } else if top_k > 0 {
            Sampling::TopKThenTopP {
                k: top_k,
                p: top_p,
                temperature,
            }
        } else {
            Sampling::TopP {
                p: top_p,
                temperature,
            }
        };
        let seed = opts.seed.unwrap_or_else(|| {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(42)
        });
        let mut sampler = LogitsProcessor::from_sampling(seed, sampling);
        let want = opts.max_tokens.unwrap_or(256).max(1);
        let max_new = want.min(MAX_CONTEXT - prompt_ids.len());
        let mut generated: Vec<u32> = Vec::new();
        let mut next_input = prompt_ids.clone();
        let mut offset = 0usize;
        let mut finish = "stop";
        for i in 0..max_new {
            let logits = self
                .forward(&next_input, offset)
                .map_err(|e| LmrError::Model(e.into()))?;
            let token = sampler
                .sample(&logits)
                .map_err(|e| LmrError::Model(e.into()))?;
            if self.eos.contains(&token) {
                break;
            }
            generated.push(token);
            offset += next_input.len();
            next_input = vec![token];
            if i + 1 == max_new {
                finish = "length";
            }
            if offset + 1 >= MAX_CONTEXT {
                finish = "length";
                break;
            }
        }
        let text = self
            .tokenizer
            .decode(&generated, true)
            .map_err(|e| anyhow!("decoding: {e}"))?;
        Ok((text, prompt_ids.len(), generated.len(), finish))
    }

    fn logits_vec(&mut self, tokens: &[u32], offset: usize) -> Result<Vec<f32>, LmrError> {
        self.forward(tokens, offset)
            .and_then(|t| Ok(t.to_vec1::<f32>()?))
            .map_err(LmrError::Model)
    }

    /// Log-probability of each label as the reply after `prefix`. A label is scored token by
    /// token only while other labels share its path (`Investment` vs `Investment › Fee`),
    /// and gets `end` appended when it is a strict prefix of another label; past that point
    /// it is already identified, so the rest of its tokens are skipped. Candle's quantized
    /// forwards return logits for the last position only, so every branching point needs
    /// its own forward; a node that extends the path already in the KV cache only runs the
    /// new tokens. Returns the scores and tokens run.
    fn label_logprobs(
        &mut self,
        prefix: &[u32],
        labels: &[Vec<u32>],
        end: u32,
    ) -> Result<(Vec<f64>, usize), LmrError> {
        let mut scores = vec![0.0; labels.len()];
        let mut tokens = 0;
        // Path currently sitting in the KV cache after `prefix`, once anything has run.
        let mut cached: Option<Vec<u32>> = None;
        // Work list of (shared token path, labels that share it).
        let mut nodes: Vec<(Vec<u32>, Vec<usize>)> =
            vec![(Vec::new(), (0..labels.len()).collect())];
        while let Some((path, members)) = nodes.pop() {
            if prefix.len() + path.len() >= MAX_CONTEXT {
                return Err(LmrError::Invalid(format!(
                    "prompt plus option is {} tokens; max context is {MAX_CONTEXT}",
                    prefix.len() + path.len()
                )));
            }
            let logits = match &cached {
                Some(c) if path.len() > c.len() && path.starts_with(c) => {
                    tokens += path.len() - c.len();
                    self.logits_vec(&path[c.len()..], prefix.len() + c.len())?
                }
                _ => {
                    self.clear_kv_cache();
                    let mut seq = prefix.to_vec();
                    seq.extend_from_slice(&path);
                    tokens += seq.len();
                    self.logits_vec(&seq, 0)?
                }
            };
            cached = Some(path.clone());
            let mut children: Vec<(u32, Vec<usize>)> = Vec::new();
            for &i in &members {
                let next = labels[i].get(path.len()).copied().unwrap_or(end);
                scores[i] += log_softmax_at(&logits, next as usize);
                if next == end {
                    continue;
                }
                match children.iter_mut().find(|(t, _)| *t == next) {
                    Some((_, group)) => group.push(i),
                    None => children.push((next, vec![i])),
                }
            }
            for (tok, group) in children {
                if group.len() < 2 {
                    continue;
                }
                let mut child = path.clone();
                child.push(tok);
                nodes.push((child, group));
            }
        }
        Ok((scores, tokens))
    }

    fn encode_ids(&self, text: &str, what: &str) -> Result<Vec<u32>, LmrError> {
        let enc = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| LmrError::Invalid(format!("tokenizing {what}: {e}")))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Log-likelihood of each option name as the reply to `state`, plus tokens run.
    fn label_scores(
        &mut self,
        state: &Value,
        q: &Question,
        options: &[String],
    ) -> Result<(Vec<f64>, usize), LmrError> {
        let user = decision_user_text(state, q, options);
        let messages = [
            ChatMessage {
                role: "system".into(),
                content: DECISION_SYSTEM.into(),
            },
            ChatMessage {
                role: "user".into(),
                content: user,
            },
        ];
        let prompt = apply_chat_template(self.family, &messages, true, false)
            .map_err(|e| LmrError::Invalid(e.to_string()))?;
        let prefix = self.encode_ids(&prompt, "prompt")?;
        if prefix.is_empty() {
            return Err(LmrError::Invalid("prompt tokenized to nothing".into()));
        }
        let end = self
            .tokenizer
            .token_to_id("<|im_end|>")
            .ok_or_else(|| LmrError::Invalid("tokenizer has no <|im_end|> token".into()))?;
        let mut labels = Vec::with_capacity(options.len());
        for label in reply_labels(options) {
            let ids = self.encode_ids(label, "option")?;
            if ids.is_empty() {
                return Err(LmrError::Invalid(format!(
                    "option {label:?} tokenized to nothing"
                )));
            }
            labels.push(ids);
        }
        self.label_logprobs(&prefix, &labels, end)
    }

    /// The option scores for a content-free state (every value replaced by `N/A`): the
    /// model's prior over this option list, which `score_question` divides out. Cached per
    /// question, since one client usually asks the same question about many states.
    fn prior_scores(
        &mut self,
        state: &Value,
        q: &Question,
        options: &[String],
    ) -> Result<(Vec<f64>, usize), LmrError> {
        let blank = content_free_state(state);
        let key = format!(
            "{}\u{0}{}",
            python_json_dumps(&blank, false),
            decision_user_text(&blank, q, options)
        );
        if let Some(z) = self.prior_cache.get(&key) {
            return Ok((z.clone(), 0));
        }
        let (z, tokens) = self.label_scores(&blank, q, options)?;
        if self.prior_cache.len() >= PRIOR_CACHE_MAX {
            self.prior_cache.clear();
        }
        self.prior_cache.insert(key, z.clone());
        Ok((z, tokens))
    }

    /// Softmax over each option's reply likelihood, scored by name so the model's own
    /// knowledge of the words does the work (see `label_logprobs`), with the model's
    /// state-independent preference for some names partly divided out (contextual
    /// calibration, Zhao et al. 2021: `p(name | state) / p(name | N/A)^PRIOR_WEIGHT`).
    fn score_question(&mut self, state: &Value, q: &Question) -> Result<Scored, LmrError> {
        let options = render_options(q)?;
        if options.len() > MAX_FLAT {
            return Err(LmrError::Invalid(format!(
                "a GGUF pass can score at most {MAX_FLAT} options"
            )));
        }
        let (z, tokens) = self.label_scores(state, q, &options)?;
        let (prior, prior_tokens) = self.prior_scores(state, q, &options)?;
        let z: Vec<f32> = z
            .iter()
            .zip(&prior)
            .map(|(a, b)| (a - PRIOR_WEIGHT * b) as f32)
            .collect();
        let p = calibrated_softmax(&z, 1.0);
        let act = p.iter().copied().fold(0.0_f64, f64::max) as f32;
        Ok(Scored {
            p,
            tokens: tokens + prior_tokens,
            act,
        })
    }

    fn answer_one(
        &mut self,
        state: &Value,
        qid: &str,
        def: &Value,
    ) -> Result<(Value, usize), LmrError> {
        let q = parse_question(qid, def)?;
        if q.qtype == QType::Choice && q.choice_keys().len() > MAX_FLAT {
            return self.answer_choice_tree(state, &q);
        }
        let scored = self.score_question(state, &q)?;
        Ok((answer_json(&q, &scored.p, scored.act), scored.tokens))
    }

    fn answer_choice_tree(
        &mut self,
        state: &Value,
        q: &Question,
    ) -> Result<(Value, usize), LmrError> {
        let keys = q.choice_keys();
        let idxs: Vec<usize> = (0..keys.len()).collect();
        let pick = self.pick_choice(state, q, &keys, &idxs)?;
        Ok((
            shape_choice(&keys, &pick.p, pick.act, pick.winner),
            pick.tokens,
        ))
    }

    fn pick_choice(
        &mut self,
        state: &Value,
        q: &Question,
        keys: &[String],
        idxs: &[usize],
    ) -> Result<Pick, LmrError> {
        if idxs.len() <= MAX_FLAT {
            let sub = subset_question(q, keys, idxs);
            let scored = self.score_question(state, &sub)?;
            return Ok(Pick {
                winner: idxs[argmax(&scored.p)],
                p: scored.p,
                tokens: scored.tokens,
                act: scored.act,
            });
        }
        let mut groups = pack_indices(idxs, keys, MAX_FLAT);
        if groups.len() == 1 && idxs.len() > MAX_FLAT {
            groups = idxs.chunks(MAX_FLAT).map(|c| c.to_vec()).collect();
        }
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

/// `state` with every leaf value replaced by `N/A`, keeping the keys so the prompt has
/// the same shape. A string state becomes `N/A`.
fn content_free_state(state: &Value) -> Value {
    match state {
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(k, v)| (k.clone(), content_free_state(v)))
                .collect(),
        ),
        Value::Array(items) => Value::Array(items.iter().map(content_free_state).collect()),
        _ => Value::String("N/A".into()),
    }
}

/// The reply text scored for each option: the part before `": "` in the rendered option
/// (the choice key, `level N`, or `true` / `false`), or the whole option when it has none.
fn reply_labels(options: &[String]) -> Vec<&str> {
    options
        .iter()
        .map(|o| o.split_once(": ").map_or(o.as_str(), |(label, _)| label))
        .collect()
}

/// `log softmax(logits)[index]`, computed in f64 so long prompts don't lose precision.
fn log_softmax_at(logits: &[f32], index: usize) -> f64 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let lse: f64 = logits.iter().map(|&l| (f64::from(l) - max).exp()).sum();
    let target = logits.get(index).copied().unwrap_or(f32::NEG_INFINITY) as f64;
    target - max - lse.ln()
}

/// User turn that lists the options by name. Each name is then scored as the reply.
fn decision_user_text(state: &Value, q: &Question, options: &[String]) -> String {
    let mut s = String::new();
    s.push_str("State:\n");
    s.push_str(&serialize_state(state));
    s.push_str("\n\nType: ");
    s.push_str(q.qtype.name());
    s.push_str("\nInstructions: ");
    s.push_str(&q.instructions);
    s.push_str("\n\nAnswers:\n");
    for opt in options {
        s.push_str("- ");
        s.push_str(opt);
        s.push('\n');
    }
    s.push_str("\nReply with the answer's name only.");
    s
}

/// `LogitsProcessor::sample` wants `[vocab]`. Candle's quantized Qwen3 and Llama forwards
/// keep a batch dim (`[1, vocab]`, sometimes `[1, seq, vocab]`).
fn logits_for_sample(logits: Tensor) -> Result<Tensor> {
    let t = match logits.rank() {
        1 => logits,
        2 => logits.i(0)?,
        3 => {
            let last = logits.dim(1)? - 1;
            logits.i((0, last))?
        }
        n => bail!("logits rank {n}; expected 1, 2, or 3"),
    };
    if t.rank() != 1 {
        bail!(
            "logits for sample have shape {:?}; expected [vocab]",
            t.dims()
        );
    }
    Ok(t)
}

/// Wrap `messages` in the model's ChatML template.
pub fn apply_chat_template(
    family: ChatFamily,
    messages: &[ChatMessage],
    add_generation_prompt: bool,
    enable_thinking: bool,
) -> Result<String> {
    if messages.is_empty() {
        bail!("messages must not be empty");
    }
    let mut out = String::from(family.bos());
    for m in messages {
        let role = m.role.trim();
        if !matches!(role, "system" | "user" | "assistant") {
            bail!("unsupported chat role {role:?}; use system, user, or assistant");
        }
        out.push_str("<|im_start|>");
        out.push_str(role);
        out.push('\n');
        out.push_str(&m.content);
        out.push_str("<|im_end|>\n");
    }
    if add_generation_prompt {
        out.push_str("<|im_start|>assistant\n");
        if !enable_thinking {
            out.push_str("<think>\n\n</think>\n\n");
        }
    }
    Ok(out)
}

/// EOS / end-of-turn ids present in this tokenizer.
fn eos_ids(tokenizer: &Tokenizer) -> Vec<u32> {
    ["<|im_end|>", "<|endoftext|>", "</s>", "<|eot_id|>"]
        .into_iter()
        .filter_map(|t| tokenizer.token_to_id(t))
        .collect()
}

impl ChatEngine for GgufEngine {
    fn chat(&mut self, messages: &[ChatMessage], opts: &ChatOpts) -> Result<Value, LmrError> {
        let prompt = apply_chat_template(self.family, messages, true, opts.enable_thinking)
            .map_err(|e| LmrError::Invalid(e.to_string()))?;
        let (text, prompt_tokens, completion_tokens, finish_reason) =
            self.generate(&prompt, opts)?;
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        Ok(json!({
            "id": format!("chatcmpl-{created}"),
            "object": "chat.completion",
            "created": created,
            "model": self.checkpoint,
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": text },
                "finish_reason": finish_reason,
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }))
    }
}

impl Decider for GgufEngine {
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
            "engine": "gguf",
            "checkpoint": self.checkpoint,
            "family": match self.family {
                ChatFamily::Qwen3 => "qwen3",
                ChatFamily::MiniCpm5 => "minicpm5",
            },
            "device": self.device_name(),
        })
    }

    fn as_chat(&mut self) -> Option<&mut dyn ChatEngine> {
        Some(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.into(),
            content: content.into(),
        }
    }

    #[test]
    fn qwen3_template_is_chatml_without_bos() {
        let t = apply_chat_template(ChatFamily::Qwen3, &[msg("user", "hi")], true, false).unwrap();
        assert!(t.starts_with("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"));
        assert!(t.contains("<think>\n\n</think>\n\n"));
        assert!(!t.starts_with("<s>"));
    }

    #[test]
    fn minicpm_template_prefixes_bos_and_can_think() {
        let t = apply_chat_template(
            ChatFamily::MiniCpm5,
            &[msg("system", "be brief"), msg("user", "1+1")],
            true,
            true,
        )
        .unwrap();
        assert!(t.starts_with("<s><|im_start|>system\nbe brief<|im_end|>\n"));
        assert!(t.ends_with("<|im_start|>assistant\n"));
        assert!(!t.contains("<think>"));
    }

    #[test]
    fn template_rejects_empty_and_bad_roles() {
        assert!(apply_chat_template(ChatFamily::Qwen3, &[], true, false).is_err());
        assert!(apply_chat_template(ChatFamily::Qwen3, &[msg("tool", "x")], false, false).is_err());
    }

    #[test]
    fn family_from_known_architectures() {
        assert_eq!(ChatFamily::from_arch("qwen3").unwrap(), ChatFamily::Qwen3);
        assert_eq!(
            ChatFamily::from_arch("llama").unwrap(),
            ChatFamily::MiniCpm5
        );
        assert!(ChatFamily::from_arch("phi3").is_err());
    }

    #[test]
    fn logits_for_sample_drops_batch_dim() {
        let t = Tensor::new(&[[1f32, 2., 3.]], &Device::Cpu).unwrap();
        let flat = logits_for_sample(t).unwrap();
        assert_eq!(flat.dims(), [3]);
        assert_eq!(flat.to_vec1::<f32>().unwrap(), [1., 2., 3.]);
    }

    #[test]
    fn logits_for_sample_takes_last_seq_step() {
        let t = Tensor::new(&[[[1f32, 2.], [3., 4.]]], &Device::Cpu).unwrap();
        let flat = logits_for_sample(t).unwrap();
        assert_eq!(flat.to_vec1::<f32>().unwrap(), [3., 4.]);
    }

    #[test]
    fn logits_for_sample_keeps_rank1() {
        let t = Tensor::new(&[1f32, 2., 3.], &Device::Cpu).unwrap();
        assert_eq!(
            logits_for_sample(t).unwrap().to_vec1::<f32>().unwrap(),
            [1., 2., 3.]
        );
    }

    #[test]
    fn content_free_state_keeps_shape_and_blanks_values() {
        let state =
            json!({"direction": "money out", "amount": 12.5, "tags": ["a", true], "n": null});
        assert_eq!(
            content_free_state(&state),
            json!({"direction": "N/A", "amount": "N/A", "tags": ["N/A", "N/A"], "n": "N/A"})
        );
        assert_eq!(content_free_state(&json!("STARBUCKS")), json!("N/A"));
    }

    #[test]
    fn reply_labels_take_the_text_before_the_colon() {
        let options = vec![
            "Food: Restaurants, cafes".to_string(),
            "other".to_string(),
            "level 2: mostly fine".to_string(),
            "true: yes, the statement holds".to_string(),
        ];
        assert_eq!(reply_labels(&options), ["Food", "other", "level 2", "true"]);
    }

    #[test]
    fn log_softmax_at_matches_direct_computation() {
        let logits = [1.0f32, 2.0, 3.0];
        let sum: f64 = logits.iter().map(|&l| f64::from(l).exp()).sum();
        let want = 3.0 - sum.ln();
        assert!((log_softmax_at(&logits, 2) - want).abs() < 1e-9);
        let p: f64 = (0..3).map(|i| log_softmax_at(&logits, i).exp()).sum();
        assert!((p - 1.0).abs() < 1e-9);
        assert_eq!(log_softmax_at(&logits, 7), f64::NEG_INFINITY);
    }

    #[test]
    fn decision_prompt_lists_options_by_name() {
        let q = crate::decide::parse_question(
            "category",
            &json!({
                "type": "choice",
                "instructions": "Which category?",
                "criteria": {"Dining": "cafes", "Gas": null}
            }),
        )
        .unwrap();
        let options = render_options(&q).unwrap();
        let text = decision_user_text(&json!({"title": "CAFE"}), &q, &options);
        assert!(text.contains("Type: choice"));
        assert!(text.contains("Instructions: Which category?"));
        assert!(text.contains("- Dining: cafes\n"));
        assert!(text.contains("- Gas\n"));
        assert!(!text.contains("A. "));
        assert!(text.contains("Reply with the answer's name only."));
        assert!(text.contains("CAFE"));
    }
}
