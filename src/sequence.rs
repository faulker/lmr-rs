//! Token sequence construction and calibration helpers, a line for line port of
//! `laya/common.py` (`render_options`, `build_sequence`, `temp_bucket`, `confidence_from_probs`).
//! Anything that changes the ids here changes the model's answer, so the Python behaviour is
//! reproduced exactly, including its `json.dumps` formatting of structured state and criteria.

use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use serde_json::Value;
use tokenizers::Tokenizer;

/// Question type, numbered as Laya's `QTYPES` so the value indexes `type_emb` and
/// `temperature`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QType {
    Choice = 0,
    Score = 1,
    Noul = 2,
}

impl QType {
    /// Parse the `type` field of a question.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "choice" => Some(Self::Choice),
            "score" => Some(Self::Score),
            "noul" => Some(Self::Noul),
            _ => None,
        }
    }

    /// The wire name, used in the sequence header and in answers.
    pub fn name(self) -> &'static str {
        match self {
            Self::Choice => "choice",
            Self::Score => "score",
            Self::Noul => "noul",
        }
    }
}

/// A question after `Agent._to_internal`: type, instructions as text, and the raw criteria.
#[derive(Debug, Clone)]
pub struct Question {
    pub qtype: QType,
    pub instructions: String,
    pub criteria: Option<Value>,
}

impl Question {
    /// Normalise a question definition. Mirrors `_to_internal`: a list of choice criteria becomes
    /// keys without descriptions, and non-string instructions are JSON encoded.
    pub fn from_def(qtype: QType, instructions: &Value, criteria: Option<&Value>) -> Result<Self> {
        let instructions = match instructions {
            Value::String(s) => s.clone(),
            other => python_json_dumps(other, true),
        };
        let criteria = match (qtype, criteria) {
            (QType::Choice, Some(Value::Array(items))) => {
                let mut map = serde_json::Map::new();
                for item in items {
                    let key = item
                        .as_str()
                        .ok_or_else(|| anyhow!("choice criteria list entries must be strings"))?;
                    map.insert(key.to_string(), Value::Null);
                }
                Some(Value::Object(map))
            }
            (_, Some(v)) => Some(v.clone()),
            (_, None) => None,
        };
        Ok(Self {
            qtype,
            instructions,
            criteria,
        })
    }

    /// Choice keys in definition order. Empty for other types.
    pub fn choice_keys(&self) -> Vec<String> {
        match (&self.qtype, &self.criteria) {
            (QType::Choice, Some(Value::Object(map))) => map.keys().cloned().collect(),
            _ => Vec::new(),
        }
    }
}

/// Serialise like Python's `json.dumps`: `", "` and `": "` separators, and `\uXXXX` escapes
/// for non-ASCII text when `ensure_ascii` is set (Python's default).
pub fn python_json_dumps(value: &Value, ensure_ascii: bool) -> String {
    let mut out = String::new();
    write_python_json(&mut out, value);
    if ensure_ascii {
        escape_non_ascii(&out)
    } else {
        out
    }
}

fn write_python_json(out: &mut String, value: &Value) {
    match value {
        Value::Object(map) => {
            out.push('{');
            for (i, (k, v)) in map.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(&serde_json::to_string(k).unwrap_or_default());
                out.push_str(": ");
                write_python_json(out, v);
            }
            out.push('}');
        }
        Value::Array(items) => {
            out.push('[');
            for (i, v) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_python_json(out, v);
            }
            out.push(']');
        }
        // Scalars serialise identically in serde_json and Python's json module.
        other => out.push_str(&serde_json::to_string(other).unwrap_or_default()),
    }
}

fn escape_non_ascii(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        if ch.is_ascii() {
            out.push(ch);
        } else {
            let mut units = [0u16; 2];
            for unit in ch.encode_utf16(&mut units) {
                let _ = write!(out, "\\u{unit:04x}");
            }
        }
    }
    out
}

/// Python's `serialize_state`: strings pass through, anything else is `json.dumps(..,
/// ensure_ascii=False)`.
pub fn serialize_state(state: &Value) -> String {
    match state {
        Value::String(s) => s.clone(),
        other => python_json_dumps(other, false),
    }
}

/// Python's `render_criterion`: strings pass through, structured values become compact JSON.
fn render_criterion(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => python_json_dumps(other, false),
    }
}

/// `None` or `""` mean "no description"; everything else, including `0` and `false`, counts.
fn is_blank(value: Option<&Value>) -> bool {
    matches!(value, None | Some(Value::Null))
        || matches!(value, Some(Value::String(s)) if s.is_empty())
}

/// Option texts in label order (`render_options`). Noul is always `[false, true]`.
pub fn render_options(q: &Question) -> Result<Vec<String>> {
    match q.qtype {
        QType::Choice => {
            let map = match &q.criteria {
                Some(Value::Object(map)) => map,
                _ => return Err(anyhow!("choice questions need a criteria object")),
            };
            Ok(map
                .iter()
                .map(|(k, v)| {
                    if is_blank(Some(v)) {
                        k.clone()
                    } else {
                        format!("{k}: {}", render_criterion(v))
                    }
                })
                .collect())
        }
        QType::Score => {
            let items = match &q.criteria {
                Some(Value::Array(items)) => items,
                _ => return Err(anyhow!("score questions need a criteria list")),
            };
            Ok(items
                .iter()
                .enumerate()
                .map(|(i, c)| format!("level {i}: {}", render_criterion(c)))
                .collect())
        }
        QType::Noul => {
            let (f, t) = match &q.criteria {
                Some(Value::Object(map)) => (map.get("false"), map.get("true")),
                None | Some(Value::Null) => (None, None),
                _ => return Err(anyhow!("noul criteria must be an object")),
            };
            let f_text = if is_blank(f) {
                "no, the statement does not hold".to_string()
            } else {
                render_criterion(f.unwrap())
            };
            let t_text = if is_blank(t) {
                "yes, the statement holds".to_string()
            } else {
                render_criterion(t.unwrap())
            };
            Ok(vec![format!("false: {f_text}"), format!("true: {t_text}")])
        }
    }
}

/// The tokenizer plus the special ids `build_sequence` needs.
pub struct Tok {
    inner: Tokenizer,
    pub cls_id: u32,
    pub sep_id: u32,
    pub mask_id: u32,
    pub pad_id: u32,
    pub mask_token: String,
}

/// Read a special token from `tokenizer_config.json`, where it is either a string or an
/// `{"content": ..}` object.
fn special_token(cfg: Option<&Value>, key: &str, default: &str) -> String {
    match cfg.and_then(|c| c.get(key)) {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Object(o)) => o
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or(default)
            .to_string(),
        _ => default.to_string(),
    }
}

impl Tok {
    /// Load `tokenizer.json` and resolve the special ids from `tokenizer_config.json`.
    pub fn load(tokenizer_json: &Path, tokenizer_config: Option<&Path>) -> Result<Self> {
        let inner = Tokenizer::from_file(tokenizer_json)
            .map_err(|e| anyhow!("loading {}: {e}", tokenizer_json.display()))?;
        let cfg: Option<Value> = match tokenizer_config {
            Some(p) => {
                let text =
                    fs::read_to_string(p).with_context(|| format!("reading {}", p.display()))?;
                Some(
                    serde_json::from_str(&text)
                        .with_context(|| format!("parsing {}", p.display()))?,
                )
            }
            None => None,
        };
        let lookup = |key: &str, default: &str| -> Result<(String, u32)> {
            let token = special_token(cfg.as_ref(), key, default);
            let id = inner
                .token_to_id(&token)
                .ok_or_else(|| anyhow!("tokenizer has no id for {key} token {token:?}"))?;
            Ok((token, id))
        };
        let (_, cls_id) = lookup("cls_token", "[CLS]")?;
        let (_, sep_id) = lookup("sep_token", "[SEP]")?;
        let (mask_token, mask_id) = lookup("mask_token", "[MASK]")?;
        let (_, pad_id) = lookup("pad_token", "[PAD]")?;
        Ok(Self {
            inner,
            cls_id,
            sep_id,
            mask_id,
            pad_id,
            mask_token,
        })
    }

    /// Encode without special tokens, like `tok(text, add_special_tokens=False)`.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow!("tokenizing: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Decode ids back to text, keeping special tokens. Used by tests to inspect packing.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, false)
            .map_err(|e| anyhow!("decoding: {e}"))
    }

    /// How many tokens `build_sequence` will spend on `state` before the length cap.
    pub fn state_token_count(&self, state: &Value) -> Result<usize> {
        Ok(self
            .encode(&serialize_state(state).replace(&self.mask_token, " "))?
            .len())
    }
}

/// Token ids for one question plus the position of each option's `[MASK]` marker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sequence {
    pub ids: Vec<u32>,
    pub markers: Vec<usize>,
}

/// Port of `build_sequence` (`truncate_left=False`, natural option order).
/// Layout: `[CLS] <type> question: <instructions> [SEP] ([MASK] option)* [SEP] state [SEP]`.
pub fn build_sequence(
    tok: &Tok,
    state: &Value,
    q: &Question,
    max_len: usize,
    head_max_len: usize,
) -> Result<Sequence> {
    let mask_tok = tok.mask_token.as_str();
    let opts = render_options(q)?;
    let ins = q.instructions.replace(mask_tok, " ");
    let mut head_ids = tok.encode(&format!("{} question: {ins}", q.qtype.name()))?;
    let mut opt_ids: Vec<Vec<u32>> = Vec::with_capacity(opts.len());
    for opt in &opts {
        let mut ids = tok.encode(&format!(" {}", opt.replace(mask_tok, " ")))?;
        ids.truncate(48);
        ids.insert(0, tok.mask_id);
        opt_ids.push(ids);
    }
    let total: usize = opt_ids.iter().map(Vec::len).sum();
    let mut opt_budget = head_max_len as i64 - total as i64;
    if opt_budget < 16 {
        let per = ((head_max_len as i64 - 16) / opt_ids.len().max(1) as i64).max(4) as usize;
        for o in &mut opt_ids {
            o.truncate(per);
        }
        let total: usize = opt_ids.iter().map(Vec::len).sum();
        opt_budget = head_max_len as i64 - total as i64;
    }
    head_ids.truncate(opt_budget.max(8) as usize);

    let mut ids = Vec::with_capacity(max_len);
    ids.push(tok.cls_id);
    ids.extend(head_ids);
    ids.push(tok.sep_id);
    let mut markers = Vec::with_capacity(opt_ids.len());
    for o in opt_ids {
        markers.push(ids.len());
        ids.extend(o);
    }
    ids.push(tok.sep_id);
    let room = (max_len as i64 - ids.len() as i64 - 1).max(0) as usize;
    let mut st = tok.encode(&serialize_state(state).replace(mask_tok, " "))?;
    st.truncate(room);
    ids.extend(st);
    ids.push(tok.sep_id);
    ids.truncate(max_len);
    markers.retain(|m| *m < max_len);
    Ok(Sequence { ids, markers })
}

/// Tokens to spend on instructions + options, given the sequence window and how long the
/// state is. Python packing uses a fixed `head_max_len` (192). That leaves most of a 512-token
/// window empty when the state is short, while 15+ options get squeezed to a handful of tokens
/// and their descriptions never reach the model. Unused room is given to the head; a long
/// state still gets the configured 192 so the Python parity cases match.
pub fn packed_head_max_len(max_len: usize, head_max_len: usize, state_tokens: usize) -> usize {
    let overhead = 4; // [CLS], [SEP] after instructions, [SEP] after options, final [SEP]
    let from_room = max_len.saturating_sub(state_tokens.saturating_add(overhead));
    from_room
        .max(head_max_len)
        .min(max_len.saturating_sub(overhead))
}

/// Token counts between consecutive option markers. The last option is omitted because its
/// span runs into the state block.
pub fn option_widths(markers: &[usize]) -> Vec<usize> {
    markers.windows(2).map(|w| w[1] - w[0]).collect()
}

/// Calibration bucket name, e.g. `choice:6-10` (`temp_bucket`).
pub fn temp_bucket(qtype: QType, k: usize) -> String {
    let size = if k <= 2 {
        "2"
    } else if k <= 5 {
        "3-5"
    } else if k <= 10 {
        "6-10"
    } else {
        "11+"
    };
    format!("{}:{size}", qtype.name())
}

/// Numerically stable softmax of `z / t`, with `t` floored at 1e-3 like the Python code.
pub fn calibrated_softmax(logits: &[f32], temperature: f32) -> Vec<f64> {
    let t = f64::from(temperature).max(1e-3);
    let z: Vec<f64> = logits.iter().map(|&l| f64::from(l) / t).collect();
    let max = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp: Vec<f64> = z.iter().map(|v| (v - max).exp()).collect();
    let sum: f64 = exp.iter().sum();
    exp.into_iter().map(|v| v / sum).collect()
}

/// Normalised Shannon entropy confidence, `1 - H(p) / ln(k)` (`confidence_from_probs`).
pub fn confidence_from_probs(p: &[f64], k: usize) -> f64 {
    if k < 2 {
        return 1.0;
    }
    let ent: f64 = p
        .iter()
        .take(k)
        .map(|&v| -v * v.clamp(1e-12, 1.0).ln())
        .sum();
    (1.0 - ent / (k as f64).ln()).clamp(0.0, 1.0)
}

/// Python's `round(x, 4)` for the values in an answer.
pub fn round4(x: f64) -> f64 {
    (x * 10_000.0).round() / 10_000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn python_json_matches_dumps_formatting() {
        let v = json!({"transactionTitle": "CAFÉ 1", "direction": "money out", "n": [1, 2.5, true, null]});
        assert_eq!(
            python_json_dumps(&v, false),
            r#"{"transactionTitle": "CAFÉ 1", "direction": "money out", "n": [1, 2.5, true, null]}"#
        );
        assert_eq!(python_json_dumps(&json!("CAFÉ"), true), r#""CAF\u00c9""#);
        assert_eq!(python_json_dumps(&json!("😀"), true), r#""\ud83d\ude00""#);
        assert_eq!(python_json_dumps(&json!({}), false), "{}");
    }

    #[test]
    fn choice_options_render_like_python() {
        let q = Question::from_def(
            QType::Choice,
            &json!("Which?"),
            Some(&json!({"Dining": "Restaurants", "Gas": null, "Fees": "", "Zero": 0, "R": {"a": 1}})),
        )
        .unwrap();
        assert_eq!(
            render_options(&q).unwrap(),
            vec![
                "Dining: Restaurants",
                "Gas",
                "Fees",
                "Zero: 0",
                r#"R: {"a": 1}"#
            ]
        );
        let q = Question::from_def(QType::Choice, &json!("x"), Some(&json!(["a", "b"]))).unwrap();
        assert_eq!(render_options(&q).unwrap(), vec!["a", "b"]);
        assert_eq!(q.choice_keys(), vec!["a", "b"]);
    }

    #[test]
    fn score_and_noul_options() {
        let q = Question::from_def(QType::Score, &json!("x"), Some(&json!(["bad", "ok"]))).unwrap();
        assert_eq!(
            render_options(&q).unwrap(),
            vec!["level 0: bad", "level 1: ok"]
        );
        let q = Question::from_def(QType::Noul, &json!("x"), None).unwrap();
        assert_eq!(
            render_options(&q).unwrap(),
            vec![
                "false: no, the statement does not hold",
                "true: yes, the statement holds"
            ]
        );
        let q =
            Question::from_def(QType::Noul, &json!("x"), Some(&json!({"true": "it is"}))).unwrap();
        assert_eq!(render_options(&q).unwrap()[1], "true: it is");
    }

    #[test]
    fn structured_instructions_are_ascii_json() {
        let q = Question::from_def(QType::Noul, &json!({"q": "é"}), None).unwrap();
        assert_eq!(q.instructions, r#"{"q": "\u00e9"}"#);
    }

    #[test]
    fn buckets_and_confidence() {
        assert_eq!(temp_bucket(QType::Choice, 2), "choice:2");
        assert_eq!(temp_bucket(QType::Choice, 5), "choice:3-5");
        assert_eq!(temp_bucket(QType::Score, 10), "score:6-10");
        assert_eq!(temp_bucket(QType::Noul, 11), "noul:11+");
        assert_eq!(confidence_from_probs(&[1.0, 0.0], 2), 1.0);
        assert!((confidence_from_probs(&[0.5, 0.5], 2)).abs() < 1e-9);
        assert_eq!(confidence_from_probs(&[1.0], 1), 1.0);
        let p = calibrated_softmax(&[2.0, 0.0], 1.0);
        assert!((p[0] + p[1] - 1.0).abs() < 1e-12);
        assert_eq!(round4(0.123456), 0.1235);
    }

    #[test]
    fn packed_head_uses_unused_room_only_when_state_is_short() {
        assert_eq!(packed_head_max_len(512, 192, 12), 496);
        assert_eq!(packed_head_max_len(512, 192, 400), 192);
        assert_eq!(packed_head_max_len(512, 192, 0), 508);
        assert!(packed_head_max_len(512, 192, 12) > 192);
    }
}

#[cfg(test)]
mod packing_with_tokenizer {
    use super::*;
    use serde_json::json;
    use std::path::Path;

    fn tok() -> Tok {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tokenizer");
        Tok::load(
            &dir.join("tokenizer.json"),
            Some(&dir.join("tokenizer_config.json")),
        )
        .unwrap()
    }

    /// The 22-category payload from myphin: descriptions are long enough that Python packing
    /// (head_max_len=192) squeezes each option to 8 tokens, so "grocery store" never reaches
    /// Food. Giving the unused 512-token room to the head keeps those words.
    fn costco_question() -> (Value, Question) {
        let state = json!({"direction": "money out", "transactionTitle": "COSTO"});
        let q = Question::from_def(
            QType::Choice,
            &json!("Which spending category does this bank transaction belong to? Pick other if none fits."),
            Some(&json!({
                "Alcohol": "Wine, wine bar, beer, liqure, bars",
                "Auto": "Auto parts, auto insurance, auto registration, parking, tolls, or maintinance",
                "Bills": "Any household or personal expense that such as storage unit rent, cell phone",
                "Donation": null,
                "Entertainment": "Music events, movie theater, amusement parks, zoos",
                "Fee": "Transaction fees, processing fees",
                "Food": "Restaurants, cafes, bars, food delivery apps, grocery store, or food mart",
                "Gas": "Gas/fuel for an auto, gas station, fuel pump",
                "Health": "Anything to do with health and medical",
                "House": null,
                "Investment": null,
                "Investment › Fee": null,
                "Investment › Interest": null,
                "Investment › Transaction": null,
                "Kids": "A transaction for one of my kids Asher faulk or Bristo Faulk",
                "Legal": null,
                "Service": null,
                "Shopping": "The purchase of goods that are not food or alcohol",
                "Subscriptions": "Subscription service such as streaming services, music service, online subscriptions",
                "Travel": null,
                "Utilities": "House hold bills, such as gas, water, and trash",
                "other": null
            })),
        )
        .unwrap();
        (state, q)
    }

    fn food_option_text(tok: &Tok, seq: &Sequence) -> String {
        let food = 6; // Alcohol, Auto, Bills, Donation, Entertainment, Fee, Food
        tok.decode(&seq.ids[seq.markers[food]..seq.markers[food + 1]])
            .unwrap()
    }

    #[test]
    fn many_short_state_options_keep_descriptions_when_packed() {
        let tok = tok();
        let (state, q) = costco_question();
        let python = build_sequence(&tok, &state, &q, 512, 192).unwrap();
        let python_widths = option_widths(&python.markers);
        let food_python = food_option_text(&tok, &python);
        assert_eq!(
            python_widths.iter().copied().max(),
            Some(8),
            "python packing should cap long options at 8 tokens, got {python_widths:?}"
        );
        assert!(
            !food_python.to_lowercase().contains("grocery"),
            "python packing should have chopped the Food description: {food_python}"
        );

        let head = packed_head_max_len(512, 192, tok.state_token_count(&state).unwrap());
        let packed = build_sequence(&tok, &state, &q, 512, head).unwrap();
        let packed_widths = option_widths(&packed.markers);
        assert!(
            packed_widths.iter().copied().max().unwrap() > 8,
            "packed option widths {packed_widths:?}"
        );
        let food = food_option_text(&tok, &packed);
        assert!(
            food.to_lowercase().contains("grocery"),
            "packed Food option should keep the grocery cue: {food}"
        );
        assert_eq!(packed.markers.len(), 22);
    }

    #[test]
    fn nine_category_myphin_payload_is_unchanged_by_packing() {
        let tok = tok();
        let state = json!({
            "transactionTitle": "AMEX EPAYMENT ACH PMT",
            "direction": "money out"
        });
        let q = Question::from_def(
            QType::Choice,
            &json!("Which spending category does this bank transaction belong to? Pick other if none fits."),
            Some(&json!({
                "Dining": "Restaurants, cafes, bars",
                "Groceries": "Supermarkets and food stores",
                "Gas": null,
                "Gas (2)": null,
                "Utilities › Electric": "Power company bills",
                "Café & Bakery": "Coffee shops, pâtisseries",
                "Credit Card Payment": "Payments to a card issuer",
                "Transfers": "Moves between own accounts",
                "other": null
            })),
        )
        .unwrap();
        let python = build_sequence(&tok, &state, &q, 512, 192).unwrap();
        let head = packed_head_max_len(512, 192, tok.state_token_count(&state).unwrap());
        let packed = build_sequence(&tok, &state, &q, 512, head).unwrap();
        assert_eq!(python, packed);
    }
}
