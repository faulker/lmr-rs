//! Parity with the Python `laya` SDK, using fixtures from `scripts/make_fixtures.py`.
//!
//! `sequences_match_python` needs only the tokenizer files in `tests/fixtures/tokenizer`.
//! `answers_match_python` needs the checkpoint in the Hugging Face cache and is ignored by
//! default: `cargo test -- --ignored` after `laya-rs download`.

use std::path::Path;

use candle_core::Device;
use laya_rs::sequence::{build_sequence, QType, Question, Tok};
use laya_rs::{hub, Agent, Decider};
use serde_json::Value;

fn fixtures() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/cases.json");
    let text = std::fs::read_to_string(&path).expect("tests/fixtures/cases.json (run scripts/make_fixtures.py)");
    serde_json::from_str(&text).unwrap()
}

fn tokenizer() -> Tok {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tokenizer");
    Tok::load(&dir.join("tokenizer.json"), Some(&dir.join("tokenizer_config.json"))).unwrap()
}

#[test]
fn sequences_match_python() {
    let fx = fixtures();
    let tok = tokenizer();
    let max_len = fx["max_len"].as_u64().unwrap() as usize;
    let head_max_len = fx["head_max_len"].as_u64().unwrap() as usize;
    let mut checked = 0;
    for case in fx["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        for (qid, def) in case["questions"].as_object().unwrap() {
            let qtype = QType::parse(def["type"].as_str().unwrap()).unwrap();
            let q = Question::from_def(qtype, &def["instructions"], def.get("criteria")).unwrap();
            let seq = build_sequence(&tok, &case["state"], &q, max_len, head_max_len).unwrap();
            let want = &case["sequences"][qid];
            let want_ids: Vec<u32> = want["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
            let want_markers: Vec<usize> =
                want["markers"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
            assert_eq!(seq.ids, want_ids, "{name}/{qid} ids");
            assert_eq!(seq.markers, want_markers, "{name}/{qid} markers");
            checked += 1;
        }
    }
    assert!(checked >= 4);
}

#[test]
#[ignore = "needs the checkpoint: run `laya-rs download` first"]
fn answers_match_python() {
    let fx = fixtures();
    let files = hub::fetch(fx["model"].as_str().unwrap(), None).unwrap();
    let agent = Agent::load(&files, &Device::Cpu).unwrap();
    for case in fx["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let got = agent
            .system_one(&case["state"], case["questions"].as_object().unwrap())
            .unwrap();
        let want = &case["answer"];
        for (qid, w) in want["answers"].as_object().unwrap() {
            let g = &got["answers"][qid];
            assert_eq!(g["type"], w["type"], "{name}/{qid} type");
            for key in ["choice", "legend"] {
                if !w[key].is_null() {
                    assert_eq!(g[key], w[key], "{name}/{qid} {key}");
                }
            }
            for key in ["confidence", "score", "noul"] {
                if let (Some(gv), Some(wv)) = (g[key].as_f64(), w[key].as_f64()) {
                    assert!((gv - wv).abs() < 0.02, "{name}/{qid} {key}: got {gv}, want {wv}");
                }
            }
            if let Some(probs) = w["probabilities"].as_object() {
                for (k, wv) in probs {
                    let gv = g["probabilities"][k].as_f64().unwrap();
                    assert!((gv - wv.as_f64().unwrap()).abs() < 0.02, "{name}/{qid} p[{k}]");
                }
            }
            let (ga, wa) = (g["action"]["act_probability"].as_f64().unwrap(), w["action"]["act_probability"].as_f64().unwrap());
            assert!((ga - wa).abs() < 0.02, "{name}/{qid} act_probability: got {ga}, want {wa}");
        }
        assert_eq!(got["usage"]["input_tokens"], want["usage"]["input_tokens"], "{name} input_tokens");
    }
}
