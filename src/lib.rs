//! Native Rust inference for local models. Serves Laya and GGUF models through one System One API.
//!
//! Module map, in the order a request flows through them:
//! - `hub`: locate, download, update, or delete a checkpoint (Hugging Face cache or a local directory).
//! - `config`: parse `rl_agent_config.json` and the encoder's `config.json`.
//! - `sequence`: turn `state + question` into token ids, a port of `laya/common.py`.
//! - `tournament`: split large `choice` questions so we stay out of the `choice:11+` bucket.
//! - `model`: the ModernBERT encoder plus Laya's decision head, loaded from safetensors.
//! - `gguf`: Qwen3 and MiniCPM5 GGUF, loaded through candle's quantized backends. Chat is extra;
//!   System One is the portable request and response.
//! - `decide`: `Agent`, which runs one System One request end to end and shapes the JSON answer.
//! - `server`: the HTTP(S) API (`POST /v1/systemone`, `POST /v1/chat/completions`, `GET /health`)
//!   and the browser UI at `/`.
//! - `web`: static files and the password cookie for that UI.
//!
//! Around the request path: `settings` reads the TOML config and `tls` prepares certificates.

pub mod config;
pub mod decide;
pub mod gguf;
pub mod hub;
pub mod model;
pub mod sequence;
pub mod server;
pub mod settings;
pub mod tls;
pub mod tournament;
pub mod web;

pub use decide::{Agent, ChatMessage, ChatOpts, DecidePolicy, Decider, LmrError};
pub use gguf::GgufEngine;
pub use hub::{CheckpointFiles, ModelFiles};
pub use server::{Auth, HttpServer};
pub use settings::Settings;
pub use web::WebConfig;
