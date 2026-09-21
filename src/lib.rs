//! Native Rust inference for the Laya System One decision model.
//!
//! Module map, in the order a request flows through them:
//! - `hub`: locate or download a checkpoint (Hugging Face cache or a local directory).
//! - `config`: parse `rl_agent_config.json` and the encoder's `config.json`.
//! - `sequence`: turn `state + question` into token ids, a port of `laya/common.py`.
//! - `model`: the ModernBERT encoder plus Laya's decision head, loaded from safetensors.
//! - `decide`: `Agent`, which runs one request end to end and shapes the JSON answer.
//! - `server`: the HTTP(S) API (`POST /v1/systemone`, `GET /health`).
//!
//! Around the request path: `settings` reads the TOML config and `tls` prepares certificates.

pub mod config;
pub mod decide;
pub mod hub;
pub mod model;
pub mod sequence;
pub mod server;
pub mod settings;
pub mod tls;

pub use decide::{Agent, Decider, LayaError};
pub use hub::CheckpointFiles;
pub use server::{Auth, HttpServer};
pub use settings::Settings;
