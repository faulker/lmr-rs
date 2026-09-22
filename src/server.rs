//! HTTP(S) API. `POST /v1/systemone` takes `{"state": .., "questions": {..}}` and returns
//! the same System One answer document for every engine; `POST /v1/chat/completions` is an
//! extra OpenAI-shaped chat path for GGUF models; `POST /v1/rerank` orders `documents` by
//! relevance to `query` on reranker checkpoints; `GET /health` reports what is loaded.
//! When the web UI is on, `/` serves a browser console that posts to `/web/systemone`
//! (session cookie, never the API key). Each inference request prints a stderr line with
//! the UTC time, client IP, whether it came from the API or the web UI, and how long it
//! took; the question is not logged. Requests run one at a time behind a mutex on a
//! blocking thread; the model is not re-entrant on Metal anyway.
//!
//! The listener is bound synchronously in `bind` so a bad port fails before the process
//! daemonizes; `run` then builds the tokio runtime and serves until the process exits.

use std::net::{SocketAddr, TcpListener};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use axum::body::Bytes;
use axum::extract::rejection::{BytesRejection, FailedToBufferBody};
use axum::extract::{ConnectInfo, DefaultBodyLimit, State};
use axum::http::header;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Map, Value};
use subtle::ConstantTimeEq;

use crate::decide::{ChatMessage, ChatOpts, Decider, LmrError};
use crate::sequence::serialize_state;
use crate::tls::{self, TlsMaterial};
use crate::web::{self, WebConfig};

/// Largest request body accepted, in bytes. A few hundred options is well under this.
const MAX_BODY: usize = 1 << 20;

#[derive(Deserialize)]
struct SystemOneRequest {
    state: Value,
    questions: Map<String, Value>,
}

#[derive(Deserialize)]
struct LoginRequest {
    password: String,
}

/// OpenAI / llama.cpp `POST /v1/chat/completions` body. Extra fields are ignored.
#[derive(Deserialize)]
struct ChatCompletionRequest {
    #[serde(default)]
    model: Option<String>,
    messages: Vec<ChatMessageIn>,
    max_tokens: Option<i64>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<i64>,
    #[allow(dead_code)]
    min_p: Option<f64>,
    seed: Option<i64>,
    stream: Option<bool>,
    chat_template_kwargs: Option<Map<String, Value>>,
}

#[derive(Deserialize)]
struct ChatMessageIn {
    role: String,
    content: String,
}

/// llama.cpp / Jina / Cohere `POST /v1/rerank` body. `criteria` is accepted for `query`.
/// Documents are strings, `{"text": ..}` objects, or any JSON value (serialized like a
/// System One state). Extra fields are ignored.
#[derive(Deserialize)]
struct RerankRequest {
    #[serde(default)]
    model: Option<String>,
    #[serde(alias = "criteria")]
    query: String,
    documents: Vec<Value>,
    top_n: Option<i64>,
    #[serde(default)]
    return_documents: bool,
}

/// The text a reranker scores for one request document.
fn document_text(doc: &Value) -> String {
    match doc {
        Value::Object(o) => match o.get("text") {
            Some(Value::String(s)) => s.clone(),
            _ => serialize_state(doc),
        },
        other => serialize_state(other),
    }
}

/// Who may call. With `key = None` every request is accepted.
#[derive(Debug, Clone, Default)]
pub struct Auth {
    pub key: Option<String>,
    /// Also gate `GET /health`; off by default so clients can probe before configuring a key.
    pub health_requires_key: bool,
}

#[derive(Clone)]
struct AppState {
    decider: Arc<Mutex<Box<dyn Decider>>>,
    auth: Arc<Auth>,
    web: Arc<WebConfig>,
}

pub struct HttpServer {
    listener: TcpListener,
    app: Router,
    tls: Option<TlsMaterial>,
}

impl HttpServer {
    /// Bind `addr` (use port 0 to pick a free one) and return the server with its address.
    /// With `tls` the listener speaks HTTPS using that certificate.
    pub fn bind(
        addr: SocketAddr,
        decider: Box<dyn Decider>,
        auth: Auth,
        tls: Option<TlsMaterial>,
        web: WebConfig,
    ) -> Result<(Self, SocketAddr)> {
        let listener = TcpListener::bind(addr).with_context(|| format!("binding {addr}"))?;
        let bound = listener.local_addr()?;
        Ok((
            Self::from_listener(listener, decider, auth, tls, web),
            bound,
        ))
    }

    /// Wrap a listener bound earlier, e.g. before the process daemonized and loaded the model.
    pub fn from_listener(
        listener: TcpListener,
        decider: Box<dyn Decider>,
        auth: Auth,
        tls: Option<TlsMaterial>,
        web: WebConfig,
    ) -> Self {
        let enabled = web.enabled;
        let state = AppState {
            decider: Arc::new(Mutex::new(decider)),
            auth: Arc::new(auth),
            web: Arc::new(web),
        };
        let mut app = Router::new()
            .route("/health", get(health))
            .route("/v1/health", get(health))
            .route("/v1/models", get(models))
            .route("/v1/systemone", post(system_one))
            .route("/v1/chat/completions", post(chat_completions))
            .route("/v1/rerank", post(rerank));
        if enabled {
            app = app
                .route("/", get(|| async { web::index() }))
                .route("/web/app.css", get(|| async { web::css() }))
                .route("/web/app.js", get(|| async { web::js() }))
                .route("/web/session", get(web_session))
                .route("/web/login", post(web_login))
                .route("/web/logout", post(web_logout))
                .route("/web/info", get(web_info))
                .route("/web/systemone", post(web_system_one))
                .route("/web/chat/completions", post(web_chat_completions))
                .route("/web/rerank", post(web_rerank));
        }
        let app = app
            .fallback(|| async { reply(404, json!({ "message": "not found" })) })
            .layer(DefaultBodyLimit::max(MAX_BODY))
            .with_state(state);
        Self { listener, app, tls }
    }

    /// Serve forever on a runtime created here, so callers stay synchronous.
    pub fn run(self) -> Result<()> {
        let rt = tokio::runtime::Runtime::new().context("starting tokio runtime")?;
        rt.block_on(async move {
            self.listener.set_nonblocking(true)?;
            let make = self.app.into_make_service_with_connect_info::<SocketAddr>();
            match &self.tls {
                Some(material) => {
                    let config = tls::load(material).await?;
                    axum_server::from_tcp_rustls(self.listener, config)?
                        .serve(make)
                        .await?;
                }
                None => axum_server::from_tcp(self.listener)?.serve(make).await?,
            }
            Ok(())
        })
    }
}

/// JSON body with an explicit status.
fn reply(status: u16, body: Value) -> Response {
    (
        StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
        Json(body),
    )
        .into_response()
}

fn unauthorized() -> Response {
    reply(401, json!({ "message": "invalid or missing api key" }))
}

fn web_unauthorized() -> Response {
    reply(401, json!({ "message": "invalid or missing password" }))
}

/// Pull the body or turn a size/read failure into the matching JSON error.
fn read_body(body: Result<Bytes, BytesRejection>) -> Result<Bytes, Response> {
    match body {
        Ok(b) => Ok(b),
        Err(BytesRejection::FailedToBufferBody(FailedToBufferBody::LengthLimitError(_))) => {
            Err(reply(413, json!({ "message": "request body too large" })))
        }
        Err(e) => Err(reply(
            400,
            json!({ "message": format!("bad request: {e}") }),
        )),
    }
}

/// Accepts `Authorization: Bearer <key>` or `X-API-Key: <key>`, compared in constant time.
fn authorized(auth: &Auth, headers: &HeaderMap) -> bool {
    let Some(expected) = &auth.key else {
        return true;
    };
    let bearer = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.trim().strip_prefix("Bearer "))
        .map(str::trim);
    let x_api_key = headers
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .map(str::trim);
    bearer
        .into_iter()
        .chain(x_api_key)
        .any(|given| given.as_bytes().ct_eq(expected.as_bytes()).into())
}

/// UTC `YYYY-MM-DDTHH:MM:SSZ` for request logs. Second resolution is enough.
fn utc_timestamp(now: SystemTime) -> String {
    let secs = now.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let hour = rem / 3_600;
    let min = (rem % 3_600) / 60;
    let sec = rem % 60;

    // Howard Hinnant's civil-from-days; `days` is days since 1970-01-01.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = u32::try_from(z - era * 146_097).unwrap_or(0);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = i64::from(yoe) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = y + if month <= 2 { 1 } else { 0 };

    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{min:02}:{sec:02}Z")
}

/// One stderr line for an inference request: time, client, api/web, duration.
/// The question and state are intentionally omitted.
fn request_log_line(when: &str, who: &str, via: &str, elapsed: Duration) -> String {
    format!("{when}  {who}  {via}  {}ms", elapsed.as_millis())
}

/// Print a request log line to stderr (and to the daemon log when stderr is redirected).
fn log_request(who: &str, via: &str, started: Instant) {
    eprintln!(
        "{}",
        request_log_line(
            &utc_timestamp(SystemTime::now()),
            who,
            via,
            started.elapsed()
        )
    );
}

/// Client identity for the log: canonical IP, so IPv4-mapped IPv6 prints as IPv4.
fn peer_who(peer: SocketAddr) -> String {
    peer.ip().to_canonical().to_string()
}

async fn health(State(s): State<AppState>, headers: HeaderMap) -> Response {
    if s.auth.health_requires_key && !authorized(&s.auth, &headers) {
        return unauthorized();
    }
    let d = s.decider.lock().unwrap_or_else(|e| e.into_inner());
    reply(200, d.info())
}

/// OpenAI / llama.cpp model list. Always one loaded checkpoint, like llama-server.
async fn models(State(s): State<AppState>, headers: HeaderMap) -> Response {
    if !authorized(&s.auth, &headers) {
        return unauthorized();
    }
    let d = s.decider.lock().unwrap_or_else(|e| e.into_inner());
    reply(
        200,
        json!({
            "object": "list",
            "data": [{
                "id": d.openai_model_id(),
                "object": "model",
                "created": 0,
                "owned_by": "lmr-rs",
            }],
        }),
    )
}

async fn system_one(
    State(s): State<AppState>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    let started = Instant::now();
    let who = peer_who(peer);
    let response = if !authorized(&s.auth, &headers) {
        unauthorized()
    } else {
        decide_system_one(s.decider.clone(), body).await
    };
    log_request(&who, "api", started);
    response
}

/// Same document as `POST /v1/systemone`, gated by the web session instead of the API key.
async fn web_system_one(
    State(s): State<AppState>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    let started = Instant::now();
    let who = peer_who(peer);
    let response = if !s.web.allows(&headers) {
        web_unauthorized()
    } else {
        decide_system_one(s.decider.clone(), body).await
    };
    log_request(&who, "web", started);
    response
}

/// Same document as `POST /v1/chat/completions`, gated by the web session instead of the API key.
async fn web_chat_completions(
    State(s): State<AppState>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    let started = Instant::now();
    let who = peer_who(peer);
    let response = if !s.web.allows(&headers) {
        web_unauthorized()
    } else {
        decide_chat(s.decider.clone(), body).await
    };
    log_request(&who, "web", started);
    response
}

/// Same document as `POST /v1/rerank`, gated by the web session instead of the API key.
async fn web_rerank(
    State(s): State<AppState>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    let started = Instant::now();
    let who = peer_who(peer);
    let response = if !s.web.allows(&headers) {
        web_unauthorized()
    } else {
        decide_rerank(s.decider.clone(), body).await
    };
    log_request(&who, "web", started);
    response
}

async fn rerank(
    State(s): State<AppState>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    let started = Instant::now();
    let who = peer_who(peer);
    let response = if !authorized(&s.auth, &headers) {
        unauthorized()
    } else {
        decide_rerank(s.decider.clone(), body).await
    };
    log_request(&who, "api", started);
    response
}

/// Parse a rerank body, score it on the blocking thread, and shape the Jina / llama.cpp
/// response: `results` best first, each `{index, relevance_score[, document]}`.
async fn decide_rerank(
    decider: Arc<Mutex<Box<dyn Decider>>>,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    let body = match read_body(body) {
        Ok(b) => b,
        Err(resp) => return resp,
    };
    let req: RerankRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => {
            return openai_error(
                422,
                "invalid_request_error",
                &format!("invalid request: {e}"),
            )
        }
    };
    if req.documents.is_empty() {
        return openai_error(422, "invalid_request_error", "documents must not be empty");
    }
    let top_n = match req.top_n {
        None => req.documents.len(),
        Some(n) if n >= 1 => n as usize,
        Some(_) => return openai_error(422, "invalid_request_error", "top_n must be at least 1"),
    };
    let texts: Vec<String> = req.documents.iter().map(document_text).collect();
    let query = req.query.clone();
    let result = tokio::task::spawn_blocking(move || {
        let mut d = decider.lock().unwrap_or_else(|e| e.into_inner());
        let id = d.openai_model_id();
        match d.as_rerank() {
            Some(engine) => engine.rerank(&query, &texts).map(|r| (id, r)),
            None => Err(LmrError::Invalid(
                "this checkpoint is not a reranker; use POST /v1/systemone".into(),
            )),
        }
    })
    .await;
    match result {
        Ok(Ok((id, ranked))) => {
            let results: Vec<Value> = ranked
                .results
                .iter()
                .take(top_n)
                .map(|r| {
                    let mut item = json!({
                        "index": r.index,
                        "relevance_score": r.relevance_score,
                    });
                    if req.return_documents {
                        item["document"] = req.documents[r.index].clone();
                    }
                    item
                })
                .collect();
            reply(
                200,
                json!({
                    "model": req.model.unwrap_or(id),
                    "object": "list",
                    "results": results,
                    "usage": { "prompt_tokens": ranked.tokens, "total_tokens": ranked.tokens },
                }),
            )
        }
        Ok(Err(LmrError::Invalid(m))) => openai_error(422, "invalid_request_error", &m),
        Ok(Err(LmrError::Model(e))) => {
            eprintln!("model error: {e:#}");
            openai_error(500, "api_error", "model error")
        }
        Err(e) => {
            eprintln!("inference task failed: {e}");
            openai_error(500, "api_error", "model error")
        }
    }
}

/// llama.cpp / OpenAI error object used on the chat and models routes.
fn openai_error(status: u16, kind: &str, message: &str) -> Response {
    reply(
        status,
        json!({ "error": { "code": status, "message": message, "type": kind } }),
    )
}

async fn chat_completions(
    State(s): State<AppState>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    let started = Instant::now();
    let who = peer_who(peer);
    let response = if !authorized(&s.auth, &headers) {
        unauthorized()
    } else {
        decide_chat(s.decider.clone(), body).await
    };
    log_request(&who, "api", started);
    response
}

async fn decide_chat(
    decider: Arc<Mutex<Box<dyn Decider>>>,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    let body = match read_body(body) {
        Ok(b) => b,
        Err(resp) => return resp,
    };
    let req: ChatCompletionRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => {
            return openai_error(
                422,
                "invalid_request_error",
                &format!("invalid request: {e}"),
            )
        }
    };
    if req.stream == Some(true) {
        return openai_error(
            400,
            "invalid_request_error",
            "stream is not supported; omit stream or set stream=false",
        );
    }
    if req.messages.is_empty() {
        return openai_error(422, "invalid_request_error", "messages must not be empty");
    }
    let messages: Vec<ChatMessage> = req
        .messages
        .into_iter()
        .map(|m| ChatMessage {
            role: m.role,
            content: m.content,
        })
        .collect();
    let enable_thinking = req
        .chat_template_kwargs
        .as_ref()
        .and_then(|m| m.get("enable_thinking"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let opts = ChatOpts {
        max_tokens: req.max_tokens.and_then(|n| usize::try_from(n).ok()),
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k.and_then(|n| usize::try_from(n).ok()),
        seed: req.seed.and_then(|n| u64::try_from(n).ok()),
        enable_thinking,
    };
    let _ = req.model;
    let result = tokio::task::spawn_blocking(move || {
        let mut d = decider.lock().unwrap_or_else(|e| e.into_inner());
        match d.as_chat() {
            Some(chat) => chat.chat(&messages, &opts),
            None => Err(LmrError::Invalid(
                "this checkpoint is a System One model; use POST /v1/systemone".into(),
            )),
        }
    })
    .await;
    match result {
        Ok(Ok(v)) => reply(200, v),
        Ok(Err(LmrError::Invalid(m))) => openai_error(422, "invalid_request_error", &m),
        Ok(Err(LmrError::Model(e))) => {
            eprintln!("model error: {e:#}");
            openai_error(500, "api_error", "model error")
        }
        Err(e) => {
            eprintln!("inference task failed: {e}");
            openai_error(500, "api_error", "model error")
        }
    }
}

async fn decide_system_one(
    decider: Arc<Mutex<Box<dyn Decider>>>,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    let body = match read_body(body) {
        Ok(b) => b,
        Err(resp) => return resp,
    };
    let req: SystemOneRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return reply(422, json!({ "message": format!("invalid request: {e}") })),
    };
    let result = tokio::task::spawn_blocking(move || {
        let mut d = decider.lock().unwrap_or_else(|e| e.into_inner());
        d.system_one(&req.state, &req.questions)
    })
    .await;
    match result {
        Ok(Ok(v)) => reply(200, v),
        Ok(Err(LmrError::Invalid(m))) => reply(422, json!({ "message": m })),
        Ok(Err(LmrError::Model(e))) => {
            eprintln!("model error: {e:#}");
            reply(500, json!({ "message": "model error" }))
        }
        Err(e) => {
            eprintln!("inference task failed: {e}");
            reply(500, json!({ "message": "model error" }))
        }
    }
}

async fn web_session(State(s): State<AppState>, headers: HeaderMap) -> Response {
    if !s.web.allows(&headers) {
        return web_unauthorized();
    }
    reply(200, json!({ "ok": true, "locked": s.web.is_locked() }))
}

async fn web_info(State(s): State<AppState>, headers: HeaderMap) -> Response {
    if !s.web.allows(&headers) {
        return web_unauthorized();
    }
    let d = s.decider.lock().unwrap_or_else(|e| e.into_inner());
    reply(200, d.info())
}

async fn web_login(State(s): State<AppState>, body: Result<Bytes, BytesRejection>) -> Response {
    let body = match read_body(body) {
        Ok(b) => b,
        Err(resp) => return resp,
    };
    let req: LoginRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return reply(422, json!({ "message": format!("invalid request: {e}") })),
    };
    if !s.web.password_matches(&req.password) {
        return web_unauthorized();
    }
    let mut res = reply(200, json!({ "ok": true }));
    if s.web.is_locked() {
        res.headers_mut()
            .insert(header::SET_COOKIE, s.web.login_cookie());
    }
    res
}

async fn web_logout(State(s): State<AppState>) -> Response {
    let mut res = reply(200, json!({ "ok": true }));
    res.headers_mut()
        .insert(header::SET_COOKIE, s.web.logout_cookie());
    res
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decide::{ChatEngine, Ranked, RerankEngine, Reranked};
    use std::io::{Read, Write};
    use std::net::TcpStream;

    struct Stub;

    impl ChatEngine for Stub {
        fn chat(&mut self, messages: &[ChatMessage], _opts: &ChatOpts) -> Result<Value, LmrError> {
            if messages.is_empty() {
                return Err(LmrError::Invalid("messages must not be empty".into()));
            }
            Ok(json!({
                "id": "chatcmpl-stub",
                "object": "chat.completion",
                "created": 0,
                "model": "stub",
                "choices": [{
                    "index": 0,
                    "message": { "role": "assistant", "content": "hello" },
                    "finish_reason": "stop",
                }],
                "usage": { "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2 },
            }))
        }
    }

    impl RerankEngine for Stub {
        /// Longer documents score higher, so the order is predictable.
        fn rerank(&mut self, query: &str, documents: &[String]) -> Result<Reranked, LmrError> {
            if query.is_empty() {
                return Err(LmrError::Invalid("query must not be empty".into()));
            }
            let mut results: Vec<Ranked> = documents
                .iter()
                .enumerate()
                .map(|(index, d)| Ranked {
                    index,
                    relevance_score: d.len() as f64 / 100.0,
                })
                .collect();
            results.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap());
            Ok(Reranked {
                results,
                tokens: documents.len(),
            })
        }
    }

    impl Decider for Stub {
        fn system_one(
            &mut self,
            _state: &Value,
            questions: &Map<String, Value>,
        ) -> Result<Value, LmrError> {
            if questions.is_empty() {
                return Err(LmrError::Invalid("questions must not be empty".into()));
            }
            Ok(json!({
                "model": "stub",
                "answers": {
                    "category": {
                        "type": "choice",
                        "choice": "Dining",
                        "probabilities": { "Dining": 0.9 },
                        "confidence": 0.9,
                        "action": { "act_probability": 0.9 },
                    }
                },
                "usage": { "input_tokens": 1, "output_tokens": 0 },
            }))
        }
        fn info(&self) -> Value {
            json!({ "status": "ok", "model": "stub", "engine": "laya", "checkpoint": "stub" })
        }
        fn as_chat(&mut self) -> Option<&mut dyn ChatEngine> {
            Some(self)
        }
        fn as_rerank(&mut self) -> Option<&mut dyn RerankEngine> {
            Some(self)
        }
    }

    fn start_with(auth: Auth, tls: Option<TlsMaterial>, web: WebConfig) -> SocketAddr {
        let (server, addr) = HttpServer::bind(
            "127.0.0.1:0".parse().unwrap(),
            Box::new(Stub),
            auth,
            tls,
            web,
        )
        .unwrap();
        std::thread::spawn(move || server.run().unwrap());
        addr
    }

    fn start(key: Option<&str>) -> SocketAddr {
        start_with(
            Auth {
                key: key.map(str::to_string),
                ..Default::default()
            },
            None,
            WebConfig::default(),
        )
    }

    /// Minimal HTTP/1.0 client so the plain tests need no extra dependency.
    fn exchange(
        addr: SocketAddr,
        method: &str,
        path: &str,
        headers: &str,
        body: &str,
    ) -> (u16, Vec<(String, String)>, String) {
        let mut s = TcpStream::connect(addr).unwrap();
        write!(
            s,
            "{method} {path} HTTP/1.0\r\nHost: x\r\nContent-Length: {}\r\n{headers}\r\n{body}",
            body.len()
        )
        .unwrap();
        let mut out = String::new();
        s.read_to_string(&mut out).unwrap();
        let (head, body) = out.split_once("\r\n\r\n").unwrap_or((out.as_str(), ""));
        let status: u16 = head.split_whitespace().nth(1).unwrap().parse().unwrap();
        let hdrs = head
            .lines()
            .skip(1)
            .filter_map(|line| {
                let (k, v) = line.split_once(':')?;
                Some((k.trim().to_ascii_lowercase(), v.trim().to_string()))
            })
            .collect();
        (status, hdrs, body.to_string())
    }

    fn call(addr: SocketAddr, method: &str, path: &str, headers: &str, body: &str) -> (u16, Value) {
        let (status, _, body) = exchange(addr, method, path, headers, body);
        (status, serde_json::from_str(&body).unwrap())
    }

    fn header<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
        headers
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }

    /// Portable System One document: same keys whether Laya or GGUF is loaded.
    fn assert_systemone_shape(body: &Value) {
        assert!(body.get("model").and_then(Value::as_str).is_some());
        assert!(body["answers"].is_object());
        assert!(body["usage"]["input_tokens"].as_u64().is_some());
        assert_eq!(body["usage"]["output_tokens"], 0);
        assert!(body.get("state").is_none());
        let ans = &body["answers"]["category"];
        assert_eq!(ans["type"], "choice");
        assert!(ans.get("choice").is_some());
        assert!(ans["probabilities"].is_object());
        assert!(ans["confidence"].as_f64().is_some());
        assert!(ans["action"]["act_probability"].as_f64().is_some());
    }

    const REQ: &str = r#"{"state":{"transactionTitle":"CAFE"},"model":"jev-latest",
        "questions":{"category":{"type":"choice","instructions":"?","criteria":{"Dining":null}}}}"#;

    #[test]
    fn answers_health_and_systemone() {
        let addr = start(None);
        let (status, body) = call(addr, "GET", "/health", "", "");
        assert_eq!((status, body["model"].as_str()), (200, Some("stub")));
        assert_eq!(body["status"], "ok");
        let (status, body) = call(addr, "GET", "/v1/health", "", "");
        assert_eq!((status, body["status"].as_str()), (200, Some("ok")));
        let (status, body) = call(addr, "POST", "/v1/systemone", "", REQ);
        assert_eq!(status, 200);
        assert_systemone_shape(&body);
        assert_eq!(body["answers"]["category"]["choice"], "Dining");
    }

    #[test]
    fn openai_chat_and_models() {
        let addr = start(None);
        let (status, body) = call(addr, "GET", "/v1/models", "", "");
        assert_eq!(status, 200);
        assert_eq!(body["object"], "list");
        assert_eq!(body["data"][0]["id"], "stub");
        assert_eq!(body["data"][0]["object"], "model");
        let (status, body) = call(
            addr,
            "POST",
            "/v1/chat/completions",
            "",
            r#"{"model":"gpt-3.5-turbo","messages":[{"role":"user","content":"hi"}],"foo":1}"#,
        );
        assert_eq!(status, 200);
        assert_eq!(body["object"], "chat.completion");
        assert_eq!(body["choices"][0]["message"]["content"], "hello");
        let (status, body) = call(
            addr,
            "POST",
            "/v1/chat/completions",
            "",
            r#"{"messages":[{"role":"user","content":"hi"}],"stream":true}"#,
        );
        assert_eq!(status, 400);
        assert_eq!(body["error"]["type"], "invalid_request_error");
        let (status, body) = call(
            addr,
            "POST",
            "/v1/chat/completions",
            "",
            r#"{"messages":[]}"#,
        );
        assert_eq!(status, 422);
        assert!(body["error"]["message"].as_str().unwrap().contains("empty"));
    }

    const RERANK_REQ: &str = r#"{"query":"coffee","documents":["tea","espresso","a latte please"],"top_n":2,"return_documents":true}"#;

    #[test]
    fn rerank_orders_documents_and_honours_top_n() {
        let addr = start(None);
        let (status, body) = call(addr, "POST", "/v1/rerank", "", RERANK_REQ);
        assert_eq!(status, 200);
        assert_eq!(body["object"], "list");
        assert_eq!(body["model"], "stub");
        let results = body["results"].as_array().unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0]["index"], 2);
        assert_eq!(results[0]["document"], "a latte please");
        assert_eq!(results[1]["index"], 1);
        assert!(
            results[0]["relevance_score"].as_f64().unwrap()
                > results[1]["relevance_score"].as_f64().unwrap()
        );
        assert_eq!(body["usage"]["prompt_tokens"], 3);
        // `criteria` is an alias for `query`; objects use their `text`; no documents echoed.
        let (status, body) = call(
            addr,
            "POST",
            "/v1/rerank",
            "",
            r#"{"criteria":"coffee","documents":[{"text":"x"},{"text":"longer text"},{"id":7}]}"#,
        );
        assert_eq!(status, 200);
        let results = body["results"].as_array().unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0]["index"], 1);
        assert!(results[0].get("document").is_none());
        let (status, body) = call(
            addr,
            "POST",
            "/v1/rerank",
            "",
            r#"{"query":"q","documents":[]}"#,
        );
        assert_eq!(status, 422);
        assert!(body["error"]["message"].as_str().unwrap().contains("empty"));
        let (status, _) = call(
            addr,
            "POST",
            "/v1/rerank",
            "",
            r#"{"query":"q","documents":["a"],"top_n":0}"#,
        );
        assert_eq!(status, 422);
        let (status, _) = call(addr, "POST", "/v1/rerank", "", r#"{"documents":["a"]}"#);
        assert_eq!(status, 422);
    }

    #[test]
    fn document_text_reads_strings_text_fields_and_json() {
        assert_eq!(document_text(&json!("plain")), "plain");
        assert_eq!(document_text(&json!({"text": "t", "id": 1})), "t");
        assert_eq!(document_text(&json!({"id": 1})), "{\"id\": 1}");
        assert_eq!(document_text(&json!(["a", 1])), "[\"a\", 1]");
    }

    #[test]
    fn rejects_bad_input_and_unknown_paths() {
        let addr = start(None);
        assert_eq!(call(addr, "POST", "/v1/systemone", "", "not json").0, 422);
        let (status, body) = call(
            addr,
            "POST",
            "/v1/systemone",
            "",
            r#"{"state":"x","questions":{}}"#,
        );
        assert_eq!(status, 422);
        assert!(body["message"].as_str().unwrap().contains("empty"));
        assert_eq!(call(addr, "GET", "/nope", "", "").0, 404);
        let huge = format!(
            r#"{{"state":"{}","questions":{{}}}}"#,
            "x".repeat(MAX_BODY + 1)
        );
        let (status, body) = call(addr, "POST", "/v1/systemone", "", &huge);
        assert_eq!(status, 413);
        assert!(body["message"].as_str().unwrap().contains("too large"));
    }

    #[test]
    fn api_key_is_enforced_only_when_set() {
        let addr = start(Some("s3cret"));
        assert_eq!(call(addr, "POST", "/v1/systemone", "", REQ).0, 401);
        assert_eq!(
            call(
                addr,
                "POST",
                "/v1/systemone",
                "Authorization: Bearer wrong\r\n",
                REQ
            )
            .0,
            401
        );
        assert_eq!(
            call(
                addr,
                "POST",
                "/v1/systemone",
                "Authorization: Bearer s3cret\r\n",
                REQ
            )
            .0,
            200
        );
        assert_eq!(
            call(addr, "POST", "/v1/systemone", "X-API-Key: s3cret\r\n", REQ).0,
            200
        );
        assert_eq!(
            call(addr, "POST", "/v1/systemone", "X-API-Key: s3cre\r\n", REQ).0,
            401
        );
        assert_eq!(call(addr, "POST", "/v1/rerank", "", RERANK_REQ).0, 401);
        assert_eq!(
            call(
                addr,
                "POST",
                "/v1/rerank",
                "X-API-Key: s3cret\r\n",
                RERANK_REQ
            )
            .0,
            200
        );
        // Health never needs a key by default so a client can probe before configuring one.
        assert_eq!(call(addr, "GET", "/health", "", "").0, 200);
    }

    #[test]
    fn health_can_require_key() {
        let addr = start_with(
            Auth {
                key: Some("k".into()),
                health_requires_key: true,
            },
            None,
            WebConfig::default(),
        );
        assert_eq!(call(addr, "GET", "/health", "", "").0, 401);
        assert_eq!(call(addr, "GET", "/health", "X-API-Key: k\r\n", "").0, 200);
    }

    #[test]
    fn serves_https_with_generated_cert() {
        let dir = std::env::temp_dir().join(format!("lmr-rs-https-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let material =
            tls::prepare(&Default::default(), &dir, "127.0.0.1".parse().unwrap()).unwrap();
        let addr = start_with(
            Auth {
                key: Some("k".into()),
                ..Default::default()
            },
            Some(material),
            WebConfig::default(),
        );
        let client = reqwest::blocking::Client::builder()
            .danger_accept_invalid_certs(true)
            .build()
            .unwrap();
        let url = format!("https://{addr}/v1/systemone");
        let res = client
            .post(&url)
            .header("X-API-Key", "k")
            .body(REQ)
            .send()
            .unwrap();
        assert_eq!(res.status(), 200);
        let body: Value = res.json().unwrap();
        assert_systemone_shape(&body);
        assert_eq!(body["answers"]["category"]["choice"], "Dining");
        assert_eq!(client.post(&url).body(REQ).send().unwrap().status(), 401);
        // Plain HTTP on a TLS port is refused rather than answered.
        assert!(reqwest::blocking::get(format!("http://{addr}/health")).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn web_off_root_is_404() {
        let addr = start(None);
        let (status, body) = call(addr, "GET", "/", "", "");
        assert_eq!(status, 404);
        assert_eq!(body["message"], "not found");
    }

    #[test]
    fn web_open_serves_html_and_query_without_api_key() {
        let addr = start_with(
            Auth {
                key: Some("s3cret".into()),
                ..Default::default()
            },
            None,
            WebConfig::new(true, None, false),
        );
        let (status, headers, body) = exchange(addr, "GET", "/", "", "");
        assert_eq!(status, 200);
        assert!(header(&headers, "content-type")
            .unwrap()
            .starts_with("text/html"));
        assert!(body.contains("lmr-rs") || body.contains("id=\"editor\""));
        let (status, headers, _) = exchange(addr, "GET", "/web/app.css", "", "");
        assert_eq!(status, 200);
        assert!(header(&headers, "content-type")
            .unwrap()
            .starts_with("text/css"));
        let (status, session) = call(addr, "GET", "/web/session", "", "");
        assert_eq!(status, 200);
        assert_eq!(session["ok"], true);
        assert_eq!(session["locked"], false);
        let (status, body) = call(addr, "POST", "/web/systemone", "", REQ);
        assert_eq!(status, 200);
        assert_systemone_shape(&body);
        assert_eq!(body["answers"]["category"]["choice"], "Dining");
        let (status, body) = call(
            addr,
            "POST",
            "/web/chat/completions",
            "",
            r#"{"messages":[{"role":"user","content":"hi"}]}"#,
        );
        assert_eq!(status, 200);
        assert_eq!(body["choices"][0]["message"]["content"], "hello");
        let (status, body) = call(addr, "POST", "/web/rerank", "", RERANK_REQ);
        assert_eq!(status, 200);
        assert_eq!(body["results"][0]["index"], 2);
        // The programmatic API still needs the key.
        assert_eq!(call(addr, "POST", "/v1/systemone", "", REQ).0, 401);
        assert_eq!(
            call(addr, "POST", "/v1/systemone", "X-API-Key: s3cret\r\n", REQ).0,
            200
        );
    }

    #[test]
    fn web_password_gates_query() {
        let addr = start_with(
            Auth::default(),
            None,
            WebConfig::new(true, Some("desk".into()), false),
        );
        assert_eq!(call(addr, "POST", "/web/systemone", "", REQ).0, 401);
        assert_eq!(
            call(
                addr,
                "POST",
                "/web/chat/completions",
                "",
                r#"{"messages":[{"role":"user","content":"hi"}]}"#
            )
            .0,
            401
        );
        assert_eq!(call(addr, "POST", "/web/rerank", "", RERANK_REQ).0, 401);
        assert_eq!(call(addr, "GET", "/web/session", "", "").0, 401);
        assert_eq!(
            call(addr, "POST", "/web/login", "", r#"{"password":"wrong"}"#).0,
            401
        );
        let (status, headers, _) =
            exchange(addr, "POST", "/web/login", "", r#"{"password":"desk"}"#);
        assert_eq!(status, 200);
        let set_cookie = header(&headers, "set-cookie").expect("set-cookie");
        let token = set_cookie
            .split(';')
            .next()
            .unwrap()
            .strip_prefix("lmr_web=")
            .expect("lmr_web cookie");
        assert!(!token.is_empty());
        let cookie = format!("Cookie: lmr_web={token}\r\n");
        let (status, session) = call(addr, "GET", "/web/session", &cookie, "");
        assert_eq!(status, 200);
        assert_eq!(session["locked"], true);
        assert_eq!(call(addr, "POST", "/web/systemone", &cookie, REQ).0, 200);
        assert_eq!(
            call(
                addr,
                "POST",
                "/web/chat/completions",
                &cookie,
                r#"{"messages":[{"role":"user","content":"hi"}]}"#
            )
            .0,
            200
        );
        assert_eq!(
            call(addr, "POST", "/web/rerank", &cookie, RERANK_REQ).0,
            200
        );
        assert_eq!(call(addr, "GET", "/web/info", &cookie, "").0, 200);
    }

    #[test]
    fn utc_timestamp_formats_known_instants() {
        assert_eq!(utc_timestamp(UNIX_EPOCH), "1970-01-01T00:00:00Z");
        assert_eq!(
            utc_timestamp(UNIX_EPOCH + Duration::from_secs(1_700_000_000)),
            "2023-11-14T22:13:20Z"
        );
        assert_eq!(
            utc_timestamp(UNIX_EPOCH + Duration::from_secs(1_750_000_000)),
            "2025-06-15T15:06:40Z"
        );
    }

    #[test]
    fn request_log_line_has_time_who_and_duration_not_the_question() {
        let line = request_log_line(
            "2026-09-21T03:17:42Z",
            "127.0.0.1",
            "api",
            Duration::from_millis(41),
        );
        assert_eq!(line, "2026-09-21T03:17:42Z  127.0.0.1  api  41ms");
    }

    #[test]
    fn peer_who_canonicalizes_ipv4_mapped_ipv6() {
        let v4: SocketAddr = "127.0.0.1:8321".parse().unwrap();
        assert_eq!(peer_who(v4), "127.0.0.1");
        let mapped: SocketAddr = "[::ffff:192.0.2.1]:8321".parse().unwrap();
        assert_eq!(peer_who(mapped), "192.0.2.1");
    }
}
