//! HTTP(S) API. `POST /v1/systemone` takes `{"state": .., "questions": {..}}` and returns the
//! System One answer document; `GET /health` reports what is loaded. Requests run one at a
//! time behind a mutex on a blocking thread; the model is not re-entrant on Metal anyway.
//!
//! The listener is bound synchronously in `bind` so a bad port fails before the process
//! daemonizes; `run` then builds the tokio runtime and serves until the process exits.

use std::net::{SocketAddr, TcpListener};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use axum::body::Bytes;
use axum::extract::rejection::{BytesRejection, FailedToBufferBody};
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Map, Value};
use subtle::ConstantTimeEq;

use crate::decide::{Decider, LayaError};
use crate::tls::{self, TlsMaterial};

/// Largest request body accepted, in bytes. A few hundred options is well under this.
const MAX_BODY: usize = 1 << 20;

#[derive(Deserialize)]
struct SystemOneRequest {
    state: Value,
    questions: Map<String, Value>,
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
    ) -> Result<(Self, SocketAddr)> {
        let listener = TcpListener::bind(addr).with_context(|| format!("binding {addr}"))?;
        let bound = listener.local_addr()?;
        Ok((Self::from_listener(listener, decider, auth, tls), bound))
    }

    /// Wrap a listener bound earlier, e.g. before the process daemonized and loaded the model.
    pub fn from_listener(
        listener: TcpListener,
        decider: Box<dyn Decider>,
        auth: Auth,
        tls: Option<TlsMaterial>,
    ) -> Self {
        let state = AppState {
            decider: Arc::new(Mutex::new(decider)),
            auth: Arc::new(auth),
        };
        let app = Router::new()
            .route("/health", get(health))
            .route("/v1/systemone", post(system_one))
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
            let make = self.app.into_make_service();
            match &self.tls {
                Some(material) => {
                    let config = tls::load(material).await?;
                    axum_server::from_tcp_rustls(self.listener, config)?.serve(make).await?;
                }
                None => axum_server::from_tcp(self.listener)?.serve(make).await?,
            }
            Ok(())
        })
    }
}

/// JSON body with an explicit status.
fn reply(status: u16, body: Value) -> Response {
    (StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR), Json(body)).into_response()
}

fn unauthorized() -> Response {
    reply(401, json!({ "message": "invalid or missing api key" }))
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
    let x_api_key = headers.get("x-api-key").and_then(|v| v.to_str().ok()).map(str::trim);
    bearer
        .into_iter()
        .chain(x_api_key)
        .any(|given| given.as_bytes().ct_eq(expected.as_bytes()).into())
}

async fn health(State(s): State<AppState>, headers: HeaderMap) -> Response {
    if s.auth.health_requires_key && !authorized(&s.auth, &headers) {
        return unauthorized();
    }
    let d = s.decider.lock().unwrap_or_else(|e| e.into_inner());
    reply(200, d.info())
}

async fn system_one(
    State(s): State<AppState>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    if !authorized(&s.auth, &headers) {
        return unauthorized();
    }
    let body = match body {
        Ok(b) => b,
        Err(BytesRejection::FailedToBufferBody(FailedToBufferBody::LengthLimitError(_))) => {
            return reply(413, json!({ "message": "request body too large" }))
        }
        Err(e) => return reply(400, json!({ "message": format!("bad request: {e}") })),
    };
    let req: SystemOneRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return reply(422, json!({ "message": format!("invalid request: {e}") })),
    };
    let decider = s.decider.clone();
    let result = tokio::task::spawn_blocking(move || {
        let d = decider.lock().unwrap_or_else(|e| e.into_inner());
        d.system_one(&req.state, &req.questions)
    })
    .await;
    match result {
        Ok(Ok(v)) => reply(200, v),
        Ok(Err(LayaError::Invalid(m))) => reply(422, json!({ "message": m })),
        Ok(Err(LayaError::Model(e))) => {
            eprintln!("model error: {e:#}");
            reply(500, json!({ "message": "model error" }))
        }
        Err(e) => {
            eprintln!("inference task failed: {e}");
            reply(500, json!({ "message": "model error" }))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpStream;

    struct Stub;

    impl Decider for Stub {
        fn system_one(&self, state: &Value, questions: &Map<String, Value>) -> Result<Value, LayaError> {
            if questions.is_empty() {
                return Err(LayaError::Invalid("questions must not be empty".into()));
            }
            Ok(json!({ "model": "stub", "answers": { "category": { "choice": "Dining",
                "confidence": 0.9 } }, "state": state }))
        }
        fn info(&self) -> Value {
            json!({ "model": "stub" })
        }
    }

    fn start_with(auth: Auth, tls: Option<TlsMaterial>) -> SocketAddr {
        let (server, addr) = HttpServer::bind("127.0.0.1:0".parse().unwrap(), Box::new(Stub), auth, tls).unwrap();
        std::thread::spawn(move || server.run().unwrap());
        addr
    }

    fn start(key: Option<&str>) -> SocketAddr {
        start_with(Auth { key: key.map(str::to_string), ..Default::default() }, None)
    }

    /// Minimal HTTP/1.0 client so the plain tests need no extra dependency.
    fn call(addr: SocketAddr, method: &str, path: &str, headers: &str, body: &str) -> (u16, Value) {
        let mut s = TcpStream::connect(addr).unwrap();
        write!(
            s,
            "{method} {path} HTTP/1.0\r\nHost: x\r\nContent-Length: {}\r\n{headers}\r\n{body}",
            body.len()
        )
        .unwrap();
        let mut out = String::new();
        s.read_to_string(&mut out).unwrap();
        let status: u16 = out.split_whitespace().nth(1).unwrap().parse().unwrap();
        let json_body = out.split("\r\n\r\n").nth(1).unwrap_or("null");
        (status, serde_json::from_str(json_body).unwrap())
    }

    const REQ: &str = r#"{"state":{"transactionTitle":"CAFE"},"model":"jev-latest",
        "questions":{"category":{"type":"choice","instructions":"?","criteria":{"Dining":null}}}}"#;

    #[test]
    fn answers_health_and_systemone() {
        let addr = start(None);
        let (status, body) = call(addr, "GET", "/health", "", "");
        assert_eq!((status, body["model"].as_str()), (200, Some("stub")));
        let (status, body) = call(addr, "POST", "/v1/systemone", "", REQ);
        assert_eq!(status, 200);
        assert_eq!(body["answers"]["category"]["choice"], "Dining");
        assert_eq!(body["state"]["transactionTitle"], "CAFE");
    }

    #[test]
    fn rejects_bad_input_and_unknown_paths() {
        let addr = start(None);
        assert_eq!(call(addr, "POST", "/v1/systemone", "", "not json").0, 422);
        let (status, body) = call(addr, "POST", "/v1/systemone", "", r#"{"state":"x","questions":{}}"#);
        assert_eq!(status, 422);
        assert!(body["message"].as_str().unwrap().contains("empty"));
        assert_eq!(call(addr, "GET", "/nope", "", "").0, 404);
        let huge = format!(r#"{{"state":"{}","questions":{{}}}}"#, "x".repeat(MAX_BODY + 1));
        let (status, body) = call(addr, "POST", "/v1/systemone", "", &huge);
        assert_eq!(status, 413);
        assert!(body["message"].as_str().unwrap().contains("too large"));
    }

    #[test]
    fn api_key_is_enforced_only_when_set() {
        let addr = start(Some("s3cret"));
        assert_eq!(call(addr, "POST", "/v1/systemone", "", REQ).0, 401);
        assert_eq!(call(addr, "POST", "/v1/systemone", "Authorization: Bearer wrong\r\n", REQ).0, 401);
        assert_eq!(call(addr, "POST", "/v1/systemone", "Authorization: Bearer s3cret\r\n", REQ).0, 200);
        assert_eq!(call(addr, "POST", "/v1/systemone", "X-API-Key: s3cret\r\n", REQ).0, 200);
        assert_eq!(call(addr, "POST", "/v1/systemone", "X-API-Key: s3cre\r\n", REQ).0, 401);
        // Health never needs a key by default so a client can probe before configuring one.
        assert_eq!(call(addr, "GET", "/health", "", "").0, 200);
    }

    #[test]
    fn health_can_require_key() {
        let addr = start_with(Auth { key: Some("k".into()), health_requires_key: true }, None);
        assert_eq!(call(addr, "GET", "/health", "", "").0, 401);
        assert_eq!(call(addr, "GET", "/health", "X-API-Key: k\r\n", "").0, 200);
    }

    #[test]
    fn serves_https_with_generated_cert() {
        let dir = std::env::temp_dir().join(format!("laya-rs-https-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let material = tls::prepare(&Default::default(), &dir, "127.0.0.1".parse().unwrap()).unwrap();
        let addr = start_with(Auth { key: Some("k".into()), ..Default::default() }, Some(material));
        let client = reqwest::blocking::Client::builder()
            .danger_accept_invalid_certs(true)
            .build()
            .unwrap();
        let url = format!("https://{addr}/v1/systemone");
        let res = client.post(&url).header("X-API-Key", "k").body(REQ).send().unwrap();
        assert_eq!(res.status(), 200);
        let body: Value = res.json().unwrap();
        assert_eq!(body["answers"]["category"]["choice"], "Dining");
        assert_eq!(client.post(&url).body(REQ).send().unwrap().status(), 401);
        // Plain HTTP on a TLS port is refused rather than answered.
        assert!(reqwest::blocking::get(format!("http://{addr}/health")).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
