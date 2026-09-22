//! Browser UI: static files, a password cookie, and the `/web/*` routes.
//!
//! The page is optional (`WebConfig::enabled`). When a password is set the cookie `lmr_web`
//! is a SHA-256 token of that password; empty password means the UI is open. The API key is
//! never sent to the browser: `POST /web/systemone` shares the decider with `/v1/systemone`
//! but is gated only by this cookie. The UI always posts System One documents, matching `/v1/systemone`.

use axum::http::header::{self, HeaderMap, HeaderValue};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;

const COOKIE_NAME: &str = "lmr_web";
const COOKIE_MAX_AGE: &str = "604800";

const INDEX_HTML: &str = include_str!("../web/index.html");
const APP_CSS: &str = include_str!("../web/app.css");
const APP_JS: &str = include_str!("../web/app.js");

/// How the listener should treat the browser UI.
#[derive(Debug, Clone, Default)]
pub struct WebConfig {
    pub enabled: bool,
    /// Expected cookie token, or `None` when the UI is open.
    token: Option<String>,
    pub secure_cookie: bool,
}

impl WebConfig {
    /// `password` empty or missing means the UI is open (anyone who can reach it may query).
    pub fn new(enabled: bool, password: Option<String>, secure_cookie: bool) -> Self {
        let token = password
            .map(|p| p.trim().to_string())
            .filter(|p| !p.is_empty())
            .map(|p| session_token(&p));
        Self {
            enabled,
            token,
            secure_cookie,
        }
    }

    /// True when a password is configured.
    pub fn is_locked(&self) -> bool {
        self.token.is_some()
    }

    /// Accept an open UI, or a cookie that matches the password token.
    pub fn allows(&self, headers: &HeaderMap) -> bool {
        let Some(expected) = &self.token else {
            return true;
        };
        cookie_value(headers, COOKIE_NAME)
            .map(|given| given.as_bytes().ct_eq(expected.as_bytes()).into())
            .unwrap_or(false)
    }

    /// Constant-time check of a submitted password against the configured one.
    pub fn password_matches(&self, password: &str) -> bool {
        let Some(expected) = &self.token else {
            return true;
        };
        session_token(password)
            .as_bytes()
            .ct_eq(expected.as_bytes())
            .into()
    }

    /// `Set-Cookie` that unlocks the UI for a week.
    pub fn login_cookie(&self) -> HeaderValue {
        let token = self.token.as_deref().unwrap_or("");
        cookie_header(token, COOKIE_MAX_AGE, self.secure_cookie)
    }

    /// `Set-Cookie` that expires the session immediately.
    pub fn logout_cookie(&self) -> HeaderValue {
        cookie_header("", "0", self.secure_cookie)
    }
}

/// SHA-256 hex of a domain-separated password. The cookie stores this, not the password.
fn session_token(password: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"lmr-rs-web\0");
    hasher.update(password.as_bytes());
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn cookie_header(value: &str, max_age: &str, secure: bool) -> HeaderValue {
    let mut s =
        format!("{COOKIE_NAME}={value}; HttpOnly; SameSite=Strict; Path=/; Max-Age={max_age}");
    if secure {
        s.push_str("; Secure");
    }
    HeaderValue::from_str(&s).unwrap_or_else(|_| HeaderValue::from_static("lmr_web="))
}

/// First `name=` value in the Cookie header(s).
fn cookie_value<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    let raw = headers.get(header::COOKIE)?.to_str().ok()?;
    for part in raw.split(';') {
        let part = part.trim();
        let Some(rest) = part.strip_prefix(name) else {
            continue;
        };
        let Some(value) = rest.strip_prefix('=') else {
            continue;
        };
        return Some(value);
    }
    None
}

/// `GET /` HTML document.
pub fn index() -> Response {
    html_response("text/html; charset=utf-8", INDEX_HTML)
}

/// `GET /web/app.css`.
pub fn css() -> Response {
    html_response("text/css; charset=utf-8", APP_CSS)
}

/// `GET /web/app.js`.
pub fn js() -> Response {
    html_response("text/javascript; charset=utf-8", APP_JS)
}

fn html_response(content_type: &'static str, body: &'static str) -> Response {
    (
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, content_type),
            (header::CACHE_CONTROL, "no-cache"),
        ],
        body,
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_ui_allows_everyone() {
        let web = WebConfig::new(true, None, false);
        assert!(!web.is_locked());
        assert!(web.allows(&HeaderMap::new()));
        assert!(web.password_matches("anything"));
    }

    #[test]
    fn locked_ui_needs_matching_cookie() {
        let web = WebConfig::new(true, Some("desk".into()), false);
        assert!(web.is_locked());
        assert!(!web.allows(&HeaderMap::new()));
        assert!(web.password_matches("desk"));
        assert!(!web.password_matches("Desk"));
        let mut headers = HeaderMap::new();
        headers.insert(
            header::COOKIE,
            HeaderValue::from_str(&format!("{COOKIE_NAME}={}", session_token("desk"))).unwrap(),
        );
        assert!(web.allows(&headers));
        headers.insert(header::COOKIE, HeaderValue::from_static("lmr_web=deadbeef"));
        assert!(!web.allows(&headers));
    }

    #[test]
    fn blank_password_is_open() {
        let web = WebConfig::new(true, Some("  ".into()), false);
        assert!(!web.is_locked());
    }

    #[test]
    fn branding_is_lmr_rs() {
        assert_eq!(COOKIE_NAME, "lmr_web");
        assert!(INDEX_HTML.contains("<title>lmr-rs</title>"));
        assert!(INDEX_HTML.contains(r#"class="mark">lmr</p>"#));
        let cookie = cookie_header("tok", "1", false);
        assert!(cookie.to_str().unwrap().starts_with("lmr_web="));
    }

    #[test]
    fn example_chips_are_spam_urgency_and_a_hidden_rerank() {
        assert!(INDEX_HTML.contains(r#"data-example="spam""#));
        assert!(INDEX_HTML.contains(r#"data-example="urgency""#));
        assert!(INDEX_HTML.contains(r#"data-example="rerank" hidden>Support<"#));
        assert!(!INDEX_HTML.contains("Descale"));
        assert!(APP_JS.contains(r#"query: "I forgot my password and can't log in""#));
        assert!(!APP_JS.contains("espresso"));
        assert!(!INDEX_HTML.contains(r#"id="chips-gguf""#));
        assert!(!INDEX_HTML.contains(r#"data-example="hello""#));
        assert!(!INDEX_HTML.contains(r#"data-example="think""#));
        assert!(!INDEX_HTML.contains(r#"data-example="category""#));
        assert!(APP_JS.contains(r#"loadExample(mode === "rerank" ? "rerank" : "spam")"#));
        assert!(APP_JS.contains("/web/systemone"));
        assert!(APP_JS.contains("/web/rerank"));
        assert!(APP_JS.contains(r#"info.body.engine === "rerank""#));
        assert!(!APP_JS.contains("/web/chat/completions"));
        assert!(APP_CSS.contains("[hidden]"));
        assert!(!APP_JS.contains(r#"loadExample("category")"#));
        assert!(!APP_JS.contains("transactionTitle"));
    }
}
