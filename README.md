# laya-rs

Native Rust inference for [Laya](https://github.com/NandhaKishorM/laya), the open-weights
(Apache 2.0) System One decision model from Convai Innovations. It downloads the checkpoint from
Hugging Face, runs it with [candle](https://github.com/huggingface/candle), and serves the same
`POST /v1/systemone` API that typesafe.ai exposes. No Python. Loopback and keyless by default;
a small TOML config turns it into a daemon with an API key and TLS that other machines can use.

Built for [myphin](../myphin)'s "Laya (local)" categorization provider, but any client that
speaks the System One request shape can use it.

## Requirements

- Rust 1.98 or newer. `rust-toolchain.toml` pins `1.98.1`, so rustup installs it on the first
  `cargo` command in this directory (candle needs it for its NEON f16 kernels).
- About 850 MB of disk for the default checkpoint, plus ~1.7 GB of RAM while serving (the model
  runs in F32).
- macOS: build with `--features metal` for the GPU. Linux with an NVIDIA GPU: `--features cuda`.
  CPU works everywhere; a ModernBERT-large request takes roughly a second on an M-series CPU and
  ~40 ms on Metal.

## Build

```sh
cargo build --release --features metal     # macOS
cargo build --release                      # CPU only
```

## Run

```sh
# Fetch the checkpoint into ~/.cache/huggingface/hub (once, ~843 MB).
./target/release/laya-rs download

# Serve on http://127.0.0.1:8321
./target/release/laya-rs serve
```

`serve` loads the model first (fast, the weights are memory mapped) and then answers:

- `POST /v1/systemone` with `{"state": <text | object>, "questions": {<id>: {...}}}`. The
  response is `{"model", "answers": {<id>: {"type", "choice" | "score" | "noul",
  "probabilities", "confidence", "action"}}, "usage"}`, the same document the Python SDK's
  `agent.system_one()` returns. Bad input gets `422 {"message": ...}`.
- `GET /health` returns the loaded checkpoint and device.

Quick manual check without the server:

```sh
./target/release/laya-rs ask \
  --state '{"transactionTitle": "STARBUCKS #1234", "direction": "money out"}' \
  --question '{"type":"choice","instructions":"Which category?","criteria":{"Dining":"cafes","Gas":null,"other":null}}'
```

## Configuration

Everything lives in one TOML file. `laya-rs config init` writes it with comments; every key is
optional and the defaults reproduce the loopback-only behaviour above.

```sh
laya-rs config init          # writes ~/.config/laya-rs/config.toml
laya-rs config show          # prints the effective settings
```

The path is `--config <file>`, else `$LAYA_RS_CONFIG`, else
`$XDG_CONFIG_HOME/laya-rs/config.toml` (`~/.config/laya-rs/config.toml`).

```toml
[server]
bind = "127.0.0.1"      # IP to listen on; "0.0.0.0" or "::" for all interfaces
port = 8321
public = false          # must be true to bind anything other than loopback
api_key = ""            # or api_key_env = "LAYA_API_KEY" / api_key_file = "/path/to/key"
health_requires_key = false

[tls]
enabled = false
cert = ""               # PEM paths; when both are empty and enabled = true, a
key = ""                # self-signed pair is generated next to this file on first run

[model]
variant = "english"     # english | multilingual | typed-decisions
# repo = "convaiinnovations/laya"   # raw HF id or local dir; overrides variant
# subfolder = ""
device = "auto"         # auto | cpu | metal | cuda

[daemon]
pid_file = ""           # defaults to <config dir>/laya-rs.pid when --daemonize is used
log_file = ""           # defaults to <config dir>/laya-rs.log
```

Flags on `serve` override the file for one run:

| Flag | Config key | Meaning |
| --- | --- | --- |
| `--bind` | `server.bind` | IP to listen on. |
| `--port` | `server.port` | Port (default 8321). |
| `--public` | `server.public` | Allow a non-loopback bind. |
| `--api-key` (alias `--token`) | `server.api_key` | Require the key on `/v1/systemone`. |
| `--health-requires-key` | `server.health_requires_key` | Gate `/health` too. |
| `--tls`, `--cert`, `--key` | `[tls]` | Serve HTTPS; `--cert`/`--key` are PEM paths. |
| `--model`, `--subfolder`, `--variant`, `--device` | `[model]` | Which checkpoint and where to run it. |
| `--daemonize` | `[daemon]` | Fork into the background (Unix). |

`HF_TOKEN` is honoured by the downloader if you need it; it is never printed.

## Choosing a model

```sh
laya-rs models
```

| Variant | What it is |
| --- | --- |
| `english` | ModernBERT-large, English. The default (~843 MB). |
| `multilingual` | mmBERT-base, 100+ languages. |
| `typed-decisions` | Tuned for typed decision questions. |

Set `model.variant` in the config, or pass `--variant multilingual` to `download`, `serve`, or
`ask`. `model.repo` (or `--model`) takes any Hugging Face id or a local checkpoint directory
instead; `--subfolder` picks a folder inside it. The engine runs Laya checkpoints only (a
ModernBERT encoder plus Laya's decision head), not general language models.

## Exposing it beyond localhost

The server refuses to start in an unsafe shape. Binding a non-loopback address needs
`public = true`, and `public = true` needs an API key. Public without TLS is allowed for
reverse-proxy setups but prints a warning, because the key would cross the network in clear text.

```toml
[server]
bind = "0.0.0.0"
public = true
api_key_file = "/etc/laya-rs/api.key"

[tls]
enabled = true
```

With `tls.enabled = true` and no `cert`/`key`, the first start writes `cert.pem` and `key.pem`
(mode 600) next to the config and prints the certificate's SHA-256 fingerprint. Clients can pin
that fingerprint, or trust the PEM, or you can point `cert`/`key` at a certificate from
mkcert or Let's Encrypt.

Clients send the key as `Authorization: Bearer <key>` or `X-API-Key: <key>`. The comparison
is constant time. `/health` stays open unless `health_requires_key = true`.

## Running as a service

Foreground is the default and is what launchd and systemd want:

```sh
laya-rs config init
laya-rs service install            # per user: ~/Library/LaunchAgents or ~/.config/systemd/user
laya-rs service install --system   # /Library/LaunchDaemons or /etc/systemd/system (as root)
```

It writes the unit, points it at the current binary and config, and prints the `launchctl` or
`systemctl` command to enable it. Templates are in `contrib/`.

Without a service manager, `laya-rs serve --daemonize` forks after binding the port and writes
`laya-rs.pid` and `laya-rs.log` next to the config (`[daemon]` overrides both). Stop it with
`kill $(cat laya-rs.pid)`. The first download's progress goes to the log, so run
`laya-rs download` once in the foreground before daemonizing.

## Pairing with myphin

1. Start `laya-rs serve` (leave it running; it uses no network after the download).
2. In myphin, Setup → AI → provider "Laya (local)". Leave the key blank unless you configured
   an API key; myphin sends it as a bearer token.
3. "Debug a transaction" shows the loopback request and answer in the AI trace.

## Test

```sh
cargo test                       # unit tests + tokenizer/sequence parity (no checkpoint needed)
cargo test -- --include-ignored  # also compares answers with the Python SDK; needs `laya-rs download`
```

Unit tests cover the config rules, certificate generation, and the HTTP and HTTPS paths with a
stub model (the TLS test uses `reqwest` against a self-signed cert).

The parity fixtures in `tests/fixtures/` were recorded from the Python SDK by
`scripts/make_fixtures.py` (`pip install laya`, then run it). Regenerate them if the upstream
checkpoint changes.

## How it works

- `src/hub.rs` fetches `rl_agent_config.json`, `encoder/config.json`, the tokenizer files, and
  `model.safetensors` through `hf-hub`, sharing the cache with the Python SDK.
- `src/sequence.rs` is a port of `laya/common.py`: `[CLS] <type> question: <instructions> [SEP]
  ([MASK] option)* [SEP] state [SEP]`, including Python's `json.dumps` formatting of structured
  state and criteria, which changes the tokens and therefore the answer.
- `src/model.rs` uses candle's ModernBERT encoder and re-implements Laya's head: a type
  embedding, two pre-LN transformer layers (ReLU), a scorer read at each option's `[MASK]`, and
  the small `act_head`.
- `src/decide.rs` applies the per bucket calibration temperature from the checkpoint config and
  the entropy based confidence, then shapes the JSON.
- `src/settings.rs` is the TOML config and the startup safety rules; `src/tls.rs` finds or
  generates the certificate; `src/server.rs` is the axum/rustls listener.
