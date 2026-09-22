# lmr-rs

Native Rust inference for local models. It currently runs [Laya](https://github.com/NandhaKishorM/laya),
the open-weights (Apache 2.0) System One decision model from Convai Innovations, small GGUF chat
models, and the [bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) cross-encoder,
and is named so it can grow beyond any one engine. It downloads the checkpoint from Hugging Face,
runs it with [candle](https://github.com/huggingface/candle), and serves the same `POST /v1/systemone`
API that typesafe.ai exposes (plus `POST /v1/rerank` for rerankers). No Python. Loopback and keyless
by default; a small TOML config turns it into a daemon with an API key and TLS that other machines
can use.

Built for [myphin](../myphin)'s "Laya (local)" categorization provider, but any client that
speaks the System One request shape can use it.

## Requirements

- Rust 1.98 or newer. `rust-toolchain.toml` pins `1.98.1`, so rustup installs it on the first
  `cargo` command in this directory (candle needs it for its NEON f16 kernels).
- About 850 MB of disk for the default Laya checkpoint, plus ~1.7 GB of RAM while serving (the
  model runs in F32). The multilingual checkpoint is smaller on disk (~678 MB: 644 MB weights
  plus a 34 MB tokenizer) but still needs over a gigabyte of RAM in F32. GGUF chat variants
  are smaller on disk: Qwen3 0.6B Q8_0 is about 640 MB, MiniCPM5 2B Q4_K_M is about 1.5 GB.
  The bge-reranker-v2-m3 checkpoint is 2.3 GB on disk and about as much in RAM (F32).
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
# Fetch a checkpoint into ~/.cache/huggingface/hub (once).
./target/release/lmr-rs models download minicpm5-2b
# or: ./target/release/lmr-rs download --variant minicpm5-2b

# Serve on http://127.0.0.1:8321 (uses a downloaded checkpoint if none is set in config)
./target/release/lmr-rs serve
```

`serve` loads the model first (Laya weights are memory mapped; GGUF is quantized) and then answers:

- `POST /v1/systemone` with `{"state": <text | object>, "questions": {<id>: {...}}}`. The
  response is `{"model", "answers": {<id>: {"type", "choice" | "score" | "noul",
  "probabilities", "confidence", "action"}}, "usage"}`, the same document the Python SDK's
  `agent.system_one()` returns. Every loaded checkpoint (Laya or GGUF) accepts and returns
  this shape, so clients can switch models without changing request or response code. Bad
  input gets `422 {"message": ...}`.
- `POST /v1/chat/completions` is an extra [llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
  OpenAI-compatible chat API for GGUF models (`messages`, `temperature`, `top_p`, `top_k`,
  `max_tokens`, `seed`, `chat_template_kwargs`). Laya rejects it. `GET /v1/models` lists the
  loaded file. `GET /v1/health` is an alias of `/health`.
- `POST /v1/rerank` orders an array by how well each item matches a criterion, on reranker
  checkpoints (`bge-reranker-v2-m3`). See [Reranking](#reranking).
- `GET /health` returns the loaded checkpoint and device (`status` is `"ok"` when ready).
- `GET /` is a browser console (on by default). It always posts a System One document to
  `POST /web/systemone`, gated by `web.password` (or open if the password is empty), not by
  the API key. `web.enabled = false` or `--no-web` hides it.

Each inference request (`/v1/systemone`, `/v1/chat/completions`, `/v1/rerank`, and the `/web/*`
twins) prints
one stderr line with the UTC timestamp, client IP, `api` or `web`, and how long it took (for
example `2026-09-21T03:17:42Z  127.0.0.1  api  41ms`). The question is not logged. When
daemonized, that line goes to the log file with the rest of stderr.

Quick manual check without the server:

```sh
./target/release/lmr-rs ask \
    --state '{"transactionTitle": "STARBUCKS #1234", "direction": "money out"}' \
    --question '{"type":"choice","instructions":"Which category?","criteria":{"Dining":"cafes","Gas":null,"other":null}}'

# GGUF chat (same JSON the HTTP API returns)
./target/release/lmr-rs ask --variant qwen3-0.6b --prompt "Who are you?"

# Rerank (same JSON as POST /v1/rerank); --criteria is an alias of --query
./target/release/lmr-rs ask --variant bge-reranker-v2-m3 --query "what is panda?" \
    --document "hi" --document "The giant panda is a bear species endemic to China."
```

## Configuration

Everything lives in one TOML file. `lmr-rs config init` writes it with comments; every key is
optional and the defaults reproduce the loopback-only behaviour above.

```sh
lmr-rs config init          # writes ~/.config/lmr-rs/config.toml
lmr-rs config show          # prints the effective settings
```

The path is `--config <file>`, else `$LMR_RS_CONFIG`, else
`$XDG_CONFIG_HOME/lmr-rs/config.toml` (`~/.config/lmr-rs/config.toml`).
Older `laya-rs` config paths are not read; copy `config.toml` into the new
directory if you still have one.

```toml
[server]
bind = "127.0.0.1"      # IP to listen on; "0.0.0.0" or "::" for all interfaces
port = 8321
public = false          # must be true to bind anything other than loopback
api_key = ""            # or api_key_env = "LMR_API_KEY" / api_key_file = "/path/to/key"
health_requires_key = false

[tls]
enabled = false
cert = ""               # PEM paths; when both are empty and enabled = true, a
key = ""                # self-signed pair is generated next to this file on first run

[model]
# variant = "minicpm5-2b"   # omit to use a downloaded checkpoint; prompt if several are cached
                            # minicpm5-2b | qwen3-0.6b | english | multilingual | typed-decisions
                            # | bge-reranker-v2-m3
# repo = "convaiinnovations/laya"   # raw HF id or local dir; overrides variant
# subfolder = ""
# filename = ""         # GGUF file; overrides the variant default
# engine = ""           # laya | gguf | rerank for a raw repo; empty infers from the files above
device = "auto"         # auto | cpu | metal | cuda
pack_head = true        # spend unused max_len tokens on option texts
tournament = true       # split choice questions larger than tournament_after
tournament_after = 10   # stay out of the published choice:11+ bucket
choice_min_temperature = 1.0  # floor for a flat 11+ choice; 0 = checkpoint (~0.1)

[daemon]
pid_file = ""           # defaults to <config dir>/lmr-rs.pid when --daemonize is used
log_file = ""           # defaults to <config dir>/lmr-rs.log

[web]
enabled = true          # JSON console at GET /; false or --no-web to hide it
password = ""           # empty = open; set a password to gate the UI
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
| `--web` | `web.enabled` | Serve the browser UI at `/` (the default). |
| `--no-web` | `web.enabled = false` | Hide that UI. |
| `--web-password` | `web.password` | Gate that UI; omit (and leave the config empty) to leave it open. |
| `--model`, `--subfolder`, `--variant`, `--filename`, `--engine`, `--device` | `[model]` | Which checkpoint and where to run it. |
| `--daemonize` | `[daemon]` | Fork into the background (Unix). |

`HF_TOKEN` is honoured by the downloader if you need it; it is never printed.

## Choosing a model

```sh
lmr-rs models
lmr-rs models download minicpm5-2b
lmr-rs models delete qwen3-0.6b
lmr-rs models update english
```

`models` lists the catalog and whether each checkpoint is in the Hugging Face cache.
`download` fetches one (same as `lmr-rs download --variant …`). `delete` removes that
checkpoint's files (a Laya subfolder is dropped without touching the others in the same
repo). `update` force-downloads again, replacing the cached copy.
Omit the name and a TTY picker asks which model: the full catalog for `download`, only
cached checkpoints for `delete` and `update`. Pass a catalog name (or `--variant` /
`--model` / `--filename`) to skip the picker. Local directories are left alone.

`serve` and `ask` do not download a default checkpoint. If `model.variant` / `model.repo`
are unset (and no `--variant` / `--model` flag), they use what is already cached: one
downloaded model is loaded as-is, several bring up a picker, and none is an error
telling you to run `lmr-rs models download`. Set `model.variant` in the config to pin
one and skip the picker (needed for `--daemonize` and service units when more than one
is cached).

| Variant | Engine | What it is |
| --- | --- | --- |
| `minicpm5-2b` | GGUF | [MiniCPM5-2B-GGUF](https://huggingface.co/openbmb/MiniCPM5-2B-GGUF) Q4_K_M (~1.5 GB). Default: the best categoriser of the catalog in our tests. |
| `qwen3-0.6b` | GGUF | [Qwen3-0.6B-GGUF](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF) Q8_0 (~640 MB). Fast, but knows few merchants. |
| `english` | Laya | ModernBERT-large, English (~843 MB). |
| `multilingual` | Laya | mmBERT-base, 100+ languages (~678 MB). |
| `typed-decisions` | Laya | Tuned for typed decision questions (~846 MB). |
| `bge-reranker-v2-m3` | Rerank | [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) multilingual cross-encoder (~2.3 GB). Serves `POST /v1/rerank` only. |

Set `model.variant` in the config, or pass `--variant minicpm5-2b` to `download`, `serve`, or
`ask`. Leave it unset to use a downloaded checkpoint. `model.repo` (or `--model`) takes any
Hugging Face id or a local checkpoint directory instead; `--subfolder` picks a folder inside
a Laya repo, `--filename` picks a GGUF file (for example `MiniCPM5-2B-Q8_0.gguf`), and
`--engine rerank` (or `model.engine`) marks a raw repo as a reranker (the catalog repo is
recognised on its own). Laya and GGUF variants serve `POST /v1/systemone` with the same request
and response; a reranker answers `POST /v1/rerank` instead and returns 422 on `/v1/systemone`.
GGUF models also serve the llama.cpp OpenAI chat API as an extra:

```sh
curl http://127.0.0.1:8321/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{"state":{"subject":"Your invoice is overdue"},"questions":{"is_spam":{"type":"noul","instructions":"Is this spam?"}}}'
```

Thinking mode on the chat path (Qwen3 / MiniCPM5) uses llama.cpp's `chat_template_kwargs`,
or `/think` and `/no_think` in the user text:

```json
{"messages":[{"role":"user","content":"Who are you?"}],"chat_template_kwargs":{"enable_thinking":false}}
```

## Reranking

With `bge-reranker-v2-m3` loaded, `POST /v1/rerank` scores every item of `documents` against
`query` (the criterion) with one cross-encoder pass per pair and returns them best first. The
request and response follow llama.cpp, Jina, and Cohere, so existing rerank clients work:

```sh
curl http://127.0.0.1:8321/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{"query":"what is panda?","top_n":2,"return_documents":true,
       "documents":["hi","The giant panda is a bear species endemic to China."]}'
```

```json
{"model":"BAAI/bge-reranker-v2-m3","object":"list",
 "results":[{"index":1,"relevance_score":0.9949,"document":"The giant panda is a bear species endemic to China."},
            {"index":0,"relevance_score":0.0003,"document":"hi"}],
 "usage":{"prompt_tokens":42,"total_tokens":42}}
```

- `query` (alias `criteria`) is the text every document is judged against. Phrase it like a
  search query; the model rewards passages that actually answer it, and scores drop to ~0 when
  a passage does not mention the subject at all.
- `documents` is any JSON array. Strings are scored as they are; objects with a `text` field
  are scored on that field; anything else is serialized to compact JSON, so `[{"id":7, ...}]`
  works. `return_documents: true` echoes the original value back in each result.
- `relevance_score` is the sigmoid of the cross-encoder logit (what FlagEmbedding's
  `compute_score(normalize=True)` returns), `top_n` keeps the best N, and `index` is the
  position in the request. Pairs longer than the tokenizer's `model_max_length` (8192 for this
  checkpoint) are truncated.
- The web console at `/` switches to a `query` + `documents` editor when a reranker is loaded,
  and `lmr-rs ask --query ... --document ... --document ...` runs the same request offline.

## Exposing it beyond localhost

The server refuses to start in an unsafe shape. Binding a non-loopback address needs
`public = true`, and `public = true` needs an API key. Public without TLS is allowed for
reverse-proxy setups but prints a warning, because the key would cross the network in clear text.

```toml
[server]
bind = "0.0.0.0"
public = true
api_key_file = "/etc/lmr-rs/api.key"

[tls]
enabled = true
```

With `tls.enabled = true` and no `cert`/`key`, the first start writes `cert.pem` and `key.pem`
(mode 600) next to the config and prints the certificate's SHA-256 fingerprint. Clients can pin
that fingerprint, or trust the PEM, or you can point `cert`/`key` at a certificate from
mkcert or Let's Encrypt.

Clients send the key as `Authorization: Bearer <key>` or `X-API-Key: <key>`. The comparison
is constant time. `/health` stays open unless `health_requires_key = true`.

## Web UI

On by default. `lmr-rs serve` serves a JSON console at `/` on the same listener.
`web.enabled = false` or `lmr-rs serve --no-web` hides it. The request is always a System
One document (`state` + `questions`); the verdict is rendered on the right (choice bars,
score, noul). Switching the loaded checkpoint does not change that document. Raw JSON is
one click away.

`web.password` is independent of `server.api_key`. The browser talks to `POST /web/systemone`
and never sees the API key. An empty password means anyone who can reach the server can run
inference from the UI, even if an API key is set for `/v1/*`. Set a password when the
listener is public. Startup prints a warning if the UI is open on a non-loopback bind.

```toml
[web]
enabled = true
password = "desk"
```

The login sets an HttpOnly `lmr_web` cookie (SameSite=Strict; Secure when TLS is on).

## Running as a service

Foreground is the default and is what launchd and systemd want:

```sh
lmr-rs config init
lmr-rs service install            # per user: ~/Library/LaunchAgents or ~/.config/systemd/user
lmr-rs service install --system   # /Library/LaunchDaemons or /etc/systemd/system (as root)
```

It writes the unit, points it at the current binary and config, and prints the `launchctl` or
`systemctl` command to enable it. Templates are in `contrib/`.

Without a service manager, `lmr-rs serve --daemonize` forks after binding the port and writes
`lmr-rs.pid` and `lmr-rs.log` next to the config (`[daemon]` overrides both). Stop it with
`kill $(cat lmr-rs.pid)`. The first download's progress goes to the log, so run
`lmr-rs models download` once in the foreground before daemonizing. If more than one
checkpoint is cached, set `model.variant` in the config so the daemon does not need a picker.

## Pairing with myphin

1. Start `lmr-rs serve` (leave it running; it uses no network after the model is on disk).
2. In myphin, Setup → AI → provider "Laya (local)". Leave the key blank unless you configured
   an API key; myphin sends it as a bearer token.
3. "Debug a transaction" shows the loopback request and answer in the AI trace.

Laya is a small classifier, not a payee encyclopedia. Two things used to make a 20-category
list look hopeless: descriptions were chopped to a handful of tokens (so rewriting them did
nothing), and the published `choice:11+` temperature of ~0.1 turned a weak guess into 100%
confidence (`COSTCO` → Fee, `STARBUCKS` → Investment › Transaction). `lmr-rs` now spends
unused room in the 512-token window on the option texts, and any `choice` with more than ten
options is answered as a tournament (group, then member) so that bucket is never used. Those
are `model.pack_head`, `model.tournament`, `model.tournament_after`, and
`model.choice_min_temperature` in the config. Set `pack_head` / `tournament` to `false` and
`choice_min_temperature` to `0` for Python-SDK packing and calibration.

Still keep descriptions short and distinct, do not put the same cue in two categories
(`bars` in Food and Alcohol, `gas` in Gas and Utilities), and put the distinctive words
first (`grocery store, warehouse club` not a long restaurant list that ends with grocery).
The model does not know `COSTCO` is a warehouse club unless a description says so.

## Test

```sh
cargo test                       # unit tests + tokenizer/sequence parity (no checkpoint needed)
cargo test -- --include-ignored  # also compares answers with the Python SDK; needs `lmr-rs models download english`
```

Unit tests cover the config rules, certificate generation, and the HTTP and HTTPS paths with a
stub model (the TLS test uses `reqwest` against a self-signed cert).

The parity fixtures in `tests/fixtures/` were recorded from the Python SDK by
`scripts/make_fixtures.py` (`pip install laya`, then run it). Regenerate them if the upstream
checkpoint changes.

## Releasing

Pushing a `v*` tag (or running the Release workflow with a bump type) builds,
tests, and publishes binaries for macOS Apple Silicon (Metal) and Linux
(x86_64 and ARM64, CPU) with a SHA-256 checksum file. The Linux ARM64 build
needs ARMv8.2 FP16 (Graviton2+, Ampere), which candle's gemm kernels require.

Linux releases are CPU-only. CUDA needs a local NVIDIA toolkit, so build that
yourself with `--features cuda`.

The macOS binary is codesigned, and notarized, when the Apple secrets are
configured on the repository; without them the release still goes out, just
unsigned. See [docs/macos-signing.md](docs/macos-signing.md) for which
certificate to get and which secrets to set.

For the first release, push `v0.1.0` (the version already in `Cargo.toml`) rather
than bumping from a missing tag.

## How it works

- `src/hub.rs` fetches a Laya checkpoint (`rl_agent_config.json`, `encoder/config.json`, the
  tokenizer files, `model.safetensors`) or a GGUF file plus `tokenizer.json` from the matching
  base repo, through `hf-hub`. `lmr-rs models` reports what is cached; `models download`,
  `models delete`, and `models update` fetch, remove, or replace those files.
- `src/sequence.rs` is a port of `laya/common.py`: `[CLS] <type> question: <instructions> [SEP]
  ([MASK] option)* [SEP] state [SEP]`, including Python's `json.dumps` formatting of structured
  state and criteria, which changes the tokens and therefore the answer. When the state is
  short, unused tokens in `max_len` go to the option texts instead of sitting empty (Python
  keeps a fixed 192-token head, which chops descriptions once there are ~15 options).
- `src/tournament.rs` splits a `choice` with more than ten options into a group question and
  a member question, so we stay in the `choice:6-10` calibration bucket. Nested names that
  share a `›` parent stay in the same group.
- `src/model.rs` uses candle's ModernBERT encoder and re-implements Laya's head: a type
  embedding, two pre-LN transformer layers (ReLU), a scorer read at each option's `[MASK]`, and
  the small `act_head`.
- `src/gguf.rs` loads Qwen3 and Llama-architecture GGUF files through candle and applies the
  ChatML template those models ship. System One questions list the options by name and score
  each name's log-likelihood as the model's reply (only as far as its tokens are distinct from
  the other options, with end-of-turn appended when one option is a prefix of another, as in
  `Investment` vs `Investment › Fee`), then softmax over the options. Options that share a
  token path share forward passes on the KV cache, so a 22 way question usually costs one
  full pass plus a few single tokens. The same options are also scored against a content-free
  state (every value `N/A`, cached per question) and half of that prior is divided out, so a
  small model's standing preference for one name (`Fee`, `other`) does not decide every
  transaction. Small models still lean on any "pick other if none fits" wording in
  `instructions`, so leave that out and let `other` be a plain option.
- `src/rerank.rs` loads an XLM-RoBERTa sequence classifier (candle's `xlm_roberta`) and scores
  `<s> query </s></s> document </s>` one pair at a time for `POST /v1/rerank`.
- `src/decide.rs` applies the per bucket calibration temperature from the checkpoint config and
  the entropy based confidence, then shapes the JSON. That shape is shared by every engine.
- `src/settings.rs` is the TOML config and the startup safety rules; `src/tls.rs` finds or
  generates the certificate; `src/server.rs` is the axum/rustls listener; `src/web.rs` is the
  browser UI.
