# Six Tongues Tokenizer + GeoSeal CLI

> SCBE-AETHERMOORE cryptographic toolkit for conlang tokenization and context-aware sealing.

A **self-contained Python CLI** that implements the core of the SCBE-AETHERMOORE system:

- Six Sacred Tongues bijective tokenization (256 tokens per tongue)
- Cross-tongue translation (KO → AV → DR, etc.)
- Blend / unblend of multi-tongue streams
- GeoSeal: context-aware envelope with HMAC-SHA256 integrity
- Built-in `selftest` for round-trip and integrity checks
- **`explain` subcommand** — free AI navigation via Ollama or HuggingFace

Designed for **secure AI-to-AI messaging**, semantic steganography, and as a playground for post-quantum-ready, context-bound data protection.

**Stdlib only.** No pip dependencies required for core operations.

---

## Features

### Six Tongues Tokenizer

- 6 independent conlang "alphabets" (256 tokens each)
- Byte ↔ token mapping is **bijective** (no collisions, full coverage)
- Human-readable, LLM-friendly token streams
- Deterministic: same input always produces the same output

| Tongue | Name | Domain | φ Weight |
|--------|------|--------|----------|
| KO | Kor'aelin | Control & Orchestration | 1.000 |
| AV | Avali | I/O & Messaging | 1.618 |
| RU | Runethic | Policy & Constraints | 2.618 |
| CA | Cassisivadan | Logic & Computation | 4.236 |
| UM | Umbroth | Security & Privacy | 6.854 |
| DR | Draumric | Types & Structures | 11.09 |

### Cross-Tongue Translation

Re-encode a token stream from one tongue to another without touching the underlying bytes. `KO → AV → DR → UM` preserves the exact payload.

### Blend / Unblend

Interleave multiple tongues according to a pattern (e.g. `KO:2,AV:1,DR:1`). Perfectly reversible.

### GeoSeal

Wraps a payload with a context envelope:

- Spatial tile (HEALpix-style projection from lat/lon)
- Timestamp
- User-supplied tag
- HMAC-SHA256 over the envelope + payload

Any tampering (payload, context, or key) causes verification to fail. Real PQC encryption plugs in underneath via the `GEOSEAL_KEY` environment variable.

### Explain (Free AI Navigation)

New: ask the built-in helper to explain any tongue or operation.

```bash
python aethermoore.py explain --tongue KO
python aethermoore.py explain --what "when should I use blend instead of xlate?"
```

Provider fallback chain:

1. **Ollama local** (`OLLAMA_HOST=http://127.0.0.1:11434`, default model `llama3.2:3b`) — private, free, runs offline
2. **HuggingFace Inference API** (`HF_TOKEN=hf_...`) — free with a token, default model `Qwen/Qwen2.5-7B-Instruct`
3. **Static curated fallback** — always available, zero dependencies

The AI is grounded in a system prompt that describes the real system, so answers stay accurate. Set `HF_MODEL` or `OLLAMA_MODEL` to use a different model.

### Self-Test Mode

Run `python aethermoore.py` with no args to verify:

- Token tables are bijective across all 6 tongues
- Encode/decode round-trips for every tongue
- Cross-tongue translation chain (KO → AV → DR → UM)
- Blend/unblend integrity
- GeoSeal wrap/verify + tamper detection

---

## Installation

Requirements:

- Python 3.9+
- No pip dependencies for core functionality
- Optional: set `HF_TOKEN` or run Ollama for AI explain feature

```bash
git clone https://github.com/issdandavis/six-tongues-geoseal.git
cd six-tongues-geoseal
python aethermoore.py                      # run selftest
```

---

## Quickstart

### 1. Encode bytes into KO tokens

```bash
python aethermoore.py encode --tongue KO --text "hello"
```

Pipe form:

```bash
echo -n "hello" | python aethermoore.py encode --tongue KO
```

### 2. Decode KO tokens back to bytes

```bash
python aethermoore.py decode --tongue KO "ko-... ko-... ko-..."
```

### 3. Cross-translate KO → AV

```bash
python aethermoore.py xlate --src KO --dst AV --text "ko-... ko-..."
```

Underlying bytes stay identical; only the tongue changes.

### 4. Blend a multi-tongue stream

```bash
python aethermoore.py blend --pattern KO:2,AV:1,DR:1 --text "secret"
python aethermoore.py unblend --pattern KO:2,AV:1,DR:1 --text "..."
```

### 5. Wrap a payload in GeoSeal

```bash
python aethermoore.py geoseal-encrypt --lat 48.118 --lon -123.430 --tag "demo" --text "hello"
```

Output is a JSON envelope with spatial tile, timestamp, nonce, tag, payload, and MAC.

```bash
python aethermoore.py geoseal-decrypt --sealed '<json>' --expect-tag demo
```

Set a real key in production:

```bash
export GEOSEAL_KEY="your-strong-secret"
```

### 6. Ask the built-in AI a question

```bash
python aethermoore.py explain --tongue UM
python aethermoore.py explain --what "difference between blend and xlate?"

# Local Ollama
export OLLAMA_HOST=http://127.0.0.1:11434
python aethermoore.py explain --tongue RU

# Or HuggingFace (free with a token)
export HF_TOKEN=hf_your_token_here
python aethermoore.py explain --tongue DR
```

### 7. List all six tongues

```bash
python aethermoore.py tongues
```

### 8. Run the self-test

```bash
python aethermoore.py
# selftest ok
#   6 tongues, 256 tokens each, 1536 total tokens
#   encode/decode/xlate/blend/unblend/geoseal all pass
```

---

## CLI Reference

| Command | Purpose |
|---------|---------|
| `encode --tongue T --text STR` | Encode bytes into a tongue |
| `decode --tongue T STREAM` | Decode a token stream back to bytes |
| `xlate --src A --dst B --text STR` | Cross-tongue translate |
| `blend --pattern P --text STR` | Interleave bytes across tongues |
| `unblend --pattern P --text STR` | Unblend a mixed stream |
| `geoseal-encrypt --lat N --lon N --tag T --text STR` | Wrap in context envelope |
| `geoseal-decrypt --sealed JSON [--expect-tag T]` | Verify and unwrap |
| `tongues` | List all six with weights |
| `explain --tongue T` or `--what "..."` | Ask the built-in AI helper |
| `(no args)` | Run selftest |

---

## Environment variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `GEOSEAL_KEY` | HMAC key for GeoSeal envelope | insecure default (selftest only) |
| `OLLAMA_HOST` | Local Ollama endpoint | `http://127.0.0.1:11434` |
| `OLLAMA_MODEL` | Ollama model for explain | `llama3.2:3b` |
| `HF_TOKEN` | HuggingFace Inference API token | none |
| `HF_MODEL` | HF model for explain | `Qwen/Qwen2.5-7B-Instruct` |

---

## Why build this?

This CLI is the **developer-facing entry point** to SCBE-AETHERMOORE, a patent-pending (USPTO #63/961,403) AI governance framework. The tokenization, blending, and sealing primitives here are the same building blocks used inside the 14-layer security pipeline.

Use cases:

- **Secure AI-to-AI messaging** — encode payloads in a tongue, seal with context, send over any channel
- **Semantic steganography** — blend streams across multiple tongues for plausible deniability
- **Data provenance** — every GeoSeal envelope is context-bound and tamper-evident
- **Learning the framework** — play with the core ideas before pulling in the full pipeline

The full framework lives at [github.com/issdandavis/SCBE-AETHERMOORE](https://github.com/issdandavis/SCBE-AETHERMOORE). The managed solutions (CX Guardrail, ISO 42001 evidence service, AI red team) live at [aethermoore.com](https://aethermoore.com/services.html).

---

## License

MIT
