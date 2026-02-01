# Six Tongues Tokenizer + GeoSeal CLI
> SCBE-AETHERMOORE cryptographic toolkit for conlang tokenization and context-aware sealing.

This is a **self-contained Python CLI** that implements the core of the SCBE‑AETHERMOORE system:

- Six Sacred Tongues bijective tokenization (256 tokens per tongue)
- Cross-tongue translation (KO→AV→DR, etc.)
- Blend / unblend of multi-tongue streams
- GeoSeal: context-aware encryption stub (Healpix/Morton projection + PQC-ready structure)
- Built-in `selftest` for round-trip and integrity checks

It's designed for **secure AI-to-AI messaging**, semantic steganography, and as a playground for post‑quantum–ready, context-bound cryptography.

---

## Features

- **Six Tongues Tokenizer**
  - 6 independent conlang "alphabets" (256 tokens each)
  - Byte ↔ token mapping is **bijective** (no collisions, full coverage)
  - Human‑readable, LLM‑friendly token streams

- **Cross-Tongue Translation**
  - Re-encode a token stream from one tongue to another without touching the underlying bytes.
  - Example: KO → AV → DR, preserving exact payload.

- **Blend / Unblend**
  - Interleave multiple tongues according to a pattern (e.g. `KO:2,AV:1,DR:1`)
  - Perfectly reversible; preserves byte‑exact data.

- **GeoSeal (Context-Aware Encryption Stub)**
  - Projects lat/long and context into a structured "seal" (e.g. Healpix/Morton style)
  - Wraps payloads with context metadata that can later be checked before decryption
  - PQC hooks for Kyber / Dilithium integration (currently stubbed for portability)

- **CLI-First Design**
  - Subcommands: `encode`, `decode`, `xlate`, `blend`, `unblend`, `geoseal-encrypt`, `geoseal-decrypt`
  - Works with pipes, files, or direct arguments.

- **Self-Test Mode**
  - Run `python aethermoore.py` with no args to execute:
    - Encode → decode round-trips
    - Cross-tongue translation sanity checks
    - Blend / unblend integrity
    - GeoSeal wrap / unwrap checks

---

## Installation

Requirements:

- Python 3.9+
- `numpy`

Quick setup:

```bash
python3 -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate
pip install numpy
```

Clone or download this repo, then place `aethermoore.py` at the project root.

---

## Quickstart

All commands assume you're in the same directory as `aethermoore.py`.

### 1. Encode bytes into KO tokens

```bash
echo -n "hello" | python aethermoore.py encode --tongue KO
```

Example output:

```text
ko'ka ke'ne ki'ni ko'no ku'nu
```

### 2. Decode KO tokens back to bytes

```bash
python aethermoore.py decode --tongue KO "ko'ka ke'ne ki'ni ko'no ku'nu"
```

Output:

```text
hello
```

### 3. Cross-translate KO → AV

```bash
echo "ko'ka ke'ne ki'ni ko'no ku'nu" \
  | python aethermoore.py xlate --src KO --dst AV
```

This keeps the underlying bytes identical, only changing the tongue.

### 4. Blend multi-tongue stream

Pattern: KO:2, AV:1, DR:1

```bash
echo -n "secret" \
  | python aethermoore.py blend --pattern KO:2,AV:1,DR:1
```

You can later `unblend` back into the original byte stream.

### 5. Run full self-test

```bash
python aethermoore.py
```

You should see a final status similar to:

```text
selftest ok
```

---

## CLI Reference

General form:

```bash
python aethermoore.py <command> [options] [args...]
```

### encode

Encode stdin or a literal string into a Sacred Tongue.

```bash
python aethermoore.py encode --tongue KO
python aethermoore.py encode --tongue AV --text "hello world"
```

Options:
- `--tongue {KO,AV,RU,CA,UM,DR}` – which tongue to use
- `--text TEXT` – encode this literal instead of stdin (optional)

### decode

Decode a token stream back to raw bytes.

```bash
python aethermoore.py decode --tongue KO "ko'ka ke'ne ..."
```

Options:
- `--tongue {KO,AV,RU,CA,UM,DR}`

### xlate

Translate between tongues without changing the payload.

```bash
python aethermoore.py xlate --src KO --dst AV
```

Reads tokens from stdin if no positional string is given.

Options:
- `--src {KO,AV,RU,CA,UM,DR}`
- `--dst {KO,AV,RU,CA,UM,DR}`

### blend

Blend multiple tongues into one stream according to a pattern.

```bash
python aethermoore.py blend --pattern KO:2,AV:1,DR:1
```

Options:
- `--pattern PATTERN` – comma-separated `TONGUE:N` entries (e.g. `KO:2,AV:1`)

### unblend

Reverse a previously blended stream.

```bash
python aethermoore.py unblend --pattern KO:2,AV:1,DR:1
```

Uses the same pattern that was used for `blend`.

### geoseal-encrypt

Wrap data with a context-aware "seal" (location, time, etc.). Current version uses HMAC/SHA-256 + structured metadata; Kyber/Dilithium hooks are stubbed in for future PQC integration.

Example:

```bash
echo -n "classified" | python aethermoore.py geoseal-encrypt \
  --lat 48.118 --lon -123.430 --tag "demo"
```

### geoseal-decrypt

Verify and unwrap a GeoSeal envelope.

```bash
python aethermoore.py geoseal-decrypt --expect-tag "demo"
```

---

## Use Cases

- **Secure AI agent messaging**  
  Encode payloads into Sacred Tongue tokens, optionally GeoSeal them with context (location, model ID, time) before sending between agents.

- **Semantic steganography**  
  Hide arbitrary bytes in conlang-like tokens that look like exotic text instead of hex/base64.

- **Post-quantum–ready experimentation**  
  The GeoSeal layer is structured around PQC slots (Kyber/Dilithium), so you can later plug in real ML‑KEM / ML‑DSA primitives without changing the interface.

- **Game / worldbuilding tools**  
  Generate consistent, reversible in‑universe "languages" with cryptographic semantics.

---

## Security Model & Caveats

This is **prototype cryptographic software**. Treat it as a research and integration tool, not as a fully audited system.

- Tokenization and blending are **exact, reversible transforms** on bytes. They do not provide confidentiality by themselves.
- GeoSeal currently uses standard primitives (HMAC/SHA-256, etc.) and placeholder logic for PQC. Do not claim "military‑grade post‑quantum" security until you wire in vetted Kyber/Dilithium libraries and get an audit.
- Do not roll your own production crypto deployments without a cryptographer reviewing the design.

That said, the design is intentionally:
- deterministic
- testable (`selftest`)
- easy to extend with standard crypto libraries.

---

## Roadmap

Planned improvements:

- Plug-in interface for real ML‑KEM (Kyber) and ML‑DSA (Dilithium)
- SCBE‑AETHERMOORE hyperbolic context binding (GeoSeal ←→ Poincaré embeddings)
- JSON/HTTP API wrapper for use in agent frameworks (LangChain, Semantic Kernel, etc.)
- More tongue sets and user-defined alphabets
- Formal spec + reference paper (arXiv)

---

## License

MIT. See `LICENSE` for details.

---

## Author

**Issac Davis** – Port Angeles, Washington, USA  
SCBE‑AETHERMOORE / Six Sacred Tongues / GeoSeal
