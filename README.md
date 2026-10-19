# Six Tongues Tokenizer + GeoSeal CLI

> SCBE-AETHERMOORE cryptographic toolkit for conlang tokenization and context-aware sealing.

A **self-contained Python CLI** that implements the core of the SCBE-AETHERMOORE system:

- Six Sacred Tongues bijective tokenization (256 tokens per tongue)
- Cross-tongue translation (KO → AV → DR, etc.)
- Blend / unblend of multi-tongue streams
- Entropy/compressibility diagnostics for a payload and its token surface
- Prediction/compression diagnostics against a traditional next-byte baseline
- Aperiodic relational projection benchmark with parameter-matched controls
- Trit light-class overlay on binary prefix-code space
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
- Byte-preserving: this is an encoding layer, so use `entropy` when you need to
  measure compressibility rather than assume that a different token surface is
  smaller

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

### Entropy / Compressibility

Measure the Shannon byte-entropy lower bound, raw zlib size, token surface size,
and compressed token surface size for the same payload:

```bash
python aethermoore.py entropy --tongue KO --text "hello hello hello"
```

This is the practical tokenizer takeaway from the information-theory framing:
bijective tongue tokens preserve the original bytes; they do not by themselves
compress content. Real compression claims should be measured on a corpus.

### Prediction Compression

Compare a standard causal next-byte predictor against a trit-class projection
using the same metric used by next-token training: cross-entropy as ideal
compression bits. Scores are normalized as `bits/input_byte`, so lower is
better and the byte and trit paths can be compared directly.

```bash
python aethermoore.py predict-compress --max-order 3 --text "hello hello hello"
```

The trit projection maps each byte to six base-3 symbols and scores next-trit
prediction against the traditional byte predictor. It is a local stdlib
diagnostic, not a trained neural network.

### Relational Projection Benchmark

Test the non-self-similar projection-depth idea as a causal next-trit model:

```powershell
Get-Content -Raw README.md | python aethermoore.py projection-benchmark
Get-Content -Raw README.md | python aethermoore.py projection-benchmark --ablations
Get-Content -Raw README.md | python aethermoore.py projection-benchmark --ablations --layered-copies 4
```

The working name is the **quasi-metronomic quadratic stenographic array**:
"quasi" is the aperiodic global stack, "metronomic" is the rational local ray
rhythm, "quadratic" is the all-pairs `O(context^2)` relation array, and
"stenographic" is prediction measured as compression bits.

The model starts with an eight-trit context, applies fixed random spatial
distortion, overlays four rational-angle relation rays at each of three depths,
and advances between depths by an irrational golden-ratio phase. Dyadic
`1, 1/2, 1/4` projections reinforce local self-similar alignments while the
distortion and irrational depth stack keep the overall process aperiodic and
non-self-similar. A prior-depth projection backdrop supplies the self/non-self
alignment channel.

This is math only; it does not render the array. The nonlinear projection array
feeds a trained three-class softmax head. Its held-out cross-entropy is compared
over three paired seeds with:

- a duplicated flat-context next-trit control
- a scrambled-angle/scale projection control
- the proposed aligned projection

All three expose 48 features to the same 147-parameter prediction head: 24 raw
trit-context features plus a matched 24-feature second branch. A
`SUPPORTED` result must beat both controls by more than twice their pooled
standard deviation. Smaller positive results are labeled `UNDERPOWERED`;
failure to beat both controls is `NO_LIFT`.

`--ablations` adds five equal-budget component tests and a separate synthesis
verdict. The synthesis keeps one seeded random local ray array, repeats it
through depth with the irrational golden-ratio rotation, and leaves scales
random. This tests the measured useful part without retaining rational local
rays or dyadic scales merely because they were in the original description.
The component comparisons hold the other geometry choices fixed wherever one
factor is being tested.

`--layered-copies N` tests whether several fixed projection arrays can stack a
weak gain without adding trainable parameters. The default `--layered-mode
probability` gives every view the same 147-parameter head and mixes their class
probabilities. `--layered-mode feature --layered-decay 0.5` instead overlays
the fixed feature branches at weights `1, 1/2, 1/4, ...`. Both receive a
copy-matched scrambled control.

Use `--split-fraction 0.6`, `0.7`, or `0.8` to move the causal boundary without
changing the capped train/test sample budgets. Current evidence spans all three
positions; a projection-seed result at one split is not treated as sufficient.

Use `--projection-gain N` to test the projection-branch strength. The same
gain is applied to aligned and scrambled projections, so it cannot grant the
candidate extra trainable capacity. `--layer-gain N` controls recurrent
backdrop reinforcement across depth, again identically for both arms.
`--quadratic-gain N` adds second-order pair coherence, and
`--reinforcement-gain N` multiplies signal by self-alignment. Every ablation is
applied to both projected arms.

### Trit Overlay

Project a ternary prefix table onto a fixed-depth binary prefix table without
rendering an image. Trit digits map to the three SCBE multipath band classes:
`0=blocker(-1)`, `1=reader(0)`, `2=writer(+1)`.

```bash
python aethermoore.py trit-overlay --trit-depth 2 --binary-depth 4
```

This keeps the compression idea mathematical: ternary intervals are compared
against binary code cells, and their ideal binary cost is `depth * log2(3)`.

### Newton operation tube

Read one byte board as six tongue roads plus an exact Newton
forward-difference face:

```bash
python aethermoore.py streams --hex 0103060a0f
python aethermoore.py streams --text "hello" --words not,run
```

The board length, signed coefficients, canonical balanced-ternary spellings,
polarity, and zero edges round-trip.  This is an algebraic view, not a model
score.  Full high-order coefficients expand ordinary text sharply, so model
experiments should use the bounded causal local-difference surface and matched
controls described in
[`docs/NEWTON_OPERATION_TUBE_AUDIT.md`](docs/NEWTON_OPERATION_TUBE_AUDIT.md).
The CLI caps one board at 512 bytes; window larger payloads with explicit offsets.

### GeoSeal

Wraps a payload with a context envelope:

- Spatial tile (HEALpix-style projection from lat/lon)
- Timestamp
- User-supplied tag
- HMAC-SHA256 over the envelope + payload

Any tampering (payload, context, or key) causes verification to fail. Real PQC encryption plugs in underneath via the `GEOSEAL_KEY` environment variable.

### Polyglot representation (represent + execute-to-verify)

Represent one small concept across **six real languages** — Python, JavaScript,
Rust, Haskell, C++, Julia — and, with `--run`, execute each present toolchain and
show which languages **agree** and which split into **camps**. Representation is
pure text (always works); execution is best-effort and honestly labels any
missing toolchain as `represent-only` / `toolchain-not-found`.

```bash
python aethermoore.py polyglot list
python aethermoore.py polyglot scalar --run       # all agree -> language-independent
python aethermoore.py polyglot neg_intdiv --run    # camp split: floor vs truncate
```

Example (`neg_intdiv`, i.e. `-7 / 2` as integers):

```
  camps (agreement first):
    '-3' : JavaScript, Rust, Julia      (truncate toward zero)
    '-4' : Python, Haskell              (floor)
  -> languages SPLIT into camps: a real semantic difference, surfaced by the box.
```

This is the "creativity inside a verifiable box" idea made concrete: a concept
is proposed once, represented many ways, and **cross-checked by execution** — so
where the languages disagree, the box surfaces a genuine semantic fact rather
than hiding it. The deeper self-verifying op-class router lives in `loom`; the
snippets here are vendored from it and extended with C++.

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
python aethermoore.py entropy --tongue KO --text "hello hello hello"
python aethermoore.py predict-compress --max-order 3 --text "hello hello hello"
Get-Content -Raw README.md | python aethermoore.py projection-benchmark
python aethermoore.py trit-overlay --trit-depth 2 --binary-depth 4
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
#   encode/decode/xlate/blend/unblend/geoseal/prediction/projection all pass
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
| `entropy --tongue T --text STR` | Measure entropy and token-surface overhead |
| `predict-compress --text STR` | Compare byte and trit causal compression scores |
| `projection-benchmark [--text STR]` | Test relational projection against matched controls |
| `trit-overlay` | Overlay trit classes on binary prefix-code space |
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
