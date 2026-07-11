#!/usr/bin/env python3
"""
SCBE-AETHERMOORE Six Tongues Tokenizer + GeoSeal CLI

A self-contained Python CLI for:
- Six Sacred Tongues bijective tokenization (256 tokens per tongue)
- Cross-tongue translation (KO -> AV -> DR, etc.)
- Blend / unblend of multi-tongue streams
- GeoSeal: context-aware sealing with HMAC integrity
- 'explain' subcommand: free AI navigation via HuggingFace or Ollama
- Built-in selftest

Stdlib only (no required pip deps). Optional: HF_TOKEN for the explain
feature, or a local Ollama server for private explanations.

Usage:
    python aethermoore.py                                  # selftest
    python aethermoore.py encode --tongue KO --text "hello"
    python aethermoore.py decode --tongue KO "ko-ka ke-ne ki-ni ko-no ku-nu"
    python aethermoore.py xlate --src KO --dst AV --text "ko-ka ke-ne"
    python aethermoore.py blend --pattern KO:2,AV:1,DR:1 --text "secret"
    python aethermoore.py unblend --pattern KO:2,AV:1,DR:1 --text "..."
    python aethermoore.py geoseal-encrypt --lat 48.118 --lon -123.430 --text "msg"
    python aethermoore.py geoseal-decrypt --sealed '<json>'
    python aethermoore.py explain --tongue KO
    python aethermoore.py explain --what "blend pattern KO:2,AV:1,DR:1"

Author: Issac Davis
License: MIT
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import random
import secrets
import sys
import time
import urllib.error
import urllib.request
from typing import Any

# ---------------------------------------------------------------------------
# Six Sacred Tongues
# ---------------------------------------------------------------------------

TONGUE_META = {
    "KO": {"name": "Kor'aelin",     "domain": "Control & Orchestration", "weight": 1.000, "seed": 0xC01},
    "AV": {"name": "Avali",         "domain": "I/O & Messaging",          "weight": 1.618, "seed": 0xA71},
    "RU": {"name": "Runethic",      "domain": "Policy & Constraints",     "weight": 2.618, "seed": 0xB03},
    "CA": {"name": "Cassisivadan",  "domain": "Logic & Computation",      "weight": 4.236, "seed": 0xC0DE},
    "UM": {"name": "Umbroth",       "domain": "Security & Privacy",       "weight": 6.854, "seed": 0xDEAD},
    "DR": {"name": "Draumric",      "domain": "Types & Structures",       "weight": 11.09, "seed": 0xFACE},
}
TONGUES = list(TONGUE_META.keys())

# Phonetic primitives used to build pronounceable tokens
CONSONANTS = ["k", "n", "r", "s", "t", "m", "l", "v", "z", "ch", "th", "sh", "br", "dr", "gl", "ph"]
VOWELS = ["a", "e", "i", "o", "u", "ae", "ei", "ou"]


def _build_token_table(tongue: str) -> list[str]:
    """Deterministically build 256 unique tokens for a tongue."""
    meta = TONGUE_META[tongue]
    rng = random.Random(meta["seed"])
    prefix = tongue.lower()
    pool: set[str] = set()
    ordered: list[str] = []
    # Generate enough unique pronounceable strings
    for c1 in CONSONANTS:
        for v1 in VOWELS:
            for c2 in CONSONANTS:
                for v2 in VOWELS:
                    tok = f"{prefix}-{c1}{v1}{c2}{v2}"
                    if tok in pool:
                        continue
                    pool.add(tok)
                    ordered.append(tok)
    # Shuffle deterministically with the tongue's seed
    rng.shuffle(ordered)
    return ordered[:256]


_TOKEN_TABLES: dict[str, list[str]] = {}
_INV_TABLES: dict[str, dict[str, int]] = {}


def _tables(tongue: str) -> tuple[list[str], dict[str, int]]:
    if tongue not in _TOKEN_TABLES:
        table = _build_token_table(tongue)
        if len(table) != 256 or len(set(table)) != 256:
            raise RuntimeError(f"Non-bijective token table for {tongue}")
        _TOKEN_TABLES[tongue] = table
        _INV_TABLES[tongue] = {tok: i for i, tok in enumerate(table)}
    return _TOKEN_TABLES[tongue], _INV_TABLES[tongue]


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------


def encode(data: bytes, tongue: str) -> str:
    tongue = tongue.upper()
    if tongue not in TONGUES:
        raise ValueError(f"Unknown tongue: {tongue}. Pick one of {TONGUES}")
    table, _ = _tables(tongue)
    return " ".join(table[b] for b in data)


def decode(stream: str, tongue: str) -> bytes:
    tongue = tongue.upper()
    if tongue not in TONGUES:
        raise ValueError(f"Unknown tongue: {tongue}")
    _, inv = _tables(tongue)
    out = bytearray()
    for tok in stream.strip().split():
        if tok not in inv:
            raise ValueError(f"Invalid {tongue} token: {tok!r}")
        out.append(inv[tok])
    return bytes(out)


def xlate(stream: str, src: str, dst: str) -> str:
    """Translate a token stream from src tongue to dst tongue byte-for-byte."""
    return encode(decode(stream, src), dst)


def _parse_pattern(pattern: str) -> list[tuple[str, int]]:
    parts = [p.strip() for p in pattern.split(",") if p.strip()]
    out = []
    for p in parts:
        if ":" not in p:
            raise ValueError(f"Bad pattern segment: {p}")
        name, count = p.split(":", 1)
        name = name.strip().upper()
        if name not in TONGUES:
            raise ValueError(f"Unknown tongue in pattern: {name}")
        out.append((name, int(count)))
    if not out:
        raise ValueError("Empty pattern")
    return out


def blend(data: bytes, pattern: str) -> str:
    """Interleave a byte stream across multiple tongues per the pattern."""
    segs = _parse_pattern(pattern)
    total = sum(c for _, c in segs)
    tokens: list[str] = []
    for i, byte in enumerate(data):
        # Which tongue owns position (i mod total)?
        slot = i % total
        cursor = 0
        chosen = segs[0][0]
        for name, count in segs:
            if slot < cursor + count:
                chosen = name
                break
            cursor += count
        table, _ = _tables(chosen)
        tokens.append(table[byte])
    return " ".join(tokens)


def unblend(stream: str, pattern: str) -> bytes:
    segs = _parse_pattern(pattern)
    total = sum(c for _, c in segs)
    toks = stream.strip().split()
    out = bytearray()
    for i, tok in enumerate(toks):
        slot = i % total
        cursor = 0
        expected = segs[0][0]
        for name, count in segs:
            if slot < cursor + count:
                expected = name
                break
            cursor += count
        _, inv = _tables(expected)
        if tok not in inv:
            raise ValueError(f"Token {tok!r} at position {i} does not belong to tongue {expected}")
        out.append(inv[tok])
    return bytes(out)


# ---------------------------------------------------------------------------
# GeoSeal: context-aware envelope
# ---------------------------------------------------------------------------

GEOSEAL_SECRET_ENV = "GEOSEAL_KEY"


def _geoseal_key() -> bytes:
    """Fetch or derive the GeoSeal HMAC key."""
    key = os.environ.get(GEOSEAL_SECRET_ENV)
    if key:
        return hashlib.sha256(key.encode()).digest()
    # Deterministic default for reproducible selftest
    return hashlib.sha256(b"scbe-aethermoore-default-key-do-not-use-in-production").digest()


def _healpix_ish(lat: float, lon: float, nside: int = 16) -> int:
    """A cheap spatial index -- not real Healpix, good enough for selftest and demo."""
    # Clamp lat to [-90, 90], lon to [-180, 180)
    lat = max(-90.0, min(90.0, lat))
    lon = ((lon + 180.0) % 360.0) - 180.0
    # Map to a tile in [0, nside*nside)
    row = int((lat + 90.0) / 180.0 * nside)
    col = int((lon + 180.0) / 360.0 * nside)
    row = min(nside - 1, row)
    col = min(nside - 1, col)
    return row * nside + col


def geoseal_encrypt(payload: bytes, lat: float, lon: float, tag: str = "") -> dict:
    """Wrap a payload with a context seal. Not actual encryption -- integrity + binding."""
    key = _geoseal_key()
    tile = _healpix_ish(lat, lon)
    ts = int(time.time())
    nonce = secrets.token_hex(8)
    context = {
        "tile": tile,
        "ts": ts,
        "nonce": nonce,
        "tag": tag,
        "version": 1,
    }
    # AAD = context JSON
    aad = json.dumps(context, sort_keys=True, separators=(",", ":")).encode()
    # Mac over AAD || payload
    mac = hmac.new(key, aad + b"||" + payload, hashlib.sha256).hexdigest()
    return {
        "context": context,
        "payload_hex": payload.hex(),
        "mac": mac,
    }


def geoseal_decrypt(sealed: dict, expect_tag: str | None = None) -> bytes:
    """Verify integrity, optionally check tag, return the payload."""
    required = {"context", "payload_hex", "mac"}
    if not required.issubset(sealed.keys()):
        raise ValueError(f"Missing fields. Need {required}")
    key = _geoseal_key()
    aad = json.dumps(sealed["context"], sort_keys=True, separators=(",", ":")).encode()
    payload = bytes.fromhex(sealed["payload_hex"])
    expected_mac = hmac.new(key, aad + b"||" + payload, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected_mac, sealed["mac"]):
        raise ValueError("GeoSeal MAC mismatch -- seal broken or tampered")
    if expect_tag is not None and sealed["context"].get("tag") != expect_tag:
        raise ValueError(f"Tag mismatch: expected {expect_tag!r}, got {sealed['context'].get('tag')!r}")
    return payload


# ---------------------------------------------------------------------------
# Free AI navigation and explanation
# ---------------------------------------------------------------------------

AI_SYSTEM_PROMPT = """You are the GeoSeal CLI's built-in helper. You explain the Six Sacred Tongues tokenization
system and the GeoSeal context-aware sealing protocol to developers using the CLI.

Context about the system:
- 6 tongues: KO (Kor'aelin, Control, weight 1.000), AV (Avali, I/O, 1.618), RU (Runethic, Policy, 2.618),
  CA (Cassisivadan, Logic, 4.236), UM (Umbroth, Security, 6.854), DR (Draumric, Types, 11.09).
- Weights are the Fibonacci / golden-ratio sequence: phi^0 through phi^5.
- Each tongue has 256 unique bijective tokens (one per byte).
- Encoding is byte-for-byte lossless. xlate, blend, unblend are exact.
- GeoSeal wraps payloads in a context envelope (lat/lon tile + timestamp + tag) and adds HMAC-SHA256 integrity.
- This is an open-source CLI. The full framework is SCBE-AETHERMOORE on GitHub.

Be concise. Give 2-4 sentence answers unless the user asks for depth. Use code examples when helpful."""


def _try_ollama(prompt: str) -> str | None:
    """Try a local Ollama instance for an explanation. Returns None if unavailable."""
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    model = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
    url = f"{host.rstrip('/')}/api/generate"
    body = json.dumps({
        "model": model,
        "prompt": f"{AI_SYSTEM_PROMPT}\n\nUser question: {prompt}\n\nAnswer:",
        "stream": False,
        "options": {"temperature": 0.4, "num_predict": 300},
    }).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            data = json.loads(resp.read())
            return (data.get("response") or "").strip() or None
    except Exception:
        return None


def _try_hf(prompt: str) -> str | None:
    """Try HuggingFace Inference API. Returns None if HF_TOKEN missing or request fails."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        return None
    model = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    url = f"https://api-inference.huggingface.co/models/{model}"
    full_prompt = (
        f"<|im_start|>system\n{AI_SYSTEM_PROMPT}\n<|im_end|>\n"
        f"<|im_start|>user\n{prompt}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    body = json.dumps({
        "inputs": full_prompt,
        "parameters": {"max_new_tokens": 300, "temperature": 0.4, "return_full_text": False},
    }).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            if isinstance(data, list) and data:
                text = data[0].get("generated_text", "") or ""
                if "<|im_end|>" in text:
                    text = text.split("<|im_end|>")[0]
                return text.strip() or None
            if isinstance(data, dict) and "error" in data:
                return None
    except Exception:
        return None
    return None


def _static_help(prompt: str) -> str:
    """Curated static fallback when no AI is available."""
    p = (prompt or "").lower()
    for tongue, meta in TONGUE_META.items():
        if tongue.lower() in p or meta["name"].lower().split("'")[0] in p:
            return (
                f"{tongue} ({meta['name']}) is the tongue for {meta['domain'].lower()}. "
                f"Its weight in the Fibonacci/phi security hierarchy is {meta['weight']:.3f}. "
                f"256 unique bijective tokens, one per byte. Encoding is lossless and cross-tongue "
                f"translation preserves bytes exactly."
            )
    if "blend" in p or "pattern" in p:
        return (
            "Blend interleaves bytes across multiple tongues. Pattern 'KO:2,AV:1,DR:1' means 2 KO tokens, "
            "then 1 AV, then 1 DR, repeating. Unblend reverses it. Useful for semantic steganography "
            "or multi-channel AI-to-AI messaging."
        )
    if "geoseal" in p or "encrypt" in p or "seal" in p:
        return (
            "GeoSeal wraps a payload with context metadata (lat/lon tile, timestamp, tag) and binds "
            "an HMAC-SHA256 over the whole envelope. If anyone modifies payload, context, or forgets "
            "the shared key, verification fails. It's a context-binding layer -- real PQC encryption "
            "plugs in underneath."
        )
    if "tongue" in p or "alphabet" in p:
        tongues_list = ", ".join(f"{t} ({m['name']})" for t, m in TONGUE_META.items())
        return (
            f"The Six Sacred Tongues are: {tongues_list}. Each has 256 unique tokens. "
            "Run `python aethermoore.py explain --tongue KO` for details on any one of them."
        )
    return (
        "GeoSeal CLI help: encode/decode/xlate/blend/unblend are lossless byte operations. "
        "geoseal-encrypt/decrypt wraps data in a context envelope. Set HF_TOKEN or run Ollama locally "
        "for AI-powered answers. Ask about a specific tongue (KO, AV, RU, CA, UM, DR) or 'blend', "
        "'geoseal', or 'pattern'."
    )


def explain(question: str) -> tuple[str, str]:
    """Return (source, text). Source: 'ollama', 'huggingface', or 'static'."""
    if not question.strip():
        question = "Give me a quick overview of the Six Sacred Tongues system."
    # Ollama first (local, free, private)
    out = _try_ollama(question)
    if out:
        return "ollama", out
    # HuggingFace Inference API (free with token)
    out = _try_hf(question)
    if out:
        return "huggingface", out
    # Static fallback (always available)
    return "static", _static_help(question)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def selftest() -> int:
    errors = []

    # Token tables must be bijective for all tongues
    for t in TONGUES:
        table, inv = _tables(t)
        if len(table) != 256 or len(set(table)) != 256:
            errors.append(f"{t} token table not bijective")
        for i in range(256):
            if inv[table[i]] != i:
                errors.append(f"{t} inverse mismatch at {i}")

    # Encode/decode round-trip for each tongue
    sample = b"Hello SCBE-AETHERMOORE \x00\x01\x02 weights=phi^n"
    for t in TONGUES:
        try:
            enc = encode(sample, t)
            dec = decode(enc, t)
            if dec != sample:
                errors.append(f"{t} encode/decode round-trip failed")
        except Exception as e:
            errors.append(f"{t} encode/decode error: {e}")

    # Cross-tongue translation must preserve bytes
    try:
        enc = encode(sample, "KO")
        path = xlate(xlate(xlate(enc, "KO", "AV"), "AV", "DR"), "DR", "UM")
        back = decode(path, "UM")
        if back != sample:
            errors.append("xlate chain did not preserve bytes")
    except Exception as e:
        errors.append(f"xlate error: {e}")

    # Blend/unblend round-trip
    try:
        bl = blend(sample, "KO:2,AV:1,DR:1")
        ub = unblend(bl, "KO:2,AV:1,DR:1")
        if ub != sample:
            errors.append("blend/unblend did not round-trip")
    except Exception as e:
        errors.append(f"blend error: {e}")

    # GeoSeal wrap/verify
    try:
        sealed = geoseal_encrypt(sample, 48.118, -123.430, tag="selftest")
        out = geoseal_decrypt(sealed, expect_tag="selftest")
        if out != sample:
            errors.append("geoseal wrap/verify payload mismatch")
        # Tamper check
        sealed_bad = dict(sealed)
        sealed_bad["payload_hex"] = sealed["payload_hex"][:-2] + "ff"
        try:
            geoseal_decrypt(sealed_bad, expect_tag="selftest")
            errors.append("geoseal did not detect tampered payload")
        except ValueError:
            pass
    except Exception as e:
        errors.append(f"geoseal error: {e}")

    if errors:
        print("selftest FAILED:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("selftest ok")
    print(f"  6 tongues, 256 tokens each, {6*256} total tokens")
    print(f"  encode/decode/xlate/blend/unblend/geoseal all pass")
    print("  try: python aethermoore.py explain --tongue KO")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _read_text(args) -> bytes:
    if getattr(args, "text", None):
        return args.text.encode()
    if not sys.stdin.isatty():
        return sys.stdin.buffer.read()
    return b""


def main() -> int:
    parser = argparse.ArgumentParser(description="Six Tongues Tokenizer + GeoSeal CLI")
    sub = parser.add_subparsers(dest="cmd")

    p_enc = sub.add_parser("encode", help="Encode bytes into a tongue")
    p_enc.add_argument("--tongue", required=True)
    p_enc.add_argument("--text")

    p_dec = sub.add_parser("decode", help="Decode a token stream")
    p_dec.add_argument("--tongue", required=True)
    p_dec.add_argument("stream", nargs="?")

    p_xl = sub.add_parser("xlate", help="Cross-tongue translate")
    p_xl.add_argument("--src", required=True)
    p_xl.add_argument("--dst", required=True)
    p_xl.add_argument("--text")

    p_bl = sub.add_parser("blend", help="Blend bytes across tongues")
    p_bl.add_argument("--pattern", required=True)
    p_bl.add_argument("--text")

    p_ub = sub.add_parser("unblend", help="Unblend a multi-tongue stream")
    p_ub.add_argument("--pattern", required=True)
    p_ub.add_argument("--text")

    p_ge = sub.add_parser("geoseal-encrypt", help="Wrap a payload with a context seal")
    p_ge.add_argument("--lat", type=float, required=True)
    p_ge.add_argument("--lon", type=float, required=True)
    p_ge.add_argument("--tag", default="")
    p_ge.add_argument("--text")

    p_gd = sub.add_parser("geoseal-decrypt", help="Verify and unwrap a sealed payload")
    p_gd.add_argument("--sealed", required=True, help="JSON string of the sealed object")
    p_gd.add_argument("--expect-tag")

    p_ex = sub.add_parser("explain", help="Ask the built-in AI to explain a tongue or operation")
    p_ex.add_argument("--tongue", help="Name of a specific tongue (KO/AV/RU/CA/UM/DR)")
    p_ex.add_argument("--what", help="Arbitrary question about the system")

    p_it = sub.add_parser("tongues", help="List the six sacred tongues with their weights")

    p_st = sub.add_parser("selftest", help="Run round-trip + integrity self-checks")

    args = parser.parse_args()

    if not args.cmd or args.cmd == "selftest":
        return selftest()

    if args.cmd == "encode":
        data = _read_text(args)
        if not data:
            print("error: no input (use --text or pipe)", file=sys.stderr)
            return 2
        print(encode(data, args.tongue))
        return 0

    if args.cmd == "decode":
        stream = args.stream
        if not stream and not sys.stdin.isatty():
            stream = sys.stdin.read()
        if not stream:
            print("error: no stream", file=sys.stderr)
            return 2
        sys.stdout.buffer.write(decode(stream, args.tongue))
        return 0

    if args.cmd == "xlate":
        stream = args.text
        if not stream and not sys.stdin.isatty():
            stream = sys.stdin.read()
        if not stream:
            print("error: no input", file=sys.stderr)
            return 2
        print(xlate(stream, args.src, args.dst))
        return 0

    if args.cmd == "blend":
        data = _read_text(args)
        if not data:
            print("error: no input", file=sys.stderr)
            return 2
        print(blend(data, args.pattern))
        return 0

    if args.cmd == "unblend":
        stream = args.text
        if not stream and not sys.stdin.isatty():
            stream = sys.stdin.read()
        if not stream:
            print("error: no input", file=sys.stderr)
            return 2
        sys.stdout.buffer.write(unblend(stream, args.pattern))
        return 0

    if args.cmd == "geoseal-encrypt":
        data = _read_text(args)
        if not data:
            print("error: no input (use --text or pipe)", file=sys.stderr)
            return 2
        sealed = geoseal_encrypt(data, args.lat, args.lon, tag=args.tag)
        print(json.dumps(sealed, indent=2))
        return 0

    if args.cmd == "geoseal-decrypt":
        try:
            obj = json.loads(args.sealed)
        except json.JSONDecodeError as e:
            print(f"error: bad JSON -- {e}", file=sys.stderr)
            return 2
        try:
            payload = geoseal_decrypt(obj, expect_tag=args.expect_tag)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 3
        sys.stdout.buffer.write(payload)
        return 0

    if args.cmd == "tongues":
        for t in TONGUES:
            m = TONGUE_META[t]
            print(f"{t}  {m['name']:14s}  weight={m['weight']:<6.3f}  {m['domain']}")
        return 0

    if args.cmd == "explain":
        question = ""
        if args.tongue:
            t = args.tongue.upper()
            if t not in TONGUES:
                print(f"error: unknown tongue {t}. Pick from {TONGUES}", file=sys.stderr)
                return 2
            m = TONGUE_META[t]
            question = (
                f"Explain the {t} tongue ({m['name']}, domain={m['domain']}, weight={m['weight']:.3f}). "
                f"When would a developer use it in the GeoSeal CLI?"
            )
        elif args.what:
            question = args.what
        source, text = explain(question)
        print(f"[{source}] {text}")
        if source == "static":
            print(
                "\ntip: set HF_TOKEN for free HuggingFace Inference, or run Ollama locally "
                "(OLLAMA_HOST=http://127.0.0.1:11434) for private answers.",
                file=sys.stderr,
            )
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
