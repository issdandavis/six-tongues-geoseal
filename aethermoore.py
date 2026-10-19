#!/usr/bin/env python3
"""
SCBE-AETHERMOORE Six Tongues Tokenizer + GeoSeal CLI

A self-contained Python CLI for:
- Six Sacred Tongues bijective tokenization (256 tokens per tongue)
- Cross-tongue translation (KO -> AV -> DR, etc.)
- Blend / unblend of multi-tongue streams
- GeoSeal: context-aware sealing with HMAC integrity
- 'explain' subcommand: free AI navigation via HuggingFace or Ollama
- 'entropy' subcommand: measure byte entropy and tokenizer surface overhead
- 'predict-compress' subcommand: score causal next-byte predictions as ideal bits
- 'projection-benchmark' subcommand: test aperiodic relational projections
- 'trit-overlay' subcommand: overlay ternary prefixes on binary code space
- 'streams' subcommand: one byte board read as six tongues, a Newton tangent, and optional stamps
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
    python aethermoore.py entropy --tongue KO --text "hello hello hello"
    python aethermoore.py predict-compress --max-order 3 --text "hello hello hello"
    Get-Content -Raw README.md | python aethermoore.py projection-benchmark
    python aethermoore.py trit-overlay --trit-depth 2 --binary-depth 4
    python aethermoore.py streams --hex 0103060a0f
    python aethermoore.py streams --text "hello" --words not,run --seal --lat 48.118 --lon -123.430

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
import zlib
from typing import Any, Optional

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


def _byte_entropy_bits(data: bytes) -> float:
    """Shannon entropy of the byte distribution, in bits per byte."""
    if not data:
        return 0.0
    counts = [0] * 256
    for b in data:
        counts[b] += 1
    total = len(data)
    entropy = 0.0
    for count in counts:
        if count:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy


def entropy_report(data: bytes, tongue: str = "KO") -> dict[str, Any]:
    """Measure corpus entropy and the surface overhead of a tongue encoding."""
    tongue = tongue.upper()
    if tongue not in TONGUES:
        raise ValueError(f"Unknown tongue: {tongue}. Pick one of {TONGUES}")

    token_stream = encode(data, tongue).encode()
    entropy_bits = _byte_entropy_bits(data)
    lower_bound_bits = entropy_bits * len(data)
    lower_bound_bytes = math.ceil(lower_bound_bits / 8) if data else 0
    raw_zlib = len(zlib.compress(data)) if data else 0
    token_zlib = len(zlib.compress(token_stream)) if token_stream else 0

    return {
        "input_bytes": len(data),
        "unique_bytes": len(set(data)),
        "byte_entropy_bits_per_byte": round(entropy_bits, 6),
        "shannon_lower_bound_bytes": lower_bound_bytes,
        "raw_zlib_bytes": raw_zlib,
        "tongue": tongue,
        "token_surface_bytes": len(token_stream),
        "token_surface_ratio": round(len(token_stream) / len(data), 6) if data else 0.0,
        "token_zlib_bytes": token_zlib,
        "token_zlib_ratio": round(token_zlib / len(data), 6) if data else 0.0,
        "note": "Bijective tongue tokens preserve bytes; this measures corpus compressibility and surface overhead.",
    }


TRITS_PER_BYTE = 6
TRIT_LIGHT_CLASSES = {
    "0": {"vote": -1, "class": "blocker"},
    "1": {"vote": 0, "class": "reader"},
    "2": {"vote": 1, "class": "writer"},
}

PROJECTION_CONTEXT_TRITS = 8
PROJECTION_DEPTHS = 3
PROJECTION_RAYS = 4
PROJECTION_CHANNELS = 2
PROJECTION_SEEDS = (7, 20, 42)
PROJECTION_BRANCH_FEATURES = PROJECTION_DEPTHS * PROJECTION_RAYS * PROJECTION_CHANNELS
PROJECTION_FEATURES = PROJECTION_CONTEXT_TRITS * 3 + PROJECTION_BRANCH_FEATURES
PROJECTION_PARAMETERS = 3 * (PROJECTION_FEATURES + 1)
GOLDEN_RATIO = (1.0 + math.sqrt(5.0)) / 2.0
PROJECTION_GEOMETRIES = {
    "scrambled": {
        "model": "scrambled_projection_control",
        "rational_angles": False,
        "repeated_random_angles": False,
        "dyadic_scales": False,
        "irrational_phase": False,
    },
    "quasi_random": {
        "model": "quasi_random_projection",
        "rational_angles": False,
        "repeated_random_angles": True,
        "dyadic_scales": False,
        "irrational_phase": True,
    },
    "dyadic_only": {
        "model": "dyadic_scale_projection",
        "rational_angles": False,
        "repeated_random_angles": False,
        "dyadic_scales": True,
        "irrational_phase": False,
    },
    "rational_only": {
        "model": "rational_angle_projection",
        "rational_angles": True,
        "repeated_random_angles": False,
        "dyadic_scales": False,
        "irrational_phase": True,
    },
    "periodic_aligned": {
        "model": "periodic_rational_dyadic_projection",
        "rational_angles": True,
        "repeated_random_angles": False,
        "dyadic_scales": True,
        "irrational_phase": False,
    },
    "aligned": {
        "model": "aperiodic_relational_projection",
        "rational_angles": True,
        "repeated_random_angles": False,
        "dyadic_scales": True,
        "irrational_phase": True,
    },
}
PROJECTION_ABLATION_KINDS = (
    "scrambled",
    "quasi_random",
    "dyadic_only",
    "rational_only",
    "periodic_aligned",
    "aligned",
)


def prediction_compression_report(
    data: bytes,
    max_order: int = 3,
    alpha: float = 0.5,
    trit_max_order: Optional[int] = None,
) -> dict[str, Any]:
    """Compare byte next-token prediction against trit-class prediction."""
    if not data:
        raise ValueError("no input bytes")
    if max_order < 0 or max_order > 8:
        raise ValueError("max_order must be between 0 and 8")
    if trit_max_order is None:
        trit_max_order = max(1, max_order * TRITS_PER_BYTE)
    if trit_max_order < 0 or trit_max_order > 48:
        raise ValueError("trit_max_order must be between 0 and 48")
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    byte_reports = _causal_symbol_reports(
        list(data),
        vocab_size=256,
        max_order=max_order,
        alpha=alpha,
        normalize_by=len(data),
    )
    trit_reports = _causal_symbol_reports(
        bytes_to_trit_symbols(data),
        vocab_size=3,
        max_order=trit_max_order,
        alpha=alpha,
        normalize_by=len(data),
    )
    byte_best = min(byte_reports, key=lambda row: row["ideal_bits_per_input_byte"])
    trit_best = min(trit_reports, key=lambda row: row["ideal_bits_per_input_byte"])
    delta = trit_best["ideal_bits_per_input_byte"] - byte_best["ideal_bits_per_input_byte"]
    winner = "tie"
    if abs(delta) > 1e-9:
        winner = "trit_light_projection" if delta < 0 else "traditional_next_byte"

    return {
        "input_bytes": len(data),
        "smoothing_alpha": alpha,
        "traditional_next_byte": {
            "symbol_vocab": 256,
            "baseline_order_0": byte_reports[0],
            "best": byte_best,
            "orders": byte_reports,
        },
        "trit_light_projection": {
            "symbol_vocab": 3,
            "trits_per_input_byte": TRITS_PER_BYTE,
            "class_mapping": TRIT_LIGHT_CLASSES,
            "best": trit_best,
            "orders": trit_reports,
        },
        "comparison": {
            "traditional_bits_per_input_byte": byte_best["ideal_bits_per_input_byte"],
            "trit_bits_per_input_byte": trit_best["ideal_bits_per_input_byte"],
            "delta_trit_minus_traditional_bits_per_input_byte": round(delta, 6),
            "winner": winner,
        },
        "note": (
            "Lower bits/input_byte wins. This is the same cross-entropy/ideal-code metric "
            "used by next-token training, without training a neural network."
        ),
    }


def _causal_symbol_reports(
    symbols: list[int],
    vocab_size: int,
    max_order: int,
    alpha: float,
    normalize_by: int,
) -> list[dict[str, Any]]:
    if not symbols:
        raise ValueError("no symbols")

    reports = []
    for order in range(max_order + 1):
        counts: dict[tuple[int, ...], list[int]] = {}
        totals: dict[tuple[int, ...], int] = {}
        total_bits = 0.0
        for i, symbol in enumerate(symbols):
            context = tuple(symbols[max(0, i - order):i]) if order else ()
            if context not in counts:
                counts[context] = [0] * vocab_size
                totals[context] = 0
            numerator = counts[context][symbol] + alpha
            denominator = totals[context] + alpha * vocab_size
            probability = numerator / denominator
            total_bits += -math.log2(probability)
            counts[context][symbol] += 1
            totals[context] += 1
        reports.append({
            "order": order,
            "contexts": len(counts),
            "total_ideal_bits": round(total_bits, 6),
            "ideal_bits_per_symbol": round(total_bits / len(symbols), 6),
            "ideal_bits_per_input_byte": round(total_bits / normalize_by, 6),
            "ideal_bytes": math.ceil(total_bits / 8),
        })
    return reports


def projection_depth_report(
    data: bytes,
    epochs: int = 8,
    learning_rate: float = 0.12,
    projection_gain: float = 1.0,
    layer_gain: float = 1.0,
    quadratic_gain: float = 0.0,
    reinforcement_gain: float = 0.0,
    split_fraction: float = 0.8,
    max_train_samples: int = 6000,
    max_test_samples: int = 1500,
    seeds: tuple[int, ...] = PROJECTION_SEEDS,
    include_ablations: bool = False,
    layered_copies: int = 1,
    layered_decay: float = 0.5,
    layered_mode: str = "probability",
) -> dict[str, Any]:
    """Benchmark a causal relational projection model on held-out trits."""
    if len(data) < 32:
        raise ValueError("projection benchmark needs at least 32 input bytes")
    if epochs < 1 or epochs > 100:
        raise ValueError("epochs must be between 1 and 100")
    if learning_rate <= 0 or learning_rate > 2:
        raise ValueError("learning_rate must be in (0, 2]")
    if projection_gain <= 0 or projection_gain > 8:
        raise ValueError("projection_gain must be in (0, 8]")
    if layer_gain < 0 or layer_gain > 4:
        raise ValueError("layer_gain must be in [0, 4]")
    if quadratic_gain < 0 or quadratic_gain > 8:
        raise ValueError("quadratic_gain must be in [0, 8]")
    if reinforcement_gain < 0 or reinforcement_gain > 8:
        raise ValueError("reinforcement_gain must be in [0, 8]")
    if split_fraction < 0.1 or split_fraction > 0.9:
        raise ValueError("split_fraction must be in [0.1, 0.9]")
    if max_train_samples < 32 or max_test_samples < 16:
        raise ValueError("sample caps are too small")
    if not seeds:
        raise ValueError("at least one seed is required")
    if layered_copies < 1 or layered_copies > 16:
        raise ValueError("layered_copies must be between 1 and 16")
    if layered_decay < 0 or layered_decay > 1:
        raise ValueError("layered_decay must be in [0, 1]")
    if layered_mode not in ("feature", "probability"):
        raise ValueError("layered_mode must be 'feature' or 'probability'")
    if layered_copies > 1 and not include_ablations:
        raise ValueError("layered_copies greater than 1 requires include_ablations")
    if PROJECTION_BRANCH_FEATURES != PROJECTION_CONTEXT_TRITS * 3:
        raise RuntimeError("projection and flat branch feature budgets do not match")

    symbols = bytes_to_trit_symbols(data)
    split_byte = max(
        1,
        min(len(data) - 1, int(len(data) * split_fraction)),
    )
    split_symbol = split_byte * TRITS_PER_BYTE
    train_start = max(PROJECTION_CONTEXT_TRITS, split_symbol - max_train_samples)
    test_end = min(len(symbols), split_symbol + max_test_samples)
    train_positions = list(range(train_start, split_symbol))
    test_positions = list(range(split_symbol, test_end))
    if len(train_positions) < 32 or len(test_positions) < 16:
        raise ValueError("not enough train/test symbols after the causal split")

    train_targets = [symbols[position] for position in train_positions]
    test_targets = [symbols[position] for position in test_positions]
    flat_train = _projection_feature_rows(symbols, train_positions, "flat", seeds[0])
    flat_test = _projection_feature_rows(symbols, test_positions, "flat", seeds[0])

    projection_kinds = (
        PROJECTION_ABLATION_KINDS
        if include_ablations
        else ("scrambled", "aligned")
    )
    projection_models = tuple(
        (kind, PROJECTION_GEOMETRIES[kind]["model"], 1)
        for kind in projection_kinds
    )
    layered_control_name = (
        f"layered_{layered_mode}_scrambled_projection_control"
    )
    layered_candidate_name = (
        f"layered_{layered_mode}_quasi_random_projection"
    )
    if layered_copies > 1:
        projection_models += (
            (
                "scrambled",
                layered_control_name,
                layered_copies,
            ),
            (
                "quasi_random",
                layered_candidate_name,
                layered_copies,
            ),
        )
    model_names = (
        "traditional_flat_context",
        *(name for _, name, _ in projection_models),
    )
    scores: dict[str, list[float]] = {name: [] for name in model_names}
    runs: list[dict[str, Any]] = []

    for seed in seeds:
        run_scores: dict[str, float] = {}
        flat_score = _fit_softmax_head(
            flat_train,
            train_targets,
            flat_test,
            test_targets,
            epochs,
            learning_rate,
        )
        run_scores["traditional_flat_context"] = flat_score
        scores["traditional_flat_context"].append(flat_score)

        for kind, name, copies in projection_models:
            if copies > 1 and layered_mode == "probability":
                train_views = [
                    _projection_feature_rows(
                        symbols,
                        train_positions,
                        kind,
                        _projection_copy_seed(seed, copy_index),
                        projection_gain,
                        layer_gain,
                        quadratic_gain,
                        reinforcement_gain,
                    )
                    for copy_index in range(copies)
                ]
                test_views = [
                    _projection_feature_rows(
                        symbols,
                        test_positions,
                        kind,
                        _projection_copy_seed(seed, copy_index),
                        projection_gain,
                        layer_gain,
                        quadratic_gain,
                        reinforcement_gain,
                    )
                    for copy_index in range(copies)
                ]
                score = _fit_softmax_head_probability_mixture(
                    train_views,
                    train_targets,
                    test_views,
                    test_targets,
                    epochs,
                    learning_rate,
                )
            else:
                train_features = _projection_feature_rows(
                    symbols,
                    train_positions,
                    kind,
                    seed,
                    projection_gain,
                    layer_gain,
                    quadratic_gain,
                    reinforcement_gain,
                    copies,
                    layered_decay,
                )
                test_features = _projection_feature_rows(
                    symbols,
                    test_positions,
                    kind,
                    seed,
                    projection_gain,
                    layer_gain,
                    quadratic_gain,
                    reinforcement_gain,
                    copies,
                    layered_decay,
                )
                score = _fit_softmax_head(
                    train_features,
                    train_targets,
                    test_features,
                    test_targets,
                    epochs,
                    learning_rate,
                )
            run_scores[name] = score
            scores[name].append(score)

        runs.append({
            "seed": seed,
            "bits_per_input_byte": {
                name: round(run_scores[name], 6) for name in model_names
            },
        })

    summaries = {
        name: _projection_score_summary(values)
        for name, values in scores.items()
    }
    versus_traditional = _projection_comparison(
        scores["aperiodic_relational_projection"],
        scores["traditional_flat_context"],
    )
    versus_scrambled = _projection_comparison(
        scores["aperiodic_relational_projection"],
        scores["scrambled_projection_control"],
    )
    ablation_comparisons: dict[str, dict[str, Any]] = {}
    synthesis: dict[str, Any] = {}
    layered_synthesis: dict[str, Any] = {}
    if include_ablations:
        ablation_comparisons = {
            "quasi_random_vs_scrambled": _projection_comparison(
                scores["quasi_random_projection"],
                scores["scrambled_projection_control"],
            ),
            "dyadic_scales_vs_scrambled": _projection_comparison(
                scores["dyadic_scale_projection"],
                scores["scrambled_projection_control"],
            ),
            "random_local_rays_vs_rational": _projection_comparison(
                scores["quasi_random_projection"],
                scores["rational_angle_projection"],
            ),
            "dyadic_scales_under_rational_angles": _projection_comparison(
                scores["aperiodic_relational_projection"],
                scores["rational_angle_projection"],
            ),
            "aperiodic_vs_periodic_depth": _projection_comparison(
                scores["aperiodic_relational_projection"],
                scores["periodic_rational_dyadic_projection"],
            ),
        }
        synthesis_vs_traditional = _projection_comparison(
            scores["quasi_random_projection"],
            scores["traditional_flat_context"],
        )
        synthesis_vs_scrambled = ablation_comparisons[
            "quasi_random_vs_scrambled"
        ]
        synthesis_verdict, synthesis_reason = _projection_claim_verdict(
            synthesis_vs_traditional,
            synthesis_vs_scrambled,
            len(seeds),
            "Quasi-random projection",
        )
        synthesis = {
            "candidate": "quasi_random_projection",
            "comparisons": {
                "candidate_vs_traditional": synthesis_vs_traditional,
                "candidate_vs_scrambled": synthesis_vs_scrambled,
            },
            "verdict": synthesis_verdict,
            "verdict_reason": synthesis_reason,
        }
        if layered_copies > 1:
            layered_vs_traditional = _projection_comparison(
                scores[layered_candidate_name],
                scores["traditional_flat_context"],
            )
            layered_vs_scrambled = _projection_comparison(
                scores[layered_candidate_name],
                scores[layered_control_name],
            )
            layered_verdict, layered_reason = _projection_claim_verdict(
                layered_vs_traditional,
                layered_vs_scrambled,
                len(seeds),
                "Layered quasi-random projection",
            )
            layered_synthesis = {
                "candidate": layered_candidate_name,
                "control": layered_control_name,
                "copies": layered_copies,
                "mode": layered_mode,
                "copy_weight_decay": layered_decay,
                "feature_aggregation": (
                    "shared-head probability mixture"
                    if layered_mode == "probability"
                    else "unit-normalized fixed-ratio sum of projection branches"
                ),
                "comparisons": {
                    "candidate_vs_traditional": layered_vs_traditional,
                    "candidate_vs_layered_scrambled": layered_vs_scrambled,
                    "candidate_vs_single_quasi_random": _projection_comparison(
                        scores[layered_candidate_name],
                        scores["quasi_random_projection"],
                    ),
                    "layered_scrambled_vs_single_scrambled": _projection_comparison(
                        scores[layered_control_name],
                        scores["scrambled_projection_control"],
                    ),
                },
                "verdict": layered_verdict,
                "verdict_reason": layered_reason,
            }

    verdict, verdict_reason = _projection_claim_verdict(
        versus_traditional,
        versus_scrambled,
        len(seeds),
        "Original aligned projection",
    )

    return {
        "input_bytes": len(data),
        "input_sha256": hashlib.sha256(data).hexdigest(),
        "causal_split": {
            "fraction": split_fraction,
            "train_bytes_end": split_byte,
            "test_bytes_start": split_byte,
            "train_trit_samples": len(train_positions),
            "test_trit_samples": len(test_positions),
        },
        "training": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "projection_seeds": list(seeds),
            "optimizer_order": "fixed across models and projection seeds",
            "objective": "3-class categorical cross-entropy",
            "score": "held-out ideal bits per input byte",
        },
        "model_order": list(model_names),
        "architecture": {
            "name": "quasi-metronomic quadratic stenographic array",
            "context_trits": PROJECTION_CONTEXT_TRITS,
            "depth_layers": PROJECTION_DEPTHS,
            "rays_per_layer": PROJECTION_RAYS,
            "channels_per_ray": PROJECTION_CHANNELS,
            "feature_count": PROJECTION_FEATURES,
            "features_per_branch": PROJECTION_BRANCH_FEATURES,
            "trainable_parameters_per_model": PROJECTION_PARAMETERS,
            "projection_gain": projection_gain,
            "layer_gain": layer_gain,
            "quadratic_gain": quadratic_gain,
            "reinforcement_gain": reinforcement_gain,
            "local_angles": "rational turns r/4",
            "layer_phase": "irrational turn d/phi",
            "local_scales": "self-similar dyadic scales 1, 1/2, 1/4",
            "spatial_distortion": "seeded fixed random warp per depth and context position",
            "relation_lines": "mean projected slope over every ordered pair in the causal context",
            "projection_backdrop": (
                "phase-aware recurrent first-harmonic reconstruction of prior ray arrays "
                "(current array at depth 0); residual magnitude is self/non-self alignment"
            ),
            "overlay": (
                "24 projected features run beside 24 raw trit features; "
                "the shared softmax head learns the fusion"
            ),
            "global_property": "aperiodic and intentionally non-self-similar",
            "factorial_ablations": include_ablations,
            "layered_projection_copies": layered_copies,
            "layered_copy_weight_decay": layered_decay,
            "layered_mode": layered_mode,
            "layered_feature_aggregation": (
                "shared-head probability mixture"
                if layered_mode == "probability"
                else (
                    "unit-normalized fixed-ratio sum; "
                    "feature and parameter counts stay fixed"
                )
            ),
        },
        "controls": {
            "traditional_flat_context": (
                "Raw one-hot causal context duplicated into two equal branches."
            ),
            "scrambled_projection_control": (
                "Same distortion, feature count, backdrop operation, and trainable head; "
                "angles and scales are independently randomized."
            ),
            "factorial_components": {
                "quasi_random_projection": (
                    "One random local ray array repeated under irrational depth rotation."
                ),
                "dyadic_scale_projection": "Random angles with dyadic scales.",
                "rational_angle_projection": (
                    "Rational local rays and irrational depth phase with random scales."
                ),
                "periodic_rational_dyadic_projection": (
                    "Rational local rays and dyadic scales without irrational depth phase."
                ),
                layered_control_name: (
                    "Matched independently seeded scrambled projection overlay."
                ),
                layered_candidate_name: (
                    "Independently seeded quasi-random projection overlay."
                ),
            },
        },
        "runs": runs,
        "models": summaries,
        "comparisons": {
            "candidate_vs_traditional": versus_traditional,
            "candidate_vs_scrambled": versus_scrambled,
        },
        "ablation_comparisons": ablation_comparisons,
        "synthesis": synthesis,
        "layered_synthesis": layered_synthesis,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "note": (
            "This is a fixed nonlinear projection array plus a trained softmax head, "
            "not a claim that the geometry is already a full language model."
        ),
    }


def _projection_feature_rows(
    symbols: list[int],
    positions: list[int],
    kind: str,
    seed: int,
    projection_gain: float = 1.0,
    layer_gain: float = 1.0,
    quadratic_gain: float = 0.0,
    reinforcement_gain: float = 0.0,
    copies: int = 1,
    copy_weight_decay: float = 0.5,
) -> list[list[float]]:
    if kind != "flat" and kind not in PROJECTION_GEOMETRIES:
        raise ValueError(f"unknown projection feature kind: {kind}")
    if copies < 1:
        raise ValueError("projection copies must be positive")
    if copy_weight_decay < 0 or copy_weight_decay > 1:
        raise ValueError("projection copy weight decay must be in [0, 1]")
    if kind == "flat" and copies != 1:
        raise ValueError("flat features do not accept projection copies")

    projection_maps: list[tuple[list[list[float]], list[str]]] = []
    if kind != "flat":
        for copy_index in range(copies):
            projection_maps.append(_projection_feature_map(
                PROJECTION_CONTEXT_TRITS,
                PROJECTION_DEPTHS,
                PROJECTION_RAYS,
                _projection_copy_seed(seed, copy_index),
                geometry=kind,
                layer_gain=layer_gain,
            ))

    rows: list[list[float]] = []
    for position in positions:
        history = symbols[position - PROJECTION_CONTEXT_TRITS:position]
        flat_features = _flat_context_features(history)
        if kind == "flat":
            rows.append(_unit_vector(flat_features + flat_features))
        else:
            projected_copies = [
                _project_context_features(
                    history,
                    matrix,
                    modes,
                    quadratic_gain,
                    reinforcement_gain,
                )
                for matrix, modes in projection_maps
            ]
            copy_weights = [
                copy_weight_decay ** copy_index
                for copy_index in range(copies)
            ]
            weight_total = sum(copy_weights)
            projected_features = _unit_vector([
                sum(
                    weight * projected[index]
                    for weight, projected in zip(copy_weights, projected_copies)
                ) / weight_total
                for index in range(PROJECTION_BRANCH_FEATURES)
            ])
            rows.append(_unit_vector(
                flat_features
                + [projection_gain * projected for projected in projected_features]
            ))
    return rows


def _projection_copy_seed(seed: int, copy_index: int) -> int:
    return seed + copy_index * 104729


def _flat_context_features(history: list[int]) -> list[float]:
    width = len(history) * 3
    features = [0.0] * width
    scale = 1.0 / math.sqrt(len(history))
    for position, symbol in enumerate(history):
        features[position * 3 + symbol] = scale
    return features


def _projection_feature_map(
    context: int,
    depths: int,
    rays: int,
    seed: int,
    geometry: str,
    layer_gain: float,
) -> tuple[list[list[float]], list[str]]:
    """Build fixed relation-line rows for the deep projection array."""
    distortion_rng = random.Random(seed ^ 0x5CBEA37)
    distortions = [
        [
            (
                distortion_rng.uniform(-0.18, 0.18),
                distortion_rng.uniform(-0.18, 0.18),
            )
            for _ in range(context)
        ]
        for _ in range(depths)
    ]
    angle_rng = random.Random(seed ^ 0xA93F107)
    scale_rng = random.Random(seed ^ 0x71C4E2D)
    random_angles = [
        [angle_rng.uniform(0.0, math.tau) for _ in range(rays)]
        for _ in range(depths)
    ]
    random_scales = [
        [
            tuple(scale_rng.uniform(0.2, 1.0) for _ in range(3))
            for _ in range(rays)
        ]
        for _ in range(depths)
    ]
    geometry_spec = PROJECTION_GEOMETRIES[geometry]
    signal_layers: list[list[list[float]]] = []
    angle_layers: list[list[float]] = []

    for depth in range(depths):
        layer_rows: list[list[float]] = []
        layer_angles: list[float] = []
        warped_x = [
            (-1.0 + (2.0 * position / (context - 1))) + distortions[depth][position][0]
            for position in range(context)
        ]
        x_center = sum(warped_x) / context

        for ray in range(rays):
            if geometry_spec["rational_angles"]:
                layer_phase = (
                    depth / GOLDEN_RATIO
                    if geometry_spec["irrational_phase"]
                    else 0.0
                )
                theta = math.tau * ((ray / rays) + layer_phase)
            elif geometry_spec["repeated_random_angles"]:
                theta = (
                    random_angles[0][ray]
                    + math.tau * depth / GOLDEN_RATIO
                )
            else:
                theta = random_angles[depth][ray]
            if geometry_spec["dyadic_scales"]:
                scales = (1.0, 0.5, 0.25)
            else:
                scales = random_scales[depth][ray]

            cos_theta = math.cos(theta)
            sin_theta = math.sin(theta)
            scale_total = sum(scales)
            row = [0.0] * (context * 3)
            for position in range(context):
                x_value = warped_x[position] - x_center
                y_warp = distortions[depth][position][1]
                sequence_phase = math.pi * (position + 1) / (context + 1)
                for symbol in range(3):
                    y_value = (symbol - 1) + y_warp
                    line_projection = x_value * cos_theta + y_value * sin_theta
                    overlay = sum(
                        scale * math.sin(
                            math.tau * line_projection / scale + sequence_phase
                        )
                        for scale in scales
                    ) / scale_total
                    row[position * 3 + symbol] = overlay
            layer_rows.append(_unit_vector(row))
            layer_angles.append(theta)
        signal_layers.append(layer_rows)
        angle_layers.append(layer_angles)

    feature_rows: list[list[float]] = []
    modes: list[str] = []
    for depth, layer_rows in enumerate(signal_layers):
        for ray, signal_row in enumerate(layer_rows):
            target_angle = angle_layers[depth][ray]
            if depth:
                source_depths = range(depth)
                backdrop_scale = 2.0 / rays
            else:
                source_depths = range(1)
                backdrop_scale = 2.0 / rays
            backdrop = [
                backdrop_scale * sum(
                    (
                        layer_gain ** (depth - 1 - source_depth)
                        if depth
                        else 1.0
                    ) * math.cos(target_angle - angle_layers[source_depth][source_ray])
                    * signal_layers[source_depth][source_ray][index]
                    for source_depth in source_depths
                    for source_ray in range(rays)
                )
                for index in range(context * 3)
            ]
            residual_row = [
                signal_row[index] - backdrop[index]
                for index in range(context * 3)
            ]
            feature_rows.append(signal_row)
            modes.append("signal")
            feature_rows.append(_unit_vector(residual_row))
            modes.append("alignment")

    return feature_rows, modes


def _project_context_features(
    history: list[int],
    matrix: list[list[float]],
    modes: list[str],
    quadratic_gain: float,
    reinforcement_gain: float,
) -> list[float]:
    active = [position * 3 + symbol for position, symbol in enumerate(history)]
    features: list[float] = []
    for row, mode in zip(matrix, modes):
        projected_points = [row[index] for index in active]
        point_mean = sum(projected_points) / len(projected_points)
        line_total = 0.0
        quadratic_total = 0.0
        line_count = 0
        for left in range(len(projected_points) - 1):
            for right in range(left + 1, len(projected_points)):
                distance = right - left
                line_total += (
                    projected_points[right] - projected_points[left]
                ) / distance
                quadratic_total += (
                    projected_points[left] * projected_points[right]
                ) / math.sqrt(distance)
                line_count += 1
        relation_response = (
            point_mean
            + line_total / line_count
            + quadratic_gain * quadratic_total / line_count
        )
        response = math.tanh(relation_response)
        if mode == "alignment":
            response = 1.0 - abs(response)
        features.append(response)
    for index in range(0, len(features), PROJECTION_CHANNELS):
        features[index] *= 1.0 + reinforcement_gain * features[index + 1]
    return _unit_vector(features)


def _unit_vector(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 1e-12:
        return [0.0] * len(values)
    return [value / norm for value in values]


def _fit_softmax_head(
    train_features: list[list[float]],
    train_targets: list[int],
    test_features: list[list[float]],
    test_targets: list[int],
    epochs: int,
    learning_rate: float,
) -> float:
    feature_count = len(train_features[0])
    weights = [[0.0] * feature_count for _ in range(3)]
    biases = [0.0, 0.0, 0.0]
    order = list(range(len(train_targets)))
    rng = random.Random(0x50F7AA)

    for epoch in range(epochs):
        rng.shuffle(order)
        rate = learning_rate / math.sqrt(epoch + 1)
        for sample_index in order:
            features = train_features[sample_index]
            target = train_targets[sample_index]
            logits = [
                biases[class_index] + sum(
                    weight * value
                    for weight, value in zip(weights[class_index], features)
                )
                for class_index in range(3)
            ]
            probabilities = _softmax(logits)
            for class_index in range(3):
                error = probabilities[class_index] - (1.0 if class_index == target else 0.0)
                class_weights = weights[class_index]
                for feature_index, value in enumerate(features):
                    class_weights[feature_index] -= rate * error * value
                biases[class_index] -= rate * error

    total_bits = 0.0
    for features, target in zip(test_features, test_targets):
        logits = [
            biases[class_index] + sum(
                weight * value
                for weight, value in zip(weights[class_index], features)
            )
            for class_index in range(3)
        ]
        probability = max(_softmax(logits)[target], 1e-15)
        total_bits += -math.log2(probability)
    return total_bits * TRITS_PER_BYTE / len(test_targets)


def _fit_softmax_head_probability_mixture(
    train_views: list[list[list[float]]],
    train_targets: list[int],
    test_views: list[list[list[float]]],
    test_targets: list[int],
    epochs: int,
    learning_rate: float,
) -> float:
    """Train one head through a mixture of fixed projection views."""
    if not train_views or len(train_views) != len(test_views):
        raise ValueError("probability mixture needs matching train/test views")
    feature_count = len(train_views[0][0])
    weights = [[0.0] * feature_count for _ in range(3)]
    biases = [0.0, 0.0, 0.0]
    order = list(range(len(train_targets)))
    rng = random.Random(0x50F7AA)

    for epoch in range(epochs):
        rng.shuffle(order)
        rate = learning_rate / math.sqrt(epoch + 1)
        for sample_index in order:
            target = train_targets[sample_index]
            view_features = [
                features[sample_index] for features in train_views
            ]
            view_probabilities = []
            for features in view_features:
                logits = [
                    biases[class_index] + sum(
                        weight * value
                        for weight, value in zip(weights[class_index], features)
                    )
                    for class_index in range(3)
                ]
                view_probabilities.append(_softmax(logits))

            target_probability_total = sum(
                probabilities[target]
                for probabilities in view_probabilities
            )
            mixture_weights = [
                probabilities[target] / target_probability_total
                for probabilities in view_probabilities
            ]
            for class_index in range(3):
                bias_gradient = 0.0
                feature_gradients = [0.0] * feature_count
                expected = 1.0 if class_index == target else 0.0
                for mixture_weight, probabilities, features in zip(
                    mixture_weights,
                    view_probabilities,
                    view_features,
                ):
                    error = mixture_weight * (
                        probabilities[class_index] - expected
                    )
                    bias_gradient += error
                    for feature_index, value in enumerate(features):
                        feature_gradients[feature_index] += error * value
                class_weights = weights[class_index]
                for feature_index, gradient in enumerate(feature_gradients):
                    class_weights[feature_index] -= rate * gradient
                biases[class_index] -= rate * bias_gradient

    total_bits = 0.0
    for sample_index, target in enumerate(test_targets):
        target_probabilities = []
        for view in test_views:
            features = view[sample_index]
            logits = [
                biases[class_index] + sum(
                    weight * value
                    for weight, value in zip(weights[class_index], features)
                )
                for class_index in range(3)
            ]
            target_probabilities.append(_softmax(logits)[target])
        probability = max(
            sum(target_probabilities) / len(target_probabilities),
            1e-15,
        )
        total_bits += -math.log2(probability)
    return total_bits * TRITS_PER_BYTE / len(test_targets)


def _softmax(logits: list[float]) -> list[float]:
    maximum = max(logits)
    exponentials = [math.exp(value - maximum) for value in logits]
    denominator = sum(exponentials)
    return [value / denominator for value in exponentials]


def _projection_score_summary(scores: list[float]) -> dict[str, Any]:
    mean = sum(scores) / len(scores)
    if len(scores) > 1:
        variance = sum((score - mean) ** 2 for score in scores) / (len(scores) - 1)
        sample_sd = math.sqrt(variance)
    else:
        sample_sd = 0.0
    return {
        "mean_bits_per_input_byte": round(mean, 6),
        "sample_sd": round(sample_sd, 6),
        "scores": [round(score, 6) for score in scores],
        "feature_count": PROJECTION_FEATURES,
        "trainable_parameters": PROJECTION_PARAMETERS,
    }


def _projection_comparison(
    candidate_scores: list[float],
    comparator_scores: list[float],
) -> dict[str, Any]:
    candidate_mean = sum(candidate_scores) / len(candidate_scores)
    comparator_mean = sum(comparator_scores) / len(comparator_scores)
    candidate_sd = _sample_sd(candidate_scores)
    comparator_sd = _sample_sd(comparator_scores)
    pooled_sd = math.sqrt((candidate_sd ** 2 + comparator_sd ** 2) / 2.0)
    improvement = comparator_mean - candidate_mean
    threshold = 2.0 * pooled_sd
    return {
        "candidate_minus_comparator_bits_per_input_byte": round(
            candidate_mean - comparator_mean, 6
        ),
        "improvement_bits_per_input_byte": round(improvement, 6),
        "pooled_sd": round(pooled_sd, 6),
        "required_improvement_2x_pooled_sd": round(threshold, 6),
        "passes_2x_pooled_sd": improvement > threshold,
        "paired_deltas_candidate_minus_comparator": [
            round(candidate - comparator, 6)
            for candidate, comparator in zip(candidate_scores, comparator_scores)
        ],
    }


def _projection_claim_verdict(
    versus_traditional: dict[str, Any],
    versus_scrambled: dict[str, Any],
    seed_count: int,
    candidate_label: str,
) -> tuple[str, str]:
    mean_lift_on_both = (
        versus_traditional["improvement_bits_per_input_byte"] > 0
        and versus_scrambled["improvement_bits_per_input_byte"] > 0
    )
    if seed_count < 3:
        return "UNDERPOWERED", "Fewer than three paired seeds were run."
    if (
        versus_traditional["passes_2x_pooled_sd"]
        and versus_scrambled["passes_2x_pooled_sd"]
    ):
        return (
            "SUPPORTED",
            f"{candidate_label} beats both controls by more than 2x pooled SD.",
        )
    if mean_lift_on_both:
        return (
            "UNDERPOWERED",
            f"{candidate_label} has positive mean lift against both controls "
            "but misses at least one 2x pooled-SD gate.",
        )
    return (
        "NO_LIFT",
        f"{candidate_label} does not beat both controls on held-out data.",
    )


def _sample_sd(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(
        sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    )


def format_projection_depth_report(report: dict[str, Any]) -> str:
    model_order = report.get("model_order", list(report["models"]))
    name_width = max(31, max(len(name) for name in model_order))
    out = [
        "aperiodic relational projection benchmark",
        (
            f"input_bytes={report['input_bytes']} "
            f"sha256={report['input_sha256'][:12]} "
            f"split={report['causal_split']['fraction']:.2f} "
            f"train_trits={report['causal_split']['train_trit_samples']} "
            f"test_trits={report['causal_split']['test_trit_samples']}"
        ),
        (
            f"features={report['architecture']['feature_count']} "
            f"trainable_parameters={report['architecture']['trainable_parameters_per_model']} "
            f"seeds={','.join(str(seed) for seed in report['training']['projection_seeds'])}"
        ),
        "",
        f"{'model':<{name_width}}  mean bits/byte  sample sd  seed scores",
        f"{'-' * name_width}  --------------  ---------  ---------------------------",
    ]
    for name in model_order:
        model = report["models"][name]
        scores = ", ".join(f"{score:.6f}" for score in model["scores"])
        out.append(
            f"{name:<{name_width}}  {model['mean_bits_per_input_byte']:>14.6f}  "
            f"{model['sample_sd']:>9.6f}  {scores}"
        )

    out.extend(["", "original aligned claim gates"])
    for label, key in (
        ("vs traditional", "candidate_vs_traditional"),
        ("vs scrambled", "candidate_vs_scrambled"),
    ):
        comparison = report["comparisons"][key]
        out.append(
            f"{label}: improvement={comparison['improvement_bits_per_input_byte']:.6f} "
            f"required={comparison['required_improvement_2x_pooled_sd']:.6f} "
            f"pass={comparison['passes_2x_pooled_sd']}"
        )
    if report.get("ablation_comparisons"):
        out.extend(["", "component ablations"])
        for label, comparison in report["ablation_comparisons"].items():
            out.append(
                f"{label}: improvement={comparison['improvement_bits_per_input_byte']:.6f} "
                f"required={comparison['required_improvement_2x_pooled_sd']:.6f} "
                f"pass={comparison['passes_2x_pooled_sd']}"
            )
    synthesis = report.get("synthesis")
    if synthesis:
        out.extend(["", "quasi-random synthesis claim gates"])
        for label, key in (
            ("vs traditional", "candidate_vs_traditional"),
            ("vs scrambled", "candidate_vs_scrambled"),
        ):
            comparison = synthesis["comparisons"][key]
            out.append(
                f"{label}: improvement={comparison['improvement_bits_per_input_byte']:.6f} "
                f"required={comparison['required_improvement_2x_pooled_sd']:.6f} "
                f"pass={comparison['passes_2x_pooled_sd']}"
            )
        out.extend([
            f"synthesis verdict: {synthesis['verdict']}",
            f"synthesis reason: {synthesis['verdict_reason']}",
        ])
    layered_synthesis = report.get("layered_synthesis")
    if layered_synthesis:
        layer_details = (
            f"{layered_synthesis['copies']} fixed copies, "
            f"mode={layered_synthesis['mode']}"
        )
        if layered_synthesis["mode"] == "feature":
            layer_details += (
                f", decay={layered_synthesis['copy_weight_decay']}"
            )
        out.extend([
            "",
            f"layered quasi-random synthesis ({layer_details})",
        ])
        for label, key in (
            ("vs traditional", "candidate_vs_traditional"),
            ("vs layered scrambled", "candidate_vs_layered_scrambled"),
            ("vs single quasi-random", "candidate_vs_single_quasi_random"),
            (
                "layered scrambled vs single scrambled",
                "layered_scrambled_vs_single_scrambled",
            ),
        ):
            comparison = layered_synthesis["comparisons"][key]
            out.append(
                f"{label}: improvement={comparison['improvement_bits_per_input_byte']:.6f} "
                f"required={comparison['required_improvement_2x_pooled_sd']:.6f} "
                f"pass={comparison['passes_2x_pooled_sd']}"
            )
        out.extend([
            f"layered verdict: {layered_synthesis['verdict']}",
            f"layered reason: {layered_synthesis['verdict_reason']}",
        ])
    out.extend([
        f"original aligned verdict: {report['verdict']}",
        f"original aligned reason: {report['verdict_reason']}",
        f"note: {report['note']}",
    ])
    return "\n".join(out)


def _byte_to_trits(byte: int) -> list[int]:
    return [int(digit) for digit in _base_digits(byte, 3, TRITS_PER_BYTE)]


def bytes_to_trit_symbols(data: bytes) -> list[int]:
    symbols: list[int] = []
    for byte in data:
        symbols.extend(_byte_to_trits(byte))
    return symbols


def _trit_votes(trits: str) -> str:
    return ",".join(str(TRIT_LIGHT_CLASSES[digit]["vote"]) for digit in trits)


def _trit_class_names(trits: str) -> str:
    return "/".join(TRIT_LIGHT_CLASSES[digit]["class"] for digit in trits)


def format_prediction_compression_report(report: dict[str, Any]) -> str:
    def add_orders(out: list[str], rows: list[dict[str, Any]]) -> None:
        out.append("order  contexts  bits/input_byte  bits/symbol  ideal_bytes")
        out.append("-----  --------  ---------------  -----------  -----------")
        for row in rows:
            out.append(
                f"{row['order']:>5}  {row['contexts']:>8}  "
                f"{row['ideal_bits_per_input_byte']:>15.6f}  "
                f"{row['ideal_bits_per_symbol']:>11.6f}  "
                f"{row['ideal_bytes']:>11}"
            )

    byte = report["traditional_next_byte"]
    trit = report["trit_light_projection"]
    comparison = report["comparison"]
    out = [
        f"input_bytes: {report['input_bytes']}",
        f"smoothing_alpha: {report['smoothing_alpha']}",
        "",
        "traditional next-byte predictor",
    ]
    add_orders(out, byte["orders"])
    out.extend([
        f"best_order: {byte['best']['order']}",
        "",
        "trit light-class predictor",
        f"trits_per_input_byte: {trit['trits_per_input_byte']}",
        "class_mapping: 0=blocker(-1), 1=reader(0), 2=writer(+1)",
    ])
    add_orders(out, trit["orders"])
    out.extend([
        f"best_order: {trit['best']['order']}",
        "",
        "comparison",
        f"traditional_bits_per_input_byte: {comparison['traditional_bits_per_input_byte']}",
        f"trit_bits_per_input_byte: {comparison['trit_bits_per_input_byte']}",
        (
            "delta_trit_minus_traditional_bits_per_input_byte: "
            f"{comparison['delta_trit_minus_traditional_bits_per_input_byte']}"
        ),
        f"winner: {comparison['winner']}",
        f"note: {report['note']}",
    ])
    return "\n".join(out)


def _base_digits(index: int, base: int, width: int) -> str:
    digits = []
    n = index
    for _ in range(width):
        digits.append(str(n % base))
        n //= base
    return "".join(reversed(digits))


def trit_prefix_rows(depth: int = 2) -> list[dict[str, Any]]:
    """Every ternary prefix up to depth, with information shown in binary bits."""
    if depth < 1 or depth > 6:
        raise ValueError("depth must be between 1 and 6")
    rows: list[dict[str, Any]] = []
    for level in range(1, depth + 1):
        width = 3 ** level
        ideal_bits = level * math.log2(3)
        for index in range(width):
            trits = _base_digits(index, 3, level)
            rows.append({
                "depth": level,
                "index": index,
                "trits": trits,
                "class_votes": _trit_votes(trits),
                "classes": _trit_class_names(trits),
                "prefix_space_fraction": f"1/{width}",
                "ideal_binary_bits": round(ideal_bits, 6),
            })
    return rows


def trit_binary_overlay_rows(trit_depth: int = 2, binary_depth: int = 4) -> list[dict[str, Any]]:
    """Overlay ternary prefix intervals onto a fixed-depth binary prefix table."""
    if trit_depth < 1 or trit_depth > 6:
        raise ValueError("trit_depth must be between 1 and 6")
    if binary_depth < 1 or binary_depth > 12:
        raise ValueError("binary_depth must be between 1 and 12")

    trit_width = 3 ** trit_depth
    binary_width = 2 ** binary_depth
    rows: list[dict[str, Any]] = []
    for index in range(trit_width):
        start = index / trit_width
        end = (index + 1) / trit_width
        first_binary = int(math.floor(start * binary_width))
        last_binary = int(math.ceil(end * binary_width)) - 1
        last_binary = min(last_binary, binary_width - 1)
        if first_binary == last_binary:
            binary_span = format(first_binary, f"0{binary_depth}b")
        else:
            binary_span = (
                f"{format(first_binary, f'0{binary_depth}b')}.."
                f"{format(last_binary, f'0{binary_depth}b')}"
            )
        rows.append({
            "trits": _base_digits(index, 3, trit_depth),
            "trit_index": index,
            "unit_interval_start": round(start, 8),
            "unit_interval_end": round(end, 8),
            "ideal_binary_bits": round(trit_depth * math.log2(3), 6),
            "binary_depth": binary_depth,
            "binary_span": binary_span,
            "binary_cells_touched": last_binary - first_binary + 1,
        })
        rows[-1]["class_votes"] = _trit_votes(rows[-1]["trits"])
        rows[-1]["classes"] = _trit_class_names(rows[-1]["trits"])
    return rows


def format_trit_binary_overlay(trit_depth: int = 2, binary_depth: int = 4) -> str:
    rows = trit_binary_overlay_rows(trit_depth, binary_depth)
    out = [
        "trit table over binary prefix table",
        f"trit_depth={trit_depth} binary_depth={binary_depth}",
        "trits  votes    classes            ideal_bits  binary_span       cells",
        "-----  -------  -----------------  ----------  ----------------  -----",
    ]
    for row in rows:
        out.append(
            f"{row['trits']:<5}  {row['class_votes']:<7}  {row['classes']:<17}  "
            f"{row['ideal_binary_bits']:>10.6f}  "
            f"{row['binary_span']:<16}  {row['binary_cells_touched']:>5}"
        )
    return "\n".join(out)


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

    # Entropy report should make the byte-preserving/token-surface trade visible
    try:
        report = entropy_report(b"aaaaabbbbbcc", "KO")
        if report["input_bytes"] != 12:
            errors.append("entropy report input byte count mismatch")
        if report["token_surface_bytes"] <= report["input_bytes"]:
            errors.append("entropy report did not show token surface overhead")
        if report["shannon_lower_bound_bytes"] <= 0:
            errors.append("entropy report lower bound missing")
    except Exception as e:
        errors.append(f"entropy report error: {e}")

    # Prediction compression: compare a traditional byte predictor to trit classes
    try:
        report = prediction_compression_report(b"abababababababab", max_order=2, trit_max_order=6)
        if report["traditional_next_byte"]["baseline_order_0"]["order"] != 0:
            errors.append("prediction report missing traditional order-0 baseline")
        if report["traditional_next_byte"]["best"]["ideal_bits_per_input_byte"] <= 0:
            errors.append("prediction report missing byte predictor score")
        if report["trit_light_projection"]["best"]["ideal_bits_per_input_byte"] <= 0:
            errors.append("prediction report missing trit predictor score")
        if report["comparison"]["winner"] not in ("traditional_next_byte", "trit_light_projection", "tie"):
            errors.append("prediction report winner mismatch")
    except Exception as e:
        errors.append(f"prediction compression error: {e}")

    # Projection depth: equal-budget models, causal split, and three paired seeds
    try:
        projection_sample = b"abc cab bac " * 16
        report = projection_depth_report(
            projection_sample,
            epochs=2,
            split_fraction=0.75,
            max_train_samples=300,
            max_test_samples=96,
            include_ablations=True,
            layered_copies=2,
        )
        if len(report["runs"]) != 3:
            errors.append("projection benchmark did not run three seeds")
        if report["input_sha256"] != hashlib.sha256(projection_sample).hexdigest():
            errors.append("projection benchmark corpus hash mismatch")
        if report["causal_split"]["fraction"] != 0.75:
            errors.append("projection benchmark split fraction mismatch")
        if report["architecture"]["feature_count"] != 48:
            errors.append("projection benchmark feature budget mismatch")
        if report["architecture"]["trainable_parameters_per_model"] != 147:
            errors.append("projection benchmark parameter budget mismatch")
        expected_ablation_models = {
            "quasi_random_projection",
            "dyadic_scale_projection",
            "rational_angle_projection",
            "periodic_rational_dyadic_projection",
        }
        if not expected_ablation_models.issubset(report["models"]):
            errors.append("projection benchmark missing factorial ablation models")
        expected_layered_models = {
            "layered_probability_scrambled_projection_control",
            "layered_probability_quasi_random_projection",
        }
        if not expected_layered_models.issubset(report["models"]):
            errors.append("projection benchmark missing layered models")
        if len(report["ablation_comparisons"]) != 5:
            errors.append("projection benchmark ablation comparison mismatch")
        synthesis = report.get("synthesis", {})
        if synthesis.get("candidate") != "quasi_random_projection":
            errors.append("projection benchmark synthesis candidate mismatch")
        if set(synthesis.get("comparisons", {})) != {
            "candidate_vs_traditional",
            "candidate_vs_scrambled",
        }:
            errors.append("projection benchmark synthesis comparison mismatch")
        if synthesis.get("verdict") not in ("SUPPORTED", "UNDERPOWERED", "NO_LIFT"):
            errors.append("projection benchmark synthesis verdict mismatch")
        layered_synthesis = report.get("layered_synthesis", {})
        if layered_synthesis.get("copies") != 2:
            errors.append("projection benchmark layered copy count mismatch")
        if layered_synthesis.get("copy_weight_decay") != 0.5:
            errors.append("projection benchmark layered decay mismatch")
        if layered_synthesis.get("mode") != "probability":
            errors.append("projection benchmark layered mode mismatch")
        if layered_synthesis.get("verdict") not in (
            "SUPPORTED",
            "UNDERPOWERED",
            "NO_LIFT",
        ):
            errors.append("projection benchmark layered verdict mismatch")
        if report["verdict"] not in ("SUPPORTED", "UNDERPOWERED", "NO_LIFT"):
            errors.append("projection benchmark verdict mismatch")
        for model in report["models"].values():
            if not math.isfinite(model["mean_bits_per_input_byte"]):
                errors.append("projection benchmark emitted a non-finite score")
        symbols = bytes_to_trit_symbols(projection_sample)
        position = 24
        altered = symbols[:]
        altered[position:] = [(symbol + 1) % 3 for symbol in altered[position:]]
        original_features = _projection_feature_rows(symbols, [position], "aligned", 7)
        altered_features = _projection_feature_rows(altered, [position], "aligned", 7)
        if original_features != altered_features:
            errors.append("projection benchmark leaked target or future symbols")
    except Exception as e:
        errors.append(f"projection benchmark error: {e}")

    # Trit overlay: ternary classes projected onto binary prefix code space
    try:
        rows = trit_binary_overlay_rows(2, 4)
        if len(rows) != 9:
            errors.append("trit overlay row count mismatch")
        if rows[0]["trits"] != "00" or rows[-1]["trits"] != "22":
            errors.append("trit overlay ordering mismatch")
        if rows[0]["class_votes"] != "-1,-1" or rows[-1]["classes"] != "writer/writer":
            errors.append("trit overlay class mapping mismatch")
    except Exception as e:
        errors.append(f"trit overlay error: {e}")

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
    print(f"  encode/decode/xlate/blend/unblend/geoseal/prediction/projection all pass")
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


def _streams_command(args) -> int:
    """Print the operation tube for one board.

    Tongue streams are the six roads. Newton coefficients are the tangent
    along operation order. Word stamps and the GeoSeal envelope are attached
    only when the caller asks for them.
    """
    import newton_tube

    if args.hex and args.text:
        print("error: pass --hex or --text, not both", file=sys.stderr)
        return 2
    if args.hex:
        try:
            board = bytes.fromhex(args.hex)
        except ValueError as exc:
            print(f"error: bad hex -- {exc}", file=sys.stderr)
            return 2
    else:
        board = _read_text(args)
        if not board:
            print("error: no input (use --text, --hex, or pipe)", file=sys.stderr)
            return 2

    tongues = {tongue: encode(board, tongue) for tongue in TONGUES}
    atomic = None
    if args.words is not None:
        words = [word.strip() for word in args.words.split(",") if word.strip()]
        if not words:
            print("error: --words needs at least one token", file=sys.stderr)
            return 2
        atomic = newton_tube.stamp_words(words)
    seal = None
    if args.seal:
        if args.lat is None or args.lon is None:
            print("error: --seal needs --lat and --lon", file=sys.stderr)
            return 2
        seal = geoseal_encrypt(board, args.lat, args.lon, tag=args.tag)
    print(json.dumps(newton_tube.stream_record(board, tongues=tongues, atomic=atomic, seal=seal), indent=2))
    return 0


def main() -> int:
    # ``clay.train`` owns a richer conversational argument surface.  Dispatch
    # before this module's parser so options such as ``--home`` and ``--model``
    # pass through unchanged instead of being mistaken for GeoSeal options.
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        from clay_train_cli import main as clay_train_main

        return clay_train_main(sys.argv[2:])

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

    p_ent = sub.add_parser("entropy", help="Measure byte entropy and tokenizer surface overhead")
    p_ent.add_argument("--tongue", default="KO", help="Tongue used for token surface measurement")
    p_ent.add_argument("--text")
    p_ent.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    p_pc = sub.add_parser(
        "predict-compress",
        help="Compare traditional next-byte prediction to trit-class prediction",
    )
    p_pc.add_argument("--max-order", type=int, default=3, help="Max byte-context order, 0..8")
    p_pc.add_argument(
        "--trit-max-order",
        type=int,
        help="Max trit-context order, 0..48; defaults to max_order * 6",
    )
    p_pc.add_argument("--alpha", type=float, default=0.5, help="Additive smoothing strength")
    p_pc.add_argument("--text")
    p_pc.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    p_pb = sub.add_parser(
        "projection-benchmark",
        help="Test aperiodic relational projection against equal-budget next-token controls",
    )
    p_pb.add_argument("--epochs", type=int, default=8, help="Softmax training epochs, 1..100")
    p_pb.add_argument("--learning-rate", type=float, default=0.12)
    p_pb.add_argument(
        "--projection-gain",
        type=float,
        default=1.0,
        help="Projection branch strength, in (0, 8]",
    )
    p_pb.add_argument(
        "--layer-gain",
        type=float,
        default=1.0,
        help="Recurrent backdrop reinforcement, in [0, 4]",
    )
    p_pb.add_argument(
        "--quadratic-gain",
        type=float,
        default=0.0,
        help="Second-order pair-coherence strength, in [0, 8]",
    )
    p_pb.add_argument(
        "--reinforcement-gain",
        type=float,
        default=0.0,
        help="Multiplicative self-alignment signal gain, in [0, 8]",
    )
    p_pb.add_argument(
        "--split-fraction",
        type=float,
        default=0.8,
        help="Causal train/test boundary as a corpus fraction, in [0.1, 0.9]",
    )
    p_pb.add_argument("--max-train-samples", type=int, default=6000)
    p_pb.add_argument("--max-test-samples", type=int, default=1500)
    p_pb.add_argument(
        "--ablations",
        action="store_true",
        help="Run matched angle/scale/depth-phase component ablations",
    )
    p_pb.add_argument(
        "--layered-copies",
        type=int,
        default=1,
        help="Average 2..16 fixed projection arrays; requires --ablations",
    )
    p_pb.add_argument(
        "--layered-decay",
        type=float,
        default=0.5,
        help="Feature-mode copy-weight ratio in [0, 1]; default 0.5",
    )
    p_pb.add_argument(
        "--layered-mode",
        choices=("feature", "probability"),
        default="probability",
        help="Combine fixed projections before or after the shared softmax",
    )
    p_pb.add_argument("--text")
    p_pb.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    p_to = sub.add_parser("trit-overlay", help="Overlay ternary light classes on binary code space")
    p_to.add_argument("--trit-depth", type=int, default=2, help="Ternary prefix depth, 1..6")
    p_to.add_argument("--binary-depth", type=int, default=4, help="Binary prefix depth, 1..12")
    p_to.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    p_streams = sub.add_parser(
        "streams",
        help="Read one byte board as tongues, a Newton tangent, and optional atomic and seal streams",
    )
    p_streams.add_argument("--text", help="UTF-8 text whose bytes are the board")
    p_streams.add_argument("--hex", help="Even-length hex of the board bytes")
    p_streams.add_argument(
        "--words",
        help="Comma-separated linguistic tokens. Destination stamps, not Newton samples.",
    )
    p_streams.add_argument("--seal", action="store_true", help="Bind the board bytes with GeoSeal")
    p_streams.add_argument("--lat", type=float)
    p_streams.add_argument("--lon", type=float)
    p_streams.add_argument("--tag", default="")

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

    p_pg = sub.add_parser("polyglot", help="Represent a concept across real languages (Python/JS/Rust/Haskell/C++/Julia); --run to execute + cross-check")
    p_pg.add_argument("op", nargs="?", help="op-class to represent, or 'list'")
    p_pg.add_argument("--run", action="store_true", help="execute available toolchains and show which languages agree")

    p_train = sub.add_parser(
        "train",
        help="Open Clay's local conversational training studio",
    )
    p_train.add_argument(
        "train_args",
        nargs=argparse.REMAINDER,
        help="arguments passed through to clay.train",
    )

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

    if args.cmd == "entropy":
        data = _read_text(args)
        if not data:
            print("error: no input (use --text or pipe)", file=sys.stderr)
            return 2
        try:
            report = entropy_report(data, args.tongue)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print(f"input_bytes: {report['input_bytes']}")
            print(f"unique_bytes: {report['unique_bytes']}")
            print(f"byte_entropy_bits_per_byte: {report['byte_entropy_bits_per_byte']}")
            print(f"shannon_lower_bound_bytes: {report['shannon_lower_bound_bytes']}")
            print(f"raw_zlib_bytes: {report['raw_zlib_bytes']}")
            print(f"tongue: {report['tongue']}")
            print(f"token_surface_bytes: {report['token_surface_bytes']}")
            print(f"token_surface_ratio: {report['token_surface_ratio']}")
            print(f"token_zlib_bytes: {report['token_zlib_bytes']}")
            print(f"token_zlib_ratio: {report['token_zlib_ratio']}")
            print(f"note: {report['note']}")
        return 0

    if args.cmd == "predict-compress":
        data = _read_text(args)
        if not data:
            print("error: no input (use --text or pipe)", file=sys.stderr)
            return 2
        try:
            report = prediction_compression_report(
                data,
                max_order=args.max_order,
                alpha=args.alpha,
                trit_max_order=args.trit_max_order,
            )
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print(format_prediction_compression_report(report))
        return 0

    if args.cmd == "projection-benchmark":
        data = _read_text(args)
        if not data:
            print("error: no input (use --text or pipe)", file=sys.stderr)
            return 2
        try:
            report = projection_depth_report(
                data,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                projection_gain=args.projection_gain,
                layer_gain=args.layer_gain,
                quadratic_gain=args.quadratic_gain,
                reinforcement_gain=args.reinforcement_gain,
                split_fraction=args.split_fraction,
                max_train_samples=args.max_train_samples,
                max_test_samples=args.max_test_samples,
                include_ablations=args.ablations,
                layered_copies=args.layered_copies,
                layered_decay=args.layered_decay,
                layered_mode=args.layered_mode,
            )
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print(format_projection_depth_report(report))
        return 0

    if args.cmd == "trit-overlay":
        try:
            if args.json:
                print(json.dumps(trit_binary_overlay_rows(args.trit_depth, args.binary_depth), indent=2))
            else:
                print(format_trit_binary_overlay(args.trit_depth, args.binary_depth))
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        return 0

    if args.cmd == "streams":
        return _streams_command(args)

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

    if args.cmd == "polyglot":
        import polyglot as pg
        if not args.op or args.op == "list":
            print("op-classes:", ", ".join(pg.OPS))
            print("langs:     ", ", ".join(
                f"{l}{'' if pg.available(l) else '(represent-only)'}" for l in pg.LANGS))
            print("\nusage: aethermoore.py polyglot <op> [--run]")
            return 0
        if args.op not in pg.OPS:
            print(f"error: unknown op-class '{args.op}'. choose from: {', '.join(pg.OPS)}", file=sys.stderr)
            return 2
        reps = pg.represent(args.op)
        print(f"# concept '{args.op}' represented across languages\n")
        for lang in pg.LANGS:
            src = reps[lang].replace("\n", "\n    ")
            print(f"[{lang}]{'' if pg.available(lang) else '  (represent-only: no toolchain)'}\n    {src}\n")
        if args.run:
            print("# execution cross-check (run each present toolchain):")
            values = {lang: pg.run_lang(lang, reps[lang]) for lang in pg.LANGS}
            for lang in pg.LANGS:
                print(f"  {lang:11} = {values[lang]}")
            ran = {l: v for l, v in values.items()
                   if v not in ("toolchain-not-found", "COMPILE-ERR") and not v.startswith("ERR:")}
            cs = pg.camps(ran)
            print("\n  camps (agreement first):")
            for val, langs in cs:
                print(f"    {val!r:>14} : {', '.join(langs)}")
            if len(cs) == 1 and ran:
                print("\n  -> all runnable languages AGREE: this concept is language-independent.")
            elif len(cs) > 1:
                print("\n  -> languages SPLIT into camps: a real semantic difference, surfaced by the box.")
        return 0

    if args.cmd == "train":
        from clay_train_cli import main as clay_train_main

        return clay_train_main(args.train_args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
