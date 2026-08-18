#!/usr/bin/env python3
"""Conversational, local-first training manager for Clay.

``clay.train`` is deliberately a manager, not a second model framework.  It
turns a conversation into sealed training artifacts and invokes Clay's existing
multi-seed trainers with exact argv lists (never generated shell text).

The lifecycle borrows a useful resource invariant from StarCraft's Zerg:

* a hatchery is a persistent training workspace;
* larva is uncommitted candidate capacity;
* a successful candidate matures into a drone;
* a drone may become another hatchery only while one spare drone remains.

Those are scheduling names.  The measurements still come from the real lane
graders, and a hill step is accepted only when it clears a multi-seed noise
threshold against the centre candidate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from aethermoore import TONGUES, decode, encode, geoseal_decrypt, geoseal_encrypt


VERSION = 1
DEFAULT_HOME = Path(os.environ.get("CLAY_TRAIN_HOME", Path.home() / ".clay" / "training-studio"))
LOCAL_TRAIN = Path(r"C:\dev\clay-quines\home\local_train\local_train.py")
LOCAL_TRAIN_ROOT = LOCAL_TRAIN.parents[2]
LOCAL_RUNS = LOCAL_TRAIN.parent / "runs"
DIALOGUE_QUINE = Path(r"C:\dev\clay-quines\home\ingest\dialogue_quine\dialogue_quine.py")
OPCODE_GRAPH = Path(r"C:\dev\clay-quines\clay-math\clay_math\charged_opcode_graph.py")
OPCODE_QUINE = Path(r"C:\dev\clay-quines\clay-math\clay_math\opcode_quine_cube.py")
OPCODE_HATCHERY = Path(r"C:\dev\clay-quines\clay-math\clay_math\opcode_hatchery.py")
OPCODE_INTERNALIZATION = Path(
    r"C:\dev\clay-quines\clay-math\clay_math\representation_internalization.py"
)
ENGLISH_BLOCK_TRAINING = Path(
    r"C:\dev\clay-quines\clay-math\clay_math\english_block_training.py"
)
FOURIER_FLASHCARD_TRAINING = Path(
    r"C:\dev\clay-quines\clay-math\clay_math\fourier_flashcard_training.py"
)
WORD_EXPLANATION_TRAINING = Path(
    r"C:\dev\clay-quines\clay-math\clay_math\word_explanation_training.py"
)
OPCODE_ROOT = OPCODE_GRAPH.parents[1]
KAGGRICULTURE_ROOT = Path(r"C:\dev\kaggriculture")
KAGGRICULTURE_COMPARE = KAGGRICULTURE_ROOT / "compare_agents.py"
KAGGRICULTURE_TARGET_SOURCE = KAGGRICULTURE_ROOT / "main_profit.py"
KAGGRICULTURE_PROFILE = KAGGRICULTURE_ROOT / "replay_behavior_profile.py"
KAGGRICULTURE_GOLD_DIR = KAGGRICULTURE_ROOT / "gold_replays_20260806"
KAGGRICULTURE_GOLD_CANDIDATE = KAGGRICULTURE_ROOT / "main_goldroute_flat.py"
KAGGRICULTURE_CONFORMANCE = KAGGRICULTURE_GOLD_DIR / "goldroute_action_conformance.json"
KAGGRICULTURE_TARGET = {
    "competition": "kaggriculture",
    "submission_id": 55283743,
    "file": "main_profit.py",
    "direction": "higher-is-better",
    "score_contract": "read target and candidate from the same authenticated Kaggle CLI snapshot",
}

# Measured 2026-08-05 across C:\dev plus the profile SCBE tree.  These are
# reference-presence seams, not proof that every cited implementation executes.
# The important boundary is structural: STISA stays in its owning SCBE lane;
# Clay consumes it through an explicit anchor instead of cloning the tokenizer.
REPRESENTATION_TOPOLOGY = {
    "schema": "clay.representation-topology.v1",
    "measured_date": "2026-08-05",
    "families": ["tongues", "trits", "cube", "opcode", "bytes", "hf-bpe", "STISA"],
    "census": {"lanes": 25, "translation_points": 12, "farm_lanes_with_five_or_more": 4},
    "farm_translation_lanes": {
        r"C:\dev\clay-one-repo": ["tongues", "trits", "cube", "opcode", "bytes", "hf-bpe"],
        r"C:\dev\clay-quines": ["tongues", "trits", "cube", "opcode", "bytes", "hf-bpe"],
        r"C:\dev\clay_home": ["tongues", "trits", "cube", "opcode", "bytes", "hf-bpe"],
        r"C:\dev\clay-wafer-net": ["tongues", "trits", "cube", "opcode", "bytes"],
    },
    "external_anchors": {
        "STISA": r"C:\Users\issda\SCBE-AETHERMOORE\python\scbe\atomic_tokenization.py",
    },
    "claim_boundary": "file references map vocabulary co-location, not runtime dependency or learned use",
}

LANES = {
    "geodesic_routing_phdm",
    "matrix",
    "multi_walk",
    "asym_mirror_routing",
    "cross_plane_relations",
    "spectral_radial_alignment",
    "braid_loom",
    "packet_routing",
    "inverse_routing",
    "dynamic_braid",
    "sufficiency_gate",
}

# This registry is intentionally honest about the boundary between runnable
# trainers, executable diagnostics, and architectural ideas.  A word is not
# promoted into an implementation merely because the analogy sounds right.
GEOMETRY_BINDINGS: dict[str, dict[str, str]] = {
    "curve": {
        "status": "trainable",
        "lane": "cross_plane_relations",
        "path": r"C:\dev\clay-quines\home\ingest\cross_plane_relations",
    },
    "interweave": {
        "status": "trainable",
        "lane": "dynamic_braid",
        "path": r"C:\dev\clay-quines\home\ingest\dynamic_braid",
    },
    "twist": {
        "status": "trainable+diagnostic",
        "lane": "dynamic_braid",
        "path": r"C:\dev\clay-wafer-net\clay_wafer_net\ribbon_twist.py",
    },
    "invert": {
        "status": "trainable",
        "lane": "inverse_routing",
        "path": r"C:\dev\clay-quines\home\ingest\inverse_routing",
    },
    "reverse": {
        "status": "trainable",
        "lane": "inverse_routing",
        "path": r"C:\dev\clay-quines\home\ingest\inverse_routing",
    },
    "dual": {
        "status": "executable-diagnostic",
        "lane": "asym_mirror_routing",
        "path": r"C:\dev\clay-wafer-net\clay_wafer_net\dual_channel.py",
    },
    "palindrome": {
        "status": "executable-codec",
        "lane": "asym_mirror_routing",
        "path": str(Path(__file__).resolve()),
    },
    "copy": {
        "status": "executable-lifecycle",
        "lane": "geodesic_routing_phdm",
        "path": r"C:\dev\clay-quines\quines\agent\clay_wave_agent.py",
    },
    "replay": {
        "status": "executable-journal",
        "lane": "geodesic_routing_phdm",
        "path": str(LOCAL_TRAIN),
    },
    "stretch": {
        "status": "design-only",
        "lane": "cross_plane_relations",
        "path": r"C:\dev\clay-wafer-net\NEURAL_STRIP_MINE_LAYER_MAP.md",
    },
    "wafer": {
        "status": "executable-diagnostic",
        "lane": "geodesic_routing_phdm",
        "path": r"C:\dev\clay-wafer-net\test_wafer_phase_conjugate.py",
    },
}

SECRET_PATTERNS = (
    re.compile(r"(?i)\b(api[_-]?key|access[_-]?token|secret|password)\b\s*[:=]\s*[^\s,;]+"),
    re.compile(r"\b(?:sk-[A-Za-z0-9_-]{12,}|ghp_[A-Za-z0-9]{12,}|KGAT_[A-Za-z0-9]{12,})\b"),
    re.compile(r"-----BEGIN [^-]+PRIVATE KEY-----.*?-----END [^-]+PRIVATE KEY-----", re.S),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def redact(text: str) -> str:
    out = text
    for pattern in SECRET_PATTERNS:
        out = pattern.sub("[REDACTED]", out)
    return out


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def representation_topology_status() -> dict[str, Any]:
    """Return the measured seam map with live path/hash checks."""

    topology = json.loads(json.dumps(REPRESENTATION_TOPOLOGY))
    topology["farm_translation_lanes"] = {
        path: {"families": families, "exists": Path(path).is_dir()}
        for path, families in REPRESENTATION_TOPOLOGY["farm_translation_lanes"].items()
    }
    topology["external_anchors"] = {}
    for name, value in REPRESENTATION_TOPOLOGY["external_anchors"].items():
        path = Path(value)
        topology["external_anchors"][name] = {
            "path": str(path),
            "exists": path.is_file(),
            "sha256": file_sha256(path) if path.is_file() else None,
            "ownership": "external-reference; do not copy into Clay",
        }
    return topology


def alternating_offsets(count: int) -> list[int]:
    """Return 0,+1,-1,+2,-2... so every hill probe has a paired reverse."""
    result = [0]
    magnitude = 1
    while len(result) < count:
        result.append(magnitude)
        if len(result) < count:
            result.append(-magnitude)
        magnitude += 1
    return result


def split_cli_line(line: str) -> list[str]:
    """Split one interactive command without eating Windows path separators."""
    if os.name != "nt":
        return shlex.split(line)
    parts = shlex.split(line, posix=False)
    cleaned = []
    for part in parts:
        if len(part) >= 2 and part[0] == part[-1] and part[0] in {'"', "'"}:
            part = part[1:-1]
        cleaned.append(part)
    return cleaned


def parse_kaggle_submission_snapshot(payload: str) -> list[dict[str, Any]]:
    """Parse one verbose Kaggle CLI CSV response into a narrow score snapshot."""

    rows: list[dict[str, Any]] = []
    reader = csv.DictReader(io.StringIO(payload.lstrip("\ufeff")))
    required = {"ref", "fileName", "date", "status", "publicScore"}
    if reader.fieldnames is None or not required.issubset(reader.fieldnames):
        raise ValueError("Kaggle submission output is missing required CSV columns")
    for raw in reader:
        ref = str(raw.get("ref") or "").strip()
        if not ref:
            continue
        score_text = str(raw.get("publicScore") or "").strip()
        try:
            score = float(score_text) if score_text else None
        except ValueError:
            score = None
        rows.append({
            "submission_id": int(ref),
            "file": str(raw.get("fileName") or "").strip(),
            "submitted_at": str(raw.get("date") or "").strip(),
            "status": str(raw.get("status") or "").removeprefix("SubmissionStatus."),
            "public_score": score,
        })
    return rows


class ClayTrainingStudio:
    """Persistent conversational manager backed by GeoSeal integrity envelopes."""

    def __init__(
        self,
        home: Path | str = DEFAULT_HOME,
        session: str = "default",
        model: Optional[str] = None,
    ) -> None:
        self.home = Path(home).expanduser().resolve()
        self.session_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", session).strip("-") or "default"
        self.root = self.home / "sessions" / self.session_name
        self.artifacts = self.root / "artifacts"
        self.candidates_dir = self.root / "candidates"
        self.exports = self.root / "exports"
        self.state_path = self.root / "state.geoseal.json"
        self.events_path = self.root / "events.jsonl"
        self.examples_path = self.root / "approved_examples.geoseal.jsonl"
        for directory in (self.root, self.artifacts, self.candidates_dir, self.exports):
            directory.mkdir(parents=True, exist_ok=True)

        if self.state_path.exists():
            self.state = self._load_state()
            self._ensure_defaults()
            self._save_state()
        else:
            self.state = self._new_state(model)
            self._save_state()
            self.record("session_created", {"session": self.session_name})
        if model:
            self.state["active_peer"] = model
            if model not in self.state["peers"]:
                self.state["peers"].append(model)
            self._save_state()

    def _ensure_defaults(self) -> None:
        """Additive state migration for sessions created by an older CLI build."""
        voice = self.state.setdefault("voice", {})
        voice.setdefault("latest_checkpoint", None)
        # Older sessions spoke from latest_checkpoint. Preserve that explicit
        # owner-visible behavior during migration, while new sessions start with
        # no active model until evidence qualifies it or the owner adopts it.
        voice.setdefault("active_checkpoint", voice.get("latest_checkpoint"))
        voice.setdefault("generation", None)
        voice.setdefault("active_generation", voice.get("generation"))
        voice.setdefault("runs", [])
        voice.setdefault("promotion_history", [])
        opcode_graph = self.state.setdefault("opcode_graph", {})
        opcode_graph.setdefault("runs", [])
        opcode_graph.setdefault("latest_summary", None)
        opcode_graph.setdefault("latest_table", None)
        opcode_graph.setdefault("hatchery_runs", [])
        opcode_graph.setdefault("latest_hatchery_summary", None)
        opcode_graph.setdefault("internalization_runs", [])
        opcode_graph.setdefault("latest_internalization_summary", None)
        english_blocks = self.state.setdefault("english_blocks", {})
        english_blocks.setdefault("runs", [])
        english_blocks.setdefault("latest_summary", None)
        word_meanings = self.state.setdefault("word_meanings", {})
        word_meanings.setdefault("runs", [])
        word_meanings.setdefault("latest_summary", None)
        fourier_cards = self.state.setdefault("fourier_cards", {})
        fourier_cards.setdefault("runs", [])
        fourier_cards.setdefault("latest_summary", None)
        benchmarks = self.state.setdefault("reasoning_benchmarks", {})
        kaggriculture = benchmarks.setdefault("kaggriculture", {})
        # The public score is a live rating and has moved repeatedly for the
        # same submission.  Never preserve an old hard-coded score as truth.
        kaggriculture["target"] = dict(KAGGRICULTURE_TARGET)
        kaggriculture.setdefault("local_runs", [])
        kaggriculture.setdefault("latest_local_receipt", None)
        kaggriculture.setdefault("external_candidate_submission", None)
        kaggriculture.setdefault("score_observations", [])
        kaggriculture.setdefault("latest_score_snapshot", None)
        kaggriculture.setdefault("gold_runs", [])
        kaggriculture.setdefault("latest_gold_receipt", None)

    # ------------------------------------------------------------------
    # Durable state, seals, and receipts
    # ------------------------------------------------------------------
    def _new_state(self, model: Optional[str]) -> dict[str, Any]:
        genome = self._genome_hash("geodesic_routing_phdm", [])
        return {
            "version": VERSION,
            "session": self.session_name,
            "created_utc": utc_now(),
            "updated_utc": utc_now(),
            "goal": "Grow Clay through measured, local, held-out training runs.",
            "lane": "geodesic_routing_phdm",
            "geometry": [],
            "datasets": [],
            "peers": [model] if model else [],
            "active_peer": model,
            "training": {
                "seeds": [20260805, 20260806, 20260807],
                "batch_size": 64,
                "per_task": 24,
                "axis": {"name": "seconds", "center": 30, "step": 10, "minimum": 1},
            },
            "resources": {
                "compute_seconds_remaining": 900,
                "storage_mb_remaining": 512,
                "nutrients": 0,
                "larva": 3,
            },
            "hatcheries": [
                {"id": "h0", "generation": 0, "genome_hash": genome, "status": "active"}
            ],
            "drones": [
                {"id": "d0", "generation": 0, "status": "reserve", "source": "seed"}
            ],
            "candidates": [],
            "voice": {
                "latest_checkpoint": None,
                "active_checkpoint": None,
                "generation": None,
                "active_generation": None,
                "runs": [],
                "promotion_history": [],
            },
            "opcode_graph": {
                "runs": [],
                "latest_summary": None,
                "latest_table": None,
                "hatchery_runs": [],
                "latest_hatchery_summary": None,
                "internalization_runs": [],
                "latest_internalization_summary": None,
            },
            "english_blocks": {
                "runs": [],
                "latest_summary": None,
            },
            "word_meanings": {
                "runs": [],
                "latest_summary": None,
            },
            "fourier_cards": {
                "runs": [],
                "latest_summary": None,
            },
            "reasoning_benchmarks": {
                "kaggriculture": {
                    "target": dict(KAGGRICULTURE_TARGET),
                    "local_runs": [],
                    "latest_local_receipt": None,
                    "external_candidate_submission": None,
                    "score_observations": [],
                    "latest_score_snapshot": None,
                    "gold_runs": [],
                    "latest_gold_receipt": None,
                }
            },
            "last_interaction": None,
            "event_head": "0" * 64,
        }

    def _state_tag(self) -> str:
        return f"clay.train/state/{self.session_name}"

    def _save_state(self) -> None:
        self.state["updated_utc"] = utc_now()
        sealed = geoseal_encrypt(canonical_json(self.state), 0.0, 0.0, tag=self._state_tag())
        atomic_json(self.state_path, sealed)

    def _load_state(self) -> dict[str, Any]:
        sealed = json.loads(self.state_path.read_text(encoding="utf-8"))
        payload = geoseal_decrypt(sealed, expect_tag=self._state_tag())
        state = json.loads(payload.decode("utf-8"))
        if state.get("version") != VERSION:
            raise ValueError(f"unsupported clay.train state version {state.get('version')!r}")
        return state

    def record(self, kind: str, detail: dict[str, Any]) -> dict[str, Any]:
        clean_detail = json.loads(json.dumps(detail, ensure_ascii=False))
        for key, value in list(clean_detail.items()):
            if isinstance(value, str):
                clean_detail[key] = redact(value)
        event = {
            "utc": utc_now(),
            "session": self.session_name,
            "kind": kind,
            "detail": clean_detail,
            "previous": self.state.get("event_head", "0" * 64),
        }
        event["hash"] = sha256_bytes(canonical_json(event))
        with self.events_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(event, sort_keys=True, ensure_ascii=False) + "\n")
        self.state["event_head"] = event["hash"]
        self._save_state()
        return event

    def verify_events(self) -> tuple[bool, int, str]:
        previous = "0" * 64
        count = 0
        if not self.events_path.exists():
            return previous == self.state.get("event_head"), 0, previous
        for line in self.events_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            stored = event.pop("hash")
            if event.get("previous") != previous or sha256_bytes(canonical_json(event)) != stored:
                return False, count, previous
            previous = stored
            count += 1
        return previous == self.state.get("event_head"), count, previous

    def _genome_hash(self, lane: str, geometry: Iterable[str]) -> str:
        sources = {
            "manager": file_sha256(Path(__file__).resolve()),
            "trainer": file_sha256(LOCAL_TRAIN) if LOCAL_TRAIN.exists() else "MISSING",
            "dialogue_quine": file_sha256(DIALOGUE_QUINE) if DIALOGUE_QUINE.exists() else "MISSING",
        }
        payload = {"lane": lane, "geometry": sorted(set(geometry)), "sources": sources}
        return sha256_bytes(canonical_json(payload))

    # ------------------------------------------------------------------
    # Reversible paired examples
    # ------------------------------------------------------------------
    @staticmethod
    def paired_record(prompt: str, response: str, source: str = "owner") -> dict[str, Any]:
        clean_prompt = redact(prompt)
        clean_response = redact(response)
        semantic = canonical_json({"prompt": clean_prompt, "response": clean_response})
        payload = canonical_json({
            "prompt": clean_prompt,
            "response": clean_response,
            "source": source,
        })
        forward = encode(payload, "KO")
        reverse = encode(payload[::-1], "DR")
        recovered_forward = decode(forward, "KO")
        recovered_reverse = decode(reverse, "DR")[::-1]
        if recovered_forward != payload or recovered_reverse != payload:
            raise AssertionError("paired tongue joint did not recover the canonical payload")
        routes = []
        for tongue in TONGUES:
            byte_order = "reversed" if tongue == "DR" else "forward"
            routed = payload[::-1] if byte_order == "reversed" else payload
            tokens = encode(routed, tongue)
            recovered = decode(tokens, tongue)
            if byte_order == "reversed":
                recovered = recovered[::-1]
            if recovered != payload:
                raise AssertionError(f"{tongue} conveyor did not recover the canonical payload")
            routes.append({"tongue": tongue, "byte_order": byte_order})
        return {
            "version": 1,
            "joint_sha256": sha256_bytes(payload),
            "semantic_sha256": sha256_bytes(semantic),
            "canonical": json.loads(payload.decode("utf-8")),
            "forward": {"tongue": "KO", "tokens": forward},
            "reverse": {"tongue": "DR", "tokens": reverse, "byte_order": "reversed"},
            "routes": routes,
            "verified": True,
        }

    def approve(self, prompt: str, response: str, source: str = "owner") -> dict[str, Any]:
        record = self.paired_record(prompt, response, source)
        tag = f"clay.train/example/{self.session_name}/{record['joint_sha256']}"
        # The tongue token surfaces are deterministic views of the same bytes;
        # storing both duplicates the payload many times.  Persist one compressed
        # canonical joint plus route descriptors and regenerate both views on read.
        compact = {
            "version": record["version"],
            "joint_sha256": record["joint_sha256"],
            "semantic_sha256": record["semantic_sha256"],
            "canonical": record["canonical"],
            "forward": {"tongue": "KO", "materialize": "encode(canonical)"},
            "reverse": {
                "tongue": "DR",
                "byte_order": "reversed",
                "materialize": "encode(reverse(canonical))",
            },
            "routes": record["routes"],
            "verified": True,
        }
        raw = canonical_json(compact)
        compressed = zlib.compress(raw, level=9)
        sealed = geoseal_encrypt(compressed, 0.0, 0.0, tag=tag)
        with self.examples_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps({
                "tag": tag,
                "codec": "zlib-json-v1",
                "raw_bytes": len(raw),
                "compressed_bytes": len(compressed),
                "sealed": sealed,
            }, sort_keys=True) + "\n")
        self.state["resources"]["nutrients"] += 1
        self.record("example_approved", {
            "joint_sha256": record["joint_sha256"],
            "source": source,
            "raw_bytes": len(raw),
            "compressed_bytes": len(compressed),
        })
        return record

    def iter_examples(self) -> Iterable[dict[str, Any]]:
        if not self.examples_path.exists():
            return
        for line in self.examples_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            wrapper = json.loads(line)
            payload = geoseal_decrypt(wrapper["sealed"], expect_tag=wrapper["tag"])
            if wrapper.get("codec") == "zlib-json-v1":
                payload = zlib.decompress(payload)
            record = json.loads(payload.decode("utf-8"))
            canonical = canonical_json(record["canonical"])
            if sha256_bytes(canonical) != record["joint_sha256"]:
                raise ValueError("compressed example joint hash mismatch")
            semantic = canonical_json({
                "prompt": record["canonical"]["prompt"],
                "response": record["canonical"]["response"],
            })
            if record.get("semantic_sha256") not in {None, sha256_bytes(semantic)}:
                raise ValueError("compressed example semantic hash mismatch")
            # Re-materialize both semantic routes and prove recovery.  The
            # compressed file therefore omits no behavior, only duplicate text.
            if decode(encode(canonical, "KO"), "KO") != canonical:
                raise ValueError("compressed example KO route failed")
            if decode(encode(canonical[::-1], "DR"), "DR")[::-1] != canonical:
                raise ValueError("compressed example DR reverse route failed")
            for route in record.get("routes", []):
                routed = canonical[::-1] if route.get("byte_order") == "reversed" else canonical
                recovered = decode(encode(routed, route["tongue"]), route["tongue"])
                if route.get("byte_order") == "reversed":
                    recovered = recovered[::-1]
                if recovered != canonical:
                    raise ValueError(f"compressed example {route['tongue']} conveyor failed")
            yield record

    def export_examples(self) -> Path:
        path = self.exports / f"clay_dialogue_{int(time.time())}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in self.iter_examples():
                stream.write(json.dumps(row["canonical"], ensure_ascii=False, sort_keys=True) + "\n")
        self.record("examples_exported", {"path": str(path), "sha256": file_sha256(path)})
        return path

    def learn_file(self, raw_path: str) -> dict[str, Any]:
        """Owner-approved bulk intake of local prompt/response pairs.

        The file may be JSONL or a JSON array.  Every accepted row goes through
        the same redaction, paired KO/DR recovery, compression, GeoSeal, and
        lineage receipt as an interactive /accept.  Re-importing the same
        semantic joint is idempotent.
        """
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        text = path.read_text(encoding="utf-8-sig")
        if not text.strip():
            raise ValueError("learning file is empty")
        if text.lstrip().startswith("["):
            rows = json.loads(text)
            if not isinstance(rows, list):
                raise ValueError("JSON learning file must contain an array")
        else:
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]

        existing = {
            row.get("semantic_sha256")
            or sha256_bytes(canonical_json({
                "prompt": row["canonical"]["prompt"],
                "response": row["canonical"]["response"],
            }))
            for row in self.iter_examples()
        }
        accepted: list[str] = []
        skipped = 0
        for index, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                raise ValueError(f"learning row {index} is not an object")
            prompt = row.get("prompt") or row.get("instruction")
            response = next(
                (row.get(key) for key in ("response", "completion", "target", "output", "answer")
                 if row.get(key) is not None),
                None,
            )
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"learning row {index} has no non-empty prompt")
            if not isinstance(response, str) or not response.strip():
                raise ValueError(f"learning row {index} has no non-empty response")
            candidate = self.paired_record(prompt, response, f"owner-file:{path.name}")
            if candidate["semantic_sha256"] in existing:
                skipped += 1
                continue
            stored = self.approve(prompt, response, f"owner-file:{path.name}")
            existing.add(stored["semantic_sha256"])
            accepted.append(stored["joint_sha256"])

        receipt = {
            "path": str(path),
            "source_sha256": file_sha256(path),
            "rows": len(rows),
            "accepted": len(accepted),
            "duplicates": skipped,
            "joint_manifest_sha256": sha256_bytes(canonical_json(accepted)),
            "first_joint_sha256": accepted[0] if accepted else None,
            "last_joint_sha256": accepted[-1] if accepted else None,
        }
        self.record("owner_learning_file_imported", receipt)
        return receipt

    # ------------------------------------------------------------------
    # Hatchery / larva / drone resource lifecycle
    # ------------------------------------------------------------------
    def larva_capacity(self) -> int:
        # Three natural slots plus three injected slots per active hatchery.
        active = sum(h.get("status") == "active" for h in self.state["hatcheries"])
        return max(3, active * 6)

    def inject(self, count: int = 1) -> int:
        resources = self.state["resources"]
        count = max(0, int(count))
        room = max(0, self.larva_capacity() - resources["larva"])
        made = min(count, room, resources["nutrients"])
        resources["larva"] += made
        resources["nutrients"] -= made
        self.record("larva_injected", {"requested": count, "created": made})
        return made

    def spawn_drones(self, count: int = 1) -> list[str]:
        resources = self.state["resources"]
        made: list[str] = []
        for _ in range(max(0, int(count))):
            if resources["larva"] < 1 or resources["compute_seconds_remaining"] < 1:
                break
            resources["larva"] -= 1
            resources["compute_seconds_remaining"] -= 1
            drone_id = f"d{uuid.uuid4().hex[:8]}"
            self.state["drones"].append({
                "id": drone_id,
                "generation": max(h["generation"] for h in self.state["hatcheries"]),
                "status": "reserve",
                "source": "larva",
            })
            made.append(drone_id)
        self.record("drones_spawned", {"requested": count, "created": made})
        return made

    def promote_drone(self, drone_id: str) -> str:
        active_drones = [d for d in self.state["drones"] if d.get("status") == "reserve"]
        selected = next((d for d in active_drones if d["id"] == drone_id), None)
        if selected is None:
            raise ValueError(f"no reserve drone named {drone_id}")
        if len(active_drones) < 2:
            raise ValueError("make one more drone first; promotion keeps one spare drone alive")
        resources = self.state["resources"]
        if resources["compute_seconds_remaining"] < 30 or resources["storage_mb_remaining"] < 10:
            raise ValueError("promotion needs 30 compute-seconds and 10 MB; adjust /budget if desired")
        resources["compute_seconds_remaining"] -= 30
        resources["storage_mb_remaining"] -= 10
        selected["status"] = "morphed"
        parent_generation = max(h["generation"] for h in self.state["hatcheries"])
        hatchery_id = f"h{uuid.uuid4().hex[:8]}"
        genome = self._genome_hash(self.state["lane"], self.state["geometry"])
        self.state["hatcheries"].append({
            "id": hatchery_id,
            "generation": parent_generation + 1,
            "genome_hash": genome,
            "status": "active",
            "from_drone": drone_id,
        })
        resources["larva"] = min(self.larva_capacity(), resources["larva"] + 3)
        self.record("drone_promoted", {"drone": drone_id, "hatchery": hatchery_id})
        return hatchery_id

    # ------------------------------------------------------------------
    # One-dimensional hill candidates and real execution
    # ------------------------------------------------------------------
    def hatch(self, count: int = 3) -> list[dict[str, Any]]:
        count = max(1, int(count))
        resources = self.state["resources"]
        possible = min(count, resources["larva"])
        if possible == 0:
            raise ValueError("no larva available; approve examples then /inject, or add a hatchery")
        axis = self.state["training"]["axis"]
        generation = 1 + max((c.get("generation", 0) for c in self.state["candidates"]), default=0)
        parent = self.state["hatcheries"][generation % len(self.state["hatcheries"])]
        made: list[dict[str, Any]] = []
        for offset in alternating_offsets(possible):
            value = max(int(axis.get("minimum", 1)), int(axis["center"]) + offset * int(axis["step"]))
            cid = f"c{generation}-{uuid.uuid4().hex[:7]}"
            candidate = {
                "id": cid,
                "generation": generation,
                "parent_hatchery": parent["id"],
                "genome_hash": self._genome_hash(self.state["lane"], self.state["geometry"]),
                "lane": self.state["lane"],
                "axis": axis["name"],
                "value": value,
                "status": "queued",
                "created_utc": utc_now(),
            }
            self.state["candidates"].append(candidate)
            atomic_json(self.candidates_dir / f"{cid}.json", candidate)
            resources["larva"] -= 1
            made.append(candidate)
        self.record("candidates_hatched", {
            "generation": generation,
            "ids": [c["id"] for c in made],
            "axis_values": [c["value"] for c in made],
        })
        return made

    def trainer_argv(self, candidate: dict[str, Any]) -> list[str]:
        if candidate["lane"] not in LANES:
            raise ValueError(f"unknown local training lane {candidate['lane']!r}")
        seconds = int(candidate["value"])
        seeds = [int(seed) for seed in self.state["training"]["seeds"]]
        runid = f"claytrain-{self.session_name}-{candidate['id']}"
        return [
            sys.executable,
            str(LOCAL_TRAIN),
            "--lane", candidate["lane"],
            "--seed-list", ",".join(str(seed) for seed in seeds),
            "--seconds", str(seconds),
            "--batch-size", str(int(self.state["training"]["batch_size"])),
            "--per-task", str(int(self.state["training"]["per_task"])),
            "--runid", runid,
            "--mirror-stdout",
        ]

    def write_plan(self, candidates: Optional[list[dict[str, Any]]] = None) -> Path:
        candidates = candidates if candidates is not None else [
            c for c in self.state["candidates"] if c.get("status") == "queued"
        ]
        plan_id = f"plan-{int(time.time())}-{uuid.uuid4().hex[:6]}"
        bindings = {name: GEOMETRY_BINDINGS[name] for name in self.state["geometry"]}
        plan = {
            "version": 1,
            "id": plan_id,
            "created_utc": utc_now(),
            "goal": self.state["goal"],
            "lane": self.state["lane"],
            "geometry_bindings": bindings,
            "representation_topology": representation_topology_status(),
            "dataset_receipts": self.state["datasets"],
            "paired_examples": sum(1 for _ in self.iter_examples()),
            "method": {
                "heldout": True,
                "seeds": self.state["training"]["seeds"],
                "noise_rule": "candidate must beat centre by more than 2x pooled seed SD",
                "claims": "hill movement only; architecture claims still require lane controls and baseline",
                "promotion": "never requested by clay.train",
            },
            "resource_snapshot": self.state["resources"],
            "commands": [self.trainer_argv(c) for c in candidates],
            "candidate_ids": [c["id"] for c in candidates],
            "voice": self.state["voice"],
        }
        plain = self.artifacts / f"{plan_id}.json"
        atomic_json(plain, plan)
        sealed = geoseal_encrypt(canonical_json(plan), 0.0, 0.0, tag=f"clay.train/{plan_id}")
        atomic_json(self.artifacts / f"{plan_id}.geoseal.json", sealed)
        self.record("plan_written", {"path": str(plain), "sha256": file_sha256(plain)})
        return plain

    @staticmethod
    def _summary_metrics(summary: dict[str, Any]) -> list[float]:
        values: list[float] = []
        for result in summary.get("seed_results", []):
            metrics = result.get("heldout_metrics", {})
            for key in ("exact", "primary", "min_op_acc"):
                if metrics.get(key) is not None:
                    values.append(float(metrics[key]))
                    break
        return values

    def run_candidate(self, candidate_id: str) -> dict[str, Any]:
        candidate = next((c for c in self.state["candidates"] if c["id"] == candidate_id), None)
        if candidate is None:
            raise ValueError(f"unknown candidate {candidate_id}")
        if candidate.get("status") == "done":
            return candidate
        if not LOCAL_TRAIN.exists():
            raise FileNotFoundError(f"real local trainer missing: {LOCAL_TRAIN}")
        seeds = self.state["training"]["seeds"]
        estimated = int(candidate["value"]) * len(seeds)
        remaining = self.state["resources"]["compute_seconds_remaining"]
        if remaining < estimated:
            raise ValueError(
                f"candidate requests {estimated} compute-seconds; {remaining} remain. "
                "The owner can change it with /budget compute N."
            )
        self.state["resources"]["compute_seconds_remaining"] -= estimated
        candidate["status"] = "running"
        candidate["started_utc"] = utc_now()
        self._save_state()
        argv = self.trainer_argv(candidate)
        self.record("candidate_started", {"id": candidate_id, "argv": argv, "reserved_seconds": estimated})
        completed = subprocess.run(argv, cwd=str(LOCAL_TRAIN_ROOT), check=False)
        runid = argv[argv.index("--runid") + 1]
        summary_path = LOCAL_RUNS / candidate["lane"] / f"summary_{runid}.json"
        candidate["returncode"] = completed.returncode
        candidate["summary_path"] = str(summary_path)
        candidate["finished_utc"] = utc_now()
        if completed.returncode == 0 and summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            metrics = self._summary_metrics(summary)
            candidate["seed_metrics"] = metrics
            candidate["mean_metric"] = statistics.fmean(metrics) if metrics else None
            candidate["status"] = "done" if metrics else "no-metric"
            if metrics:
                drone_id = f"d{uuid.uuid4().hex[:8]}"
                self.state["drones"].append({
                    "id": drone_id,
                    "generation": candidate["generation"],
                    "status": "reserve",
                    "source": candidate_id,
                    "artifact": str(summary_path),
                })
                candidate["matured_drone"] = drone_id
        else:
            candidate["status"] = "failed"
        atomic_json(self.candidates_dir / f"{candidate_id}.json", candidate)
        self.record("candidate_finished", {
            "id": candidate_id,
            "status": candidate["status"],
            "returncode": completed.returncode,
            "mean_metric": candidate.get("mean_metric"),
        })
        return candidate

    def run_probe(self) -> dict[str, Any]:
        candidates = self.hatch(1)
        self.write_plan(candidates)
        return self.run_candidate(candidates[0]["id"])

    def run_swarm(self) -> list[dict[str, Any]]:
        queued = [c for c in self.state["candidates"] if c.get("status") == "queued"]
        if not queued:
            raise ValueError("no queued candidates; use /hatch 3 first")
        self.write_plan(queued)
        results = [self.run_candidate(candidate["id"]) for candidate in queued]
        self._evaluate_generation(queued[0]["generation"])
        return results

    @staticmethod
    def _pooled_sd(a: list[float], b: list[float]) -> float:
        if len(a) < 2 or len(b) < 2:
            return math.inf
        numerator = sum((x - statistics.fmean(a)) ** 2 for x in a)
        numerator += sum((x - statistics.fmean(b)) ** 2 for x in b)
        return math.sqrt(numerator / (len(a) + len(b) - 2))

    def _evaluate_generation(self, generation: int) -> dict[str, Any]:
        group = [
            c for c in self.state["candidates"]
            if c.get("generation") == generation and c.get("seed_metrics")
        ]
        axis = self.state["training"]["axis"]
        centre = next((c for c in group if int(c["value"]) == int(axis["center"])), None)
        if not group or centre is None:
            verdict = {"status": "UNDERPOWERED", "reason": "missing measured centre candidate"}
        else:
            winner = max(group, key=lambda c: float(c["mean_metric"]))
            if winner["id"] == centre["id"]:
                old_step = int(axis["step"])
                axis["step"] = max(int(axis.get("minimum", 1)), old_step // 2)
                verdict = {
                    "status": "CENTRE_HELD",
                    "centre": centre["value"],
                    "step": axis["step"],
                    "metric": centre["mean_metric"],
                }
            else:
                pooled = self._pooled_sd(winner["seed_metrics"], centre["seed_metrics"])
                delta = float(winner["mean_metric"]) - float(centre["mean_metric"])
                threshold = 2.0 * pooled
                if len(winner["seed_metrics"]) < 3 or len(centre["seed_metrics"]) < 3 or delta <= threshold:
                    verdict = {
                        "status": "UNDERPOWERED",
                        "winner": winner["value"],
                        "centre": centre["value"],
                        "delta": delta,
                        "required_delta": threshold,
                    }
                else:
                    old = axis["center"]
                    axis["center"] = winner["value"]
                    verdict = {
                        "status": "HILL_STEP_ACCEPTED",
                        "from": old,
                        "to": winner["value"],
                        "direction": "up" if winner["value"] > old else "down",
                        "delta": delta,
                        "required_delta": threshold,
                    }
        self.record("generation_evaluated", {"generation": generation, **verdict})
        return verdict

    # ------------------------------------------------------------------
    # Owner-dialogue neural offspring
    # ------------------------------------------------------------------
    def voice_train(self, seconds: float = 5.0) -> dict[str, Any]:
        if not DIALOGUE_QUINE.exists():
            raise FileNotFoundError(f"dialogue offspring trainer missing: {DIALOGUE_QUINE}")
        examples = list(self.iter_examples())
        if not examples:
            raise ValueError("approve at least one exchange with /accept or /teach first")
        seeds = [int(seed) for seed in self.state["training"]["seeds"]]
        seconds = max(0.05, float(seconds))
        # Three trained arms receive the same budget per seed: routed real,
        # shuffled-response, and a parameter-matched always-on ring.  The
        # untrained arm consumes no optimizer time.
        estimated = max(1, int(math.ceil(seconds * 3 * len(seeds))))
        remaining = self.state["resources"]["compute_seconds_remaining"]
        if remaining < estimated:
            raise ValueError(
                f"voice experiment requests {estimated} compute-seconds; {remaining} remain. "
                "The owner can change it with /budget compute N."
            )
        dataset = self.export_examples()
        run_id = f"voice-{int(time.time())}-{uuid.uuid4().hex[:6]}"
        output = self.artifacts / "voice" / run_id
        argv = [
            sys.executable,
            str(DIALOGUE_QUINE),
            "train",
            "--dataset", str(dataset),
            "--output", str(output),
            "--seed-list", ",".join(str(seed) for seed in seeds),
            "--seconds", str(seconds),
        ]
        # A child inherits only from the active lineage. A merely recent
        # candidate cannot quietly become the next parent.
        parent = self.state["voice"].get("active_checkpoint")
        if parent and Path(parent).exists():
            argv.extend(("--parent", parent))
        self.state["resources"]["compute_seconds_remaining"] -= estimated
        self._save_state()
        self.record("voice_training_started", {
            "run_id": run_id,
            "argv": argv,
            "pairs": len(examples),
            "reserved_seconds": estimated,
            "parent": parent,
        })
        completed = subprocess.run(argv, cwd=str(DIALOGUE_QUINE.parent), check=False)
        summary_path = output / "summary.json"
        if completed.returncode != 0 or not summary_path.exists():
            self.record("voice_training_failed", {
                "run_id": run_id,
                "returncode": completed.returncode,
                "summary_exists": summary_path.exists(),
            })
            raise RuntimeError(f"dialogue offspring failed with exit code {completed.returncode}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        checkpoint = Path(summary["best_checkpoint"])
        if not checkpoint.exists():
            raise RuntimeError("dialogue offspring returned no checkpoint")
        generations = [int(row["generation"]) for row in summary.get("seed_results", [])]
        receipt = {
            "run_id": run_id,
            "summary": str(summary_path),
            "summary_sha256": file_sha256(summary_path),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint),
            "generation": max(generations) if generations else None,
            "verdict": summary.get("verdict"),
            "mean_token_accuracy": summary.get("mean_token_accuracy"),
        }
        verdict_status = (summary.get("verdict") or {}).get("status")
        qualified = verdict_status == "QUALIFIED"
        receipt["auto_promoted"] = qualified
        receipt["promotion_reason"] = (
            "cleared multi-seed untrained, shuffled-response, and parameter-matched controls"
            if qualified
            else f"candidate retained; evidence verdict is {verdict_status or 'MISSING'}"
        )
        sealed = geoseal_encrypt(canonical_json(receipt), 0.0, 0.0, tag=f"clay.train/{run_id}")
        atomic_json(output / "receipt.geoseal.json", sealed)
        self.state["voice"]["latest_checkpoint"] = str(checkpoint)
        self.state["voice"]["generation"] = receipt["generation"]
        self.state["voice"]["runs"].append(receipt)
        if qualified:
            self.state["voice"]["active_checkpoint"] = str(checkpoint)
            self.state["voice"]["active_generation"] = receipt["generation"]
            self.state["voice"]["promotion_history"].append({
                "utc": utc_now(),
                "run_id": run_id,
                "checkpoint": str(checkpoint),
                "basis": "evidence-qualified",
            })
        drone_id = f"d{uuid.uuid4().hex[:8]}"
        self.state["drones"].append({
            "id": drone_id,
            "generation": receipt["generation"],
            "status": "reserve" if qualified else "candidate",
            "source": run_id,
            "artifact": str(checkpoint),
            "role": "dialogue",
        })
        receipt["matured_drone"] = drone_id
        self.record("voice_training_finished", receipt)
        return receipt

    def voice_adopt(self, selector: str = "latest") -> dict[str, Any]:
        """Explicit owner promotion, including candidates that did not qualify."""
        runs = self.state["voice"].get("runs", [])
        if not runs:
            raise ValueError("there is no dialogue candidate to adopt")
        if selector == "latest":
            receipt = runs[-1]
        else:
            receipt = next((row for row in runs if row.get("run_id") == selector), None)
            if receipt is None:
                raise ValueError(f"no dialogue run named {selector!r}")
        checkpoint = Path(receipt["checkpoint"])
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        if file_sha256(checkpoint) != receipt.get("checkpoint_sha256"):
            raise ValueError("dialogue candidate checkpoint hash mismatch")
        previous = self.state["voice"].get("active_checkpoint")
        self.state["voice"]["active_checkpoint"] = str(checkpoint)
        self.state["voice"]["active_generation"] = receipt.get("generation")
        promotion = {
            "utc": utc_now(),
            "run_id": receipt["run_id"],
            "checkpoint": str(checkpoint),
            "basis": "owner-override",
            "previous": previous,
            "evidence_verdict": receipt.get("verdict"),
        }
        self.state["voice"]["promotion_history"].append(promotion)
        for drone in self.state["drones"]:
            if drone.get("source") == receipt["run_id"] and drone.get("status") == "candidate":
                drone["status"] = "reserve"
        self.record("voice_owner_adopted", promotion)
        return promotion

    def voice_say(
        self,
        prompt: str,
        *,
        candidate: bool = False,
        record_event: bool = True,
    ) -> str:
        field = "latest_checkpoint" if candidate else "active_checkpoint"
        checkpoint = self.state["voice"].get(field)
        if not checkpoint or not Path(checkpoint).exists():
            if candidate:
                raise ValueError("Clay has no dialogue candidate yet; approve examples then /voice train")
            raise ValueError(
                "Clay has no active dialogue checkpoint; run /voice train, then either clear the "
                "evidence gate or explicitly /voice adopt latest"
            )
        argv = [
            sys.executable,
            str(DIALOGUE_QUINE),
            "say",
            "--checkpoint", checkpoint,
            "--prompt", redact(prompt),
        ]
        completed = subprocess.run(
            argv,
            cwd=str(DIALOGUE_QUINE.parent),
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or "dialogue inference failed")
        response = completed.stdout.rstrip("\r\n")
        if record_event:
            self.record("voice_spoke", {
                "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
                "response_sha256": sha256_bytes(response.encode("utf-8")),
                "checkpoint": checkpoint,
                "candidate": candidate,
            })
        return response

    # ------------------------------------------------------------------
    # Six-grid opcode quine + charged PyTorch graph
    # ------------------------------------------------------------------
    @staticmethod
    def _run_opcode_module(module: str, *arguments: str) -> subprocess.CompletedProcess[str]:
        if not OPCODE_GRAPH.exists() or not OPCODE_QUINE.exists():
            raise FileNotFoundError(
                f"opcode quine/trainer missing below {OPCODE_ROOT}"
            )
        return subprocess.run(
            [sys.executable, "-m", module, *arguments],
            cwd=str(OPCODE_ROOT),
            check=False,
            capture_output=True,
            text=True,
        )

    def opcode_diagnostic(self, action: str, generation: int = 0) -> str:
        """Run an exact non-training opcode inspection and return its receipt."""

        if action == "cycle":
            completed = self._run_opcode_module("clay_math.opcode_quine_cube", "cycle")
        elif action == "table":
            completed = self._run_opcode_module(
                "clay_math.opcode_quine_cube",
                "table",
                "--generation",
                str(int(generation)),
            )
        elif action == "inspect":
            completed = self._run_opcode_module("clay_math.charged_opcode_graph", "inspect")
        elif action == "selftest":
            completed = self._run_opcode_module("clay_math.charged_opcode_graph", "selftest")
        else:
            raise ValueError("opcode diagnostic must be inspect, selftest, cycle, or table")
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or f"opcode {action} failed")
        payload = completed.stdout.strip()
        if action == "table":
            target = self.artifacts / "opcode-graph" / f"generation-{int(generation)}-table.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            # Parse and re-serialize so a partial/non-JSON subprocess result can
            # never be mistaken for a completed table receipt.
            atomic_json(target, json.loads(payload))
            self.state["opcode_graph"]["latest_table"] = str(target)
            self.record("opcode_table_written", {
                "generation": int(generation),
                "path": str(target),
                "sha256": file_sha256(target),
            })
            return str(target)
        self.record("opcode_diagnostic", {
            "action": action,
            "stdout_sha256": sha256_bytes(payload.encode("utf-8")),
        })
        return payload

    def opcode_train(self, seconds: float = 5.0, generations: int = 192) -> dict[str, Any]:
        """Train charged, flat, and shuffled arms over reversible generations."""

        if not OPCODE_GRAPH.exists():
            raise FileNotFoundError(OPCODE_GRAPH)
        seconds = max(0.05, float(seconds))
        generations = int(generations)
        if not 1 <= generations <= 192:
            raise ValueError("opcode generations must be in 1..192")
        seeds = [int(seed) for seed in self.state["training"]["seeds"]]
        variants = ("charged", "flat", "shuffled")
        estimated = max(1, int(math.ceil(seconds * len(seeds) * len(variants))))
        remaining = self.state["resources"]["compute_seconds_remaining"]
        if remaining < estimated:
            raise ValueError(
                f"opcode experiment requests {estimated} compute-seconds; {remaining} remain. "
                "The owner can change it with /budget compute N."
            )
        run_id = f"opcode-{int(time.time())}-{uuid.uuid4().hex[:6]}"
        output = self.artifacts / "opcode-graph" / run_id
        argv = [
            sys.executable,
            "-m",
            "clay_math.charged_opcode_graph",
            "train",
            "--output",
            str(output),
            "--seeds",
            ",".join(str(seed) for seed in seeds),
            "--variants",
            ",".join(variants),
            "--seconds-per-arm",
            str(seconds),
            "--generations",
            str(generations),
        ]
        self.state["resources"]["compute_seconds_remaining"] -= estimated
        self._save_state()
        self.record("opcode_training_started", {
            "run_id": run_id,
            "argv": argv,
            "reserved_seconds": estimated,
            "generations": generations,
            "seeds": seeds,
        })
        completed = subprocess.run(argv, cwd=str(OPCODE_ROOT), check=False)
        summary_path = output / "summary.json"
        if completed.returncode != 0 or not summary_path.exists():
            self.record("opcode_training_failed", {
                "run_id": run_id,
                "returncode": completed.returncode,
                "summary_exists": summary_path.exists(),
            })
            raise RuntimeError(f"opcode graph training failed with exit code {completed.returncode}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        receipt = {
            "run_id": run_id,
            "summary": str(summary_path),
            "summary_sha256": file_sha256(summary_path),
            "generations": generations,
            "seeds": seeds,
            "variants": list(variants),
            "verdict": summary.get("verdict"),
            "charge_partition": summary.get("charge_partition"),
            "quine_cycle": summary.get("quine_cycle"),
        }
        sealed = geoseal_encrypt(canonical_json(receipt), 0.0, 0.0, tag=f"clay.train/{run_id}")
        atomic_json(output / "receipt.geoseal.json", sealed)
        self.state["opcode_graph"]["runs"].append(receipt)
        self.state["opcode_graph"]["latest_summary"] = str(summary_path)
        self.record("opcode_training_finished", receipt)
        return receipt

    def opcode_hatch(self, profile: str = "smoke") -> dict[str, Any]:
        """Evolve a compact six-byte layer map and test it on untouched opcodes."""

        if not OPCODE_HATCHERY.exists():
            raise FileNotFoundError(OPCODE_HATCHERY)
        profiles = {
            "smoke": {
                "search_generations": 2,
                "candidate_epochs": 1,
                "promotion_epochs": 1,
                "quine_generations": 4,
                "seconds_per_candidate": 2.0,
                "seconds_per_promotion_arm": 2.0,
                "smoke": True,
            },
            "local": {
                "search_generations": 4,
                "candidate_epochs": 3,
                "promotion_epochs": 4,
                "quine_generations": 12,
                "seconds_per_candidate": 12.0,
                "seconds_per_promotion_arm": 15.0,
                "smoke": False,
            },
        }
        if profile not in profiles:
            raise ValueError("opcode hatch profile must be smoke or local")
        config = profiles[profile]
        seeds = [int(seed) for seed in self.state["training"]["seeds"]]
        if profile == "local" and len(seeds) < 3:
            raise ValueError("opcode hatch local requires at least three promotion seeds")

        search_budget = (
            int(config["search_generations"])
            * 4
            * float(config["seconds_per_candidate"])
        )
        promotion_budget = (
            4
            * len(seeds)
            * float(config["seconds_per_promotion_arm"])
        )
        estimated = max(1, int(math.ceil(search_budget + promotion_budget)))
        remaining = int(self.state["resources"]["compute_seconds_remaining"])
        if remaining < estimated:
            raise ValueError(
                f"opcode hatch {profile} reserves {estimated} compute-seconds; {remaining} remain. "
                "The owner can change it with /budget compute N."
            )

        run_id = f"opcode-hatch-{profile}-{int(time.time())}-{uuid.uuid4().hex[:6]}"
        output = self.artifacts / "opcode-hatchery" / run_id
        argv = [
            sys.executable,
            "-m",
            "clay_math.charged_opcode_graph",
            "hatchery",
            "--output",
            str(output),
            "--search-generations",
            str(config["search_generations"]),
            "--candidate-epochs",
            str(config["candidate_epochs"]),
            "--promotion-epochs",
            str(config["promotion_epochs"]),
            "--quine-generations",
            str(config["quine_generations"]),
            "--population-size",
            "4",
            "--seeds",
            ",".join(str(seed) for seed in seeds),
            "--seconds-per-candidate",
            str(config["seconds_per_candidate"]),
            "--seconds-per-promotion-arm",
            str(config["seconds_per_promotion_arm"]),
        ]
        if config["smoke"]:
            argv.append("--smoke")

        self.state["resources"]["compute_seconds_remaining"] -= estimated
        self._save_state()
        self.record("opcode_hatchery_started", {
            "run_id": run_id,
            "profile": profile,
            "argv": argv,
            "reserved_seconds": estimated,
            "seeds": seeds,
        })
        completed = subprocess.run(argv, cwd=str(OPCODE_ROOT), check=False)
        summary_path = output / "summary.json"
        if completed.returncode != 0 or not summary_path.exists():
            self.record("opcode_hatchery_failed", {
                "run_id": run_id,
                "returncode": completed.returncode,
                "summary_exists": summary_path.exists(),
            })
            raise RuntimeError(f"opcode hatchery failed with exit code {completed.returncode}")

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        receipt = {
            "run_id": run_id,
            "profile": profile,
            "summary": str(summary_path),
            "summary_sha256": file_sha256(summary_path),
            "champion": summary.get("champion", {}).get("genome"),
            "search": summary.get("search"),
            "verdict": summary.get("verdict"),
            "claim_boundary": summary.get("claim_boundary"),
            "external_execution_started": bool(
                summary.get("offload", {}).get("external_execution_started", False)
            ),
        }
        sealed = geoseal_encrypt(canonical_json(receipt), 0.0, 0.0, tag=f"clay.train/{run_id}")
        atomic_json(output / "receipt.geoseal.json", sealed)
        self.state["opcode_graph"]["hatchery_runs"].append(receipt)
        self.state["opcode_graph"]["latest_hatchery_summary"] = str(summary_path)
        self.record("opcode_hatchery_finished", receipt)
        return receipt

    def opcode_internalize(self, profile: str = "full") -> dict[str, Any]:
        """Train a grounded parent and three controlled cross-view offspring."""

        if not OPCODE_INTERNALIZATION.exists():
            raise FileNotFoundError(OPCODE_INTERNALIZATION)
        profiles = {
            "smoke": {"parent_epochs": 2, "child_epochs": 3, "seconds_per_seed": 12},
            "full": {"parent_epochs": 32, "child_epochs": 192, "seconds_per_seed": 270},
        }
        if profile not in profiles:
            raise ValueError("opcode internalize profile must be smoke or full")
        config = profiles[profile]
        seeds = [int(seed) for seed in self.state["training"]["seeds"]]
        if profile == "full" and len(seeds) < 3:
            raise ValueError("opcode internalize full requires at least three seeds")
        estimated = max(1, int(config["seconds_per_seed"]) * len(seeds))
        remaining = int(self.state["resources"]["compute_seconds_remaining"])
        if remaining < estimated:
            raise ValueError(
                f"opcode internalize {profile} reserves {estimated} compute-seconds; "
                f"{remaining} remain. The owner can change it with /budget compute N."
            )

        run_id = f"opcode-internalize-{profile}-{int(time.time())}-{uuid.uuid4().hex[:6]}"
        output = self.artifacts / "opcode-internalization" / run_id
        argv = [
            sys.executable,
            "-m",
            "clay_math.representation_internalization",
            "train",
            "--output",
            str(output),
            "--seeds",
            ",".join(str(seed) for seed in seeds),
            "--parent-epochs",
            str(config["parent_epochs"]),
            "--child-epochs",
            str(config["child_epochs"]),
        ]
        self.state["resources"]["compute_seconds_remaining"] -= estimated
        self._save_state()
        self.record(
            "opcode_internalization_started",
            {
                "run_id": run_id,
                "profile": profile,
                "argv": argv,
                "reserved_seconds": estimated,
                "seeds": seeds,
            },
        )
        completed = subprocess.run(argv, cwd=str(OPCODE_ROOT), check=False)
        summary_path = output / "summary.json"
        if completed.returncode != 0 or not summary_path.exists():
            self.record(
                "opcode_internalization_failed",
                {
                    "run_id": run_id,
                    "returncode": completed.returncode,
                    "summary_exists": summary_path.exists(),
                },
            )
            raise RuntimeError(
                f"opcode representation internalization failed with exit code {completed.returncode}"
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        receipt = {
            "run_id": run_id,
            "profile": profile,
            "summary": str(summary_path),
            "summary_sha256": file_sha256(summary_path),
            "seeds": seeds,
            "parent_epochs": int(config["parent_epochs"]),
            "child_epochs": int(config["child_epochs"]),
            "protocol": summary.get("protocol"),
            "aggregates": summary.get("aggregates"),
            "verdict": summary.get("verdict"),
            "external_execution_started": False,
        }
        sealed = geoseal_encrypt(
            canonical_json(receipt), 0.0, 0.0, tag=f"clay.train/{run_id}"
        )
        atomic_json(output / "receipt.geoseal.json", sealed)
        self.state["opcode_graph"]["internalization_runs"].append(receipt)
        self.state["opcode_graph"]["latest_internalization_summary"] = str(summary_path)
        self.record("opcode_internalization_finished", receipt)
        return receipt

    def opcode_internalize_import(self, raw_summary: str) -> dict[str, Any]:
        """Attach an already completed local run without rerunning or promoting it."""

        summary_path = Path(raw_summary).expanduser().resolve()
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("schema") != "clay.representation_internalization.experiment.v1":
            raise ValueError("not a Clay representation-internalization summary")
        if summary.get("protocol", {}).get("status") != "PASS":
            raise ValueError("cannot import an internalization run whose leakage protocol did not pass")
        summary_hash = file_sha256(summary_path)
        for existing in self.state["opcode_graph"]["internalization_runs"]:
            if existing.get("summary_sha256") == summary_hash:
                return existing

        run_id = f"opcode-internalize-import-{int(time.time())}-{uuid.uuid4().hex[:6]}"
        receipt = {
            "run_id": run_id,
            "profile": "imported-local-full-cycle",
            "summary": str(summary_path),
            "summary_sha256": summary_hash,
            "seeds": summary.get("config", {}).get("seeds", []),
            "parent_epochs": summary.get("config", {}).get("parent_epochs"),
            "child_epochs": summary.get("config", {}).get("child_epochs"),
            "protocol": summary.get("protocol"),
            "aggregates": summary.get("aggregates"),
            "verdict": summary.get("verdict"),
            "external_execution_started": False,
            "promotion": "none; evidence receipt only",
        }
        target = self.artifacts / "opcode-internalization" / run_id
        sealed = geoseal_encrypt(
            canonical_json(receipt), 0.0, 0.0, tag=f"clay.train/{run_id}"
        )
        atomic_json(target / "receipt.geoseal.json", sealed)
        self.state["opcode_graph"]["internalization_runs"].append(receipt)
        self.state["opcode_graph"]["latest_internalization_summary"] = str(summary_path)
        self.record("opcode_internalization_imported", receipt)
        return receipt

    # ------------------------------------------------------------------
    # English step cells -> word blocks -> executable Python -> harmonics
    # ------------------------------------------------------------------
    def english_blocks_train(self, profile: str = "full") -> dict[str, Any]:
        """Run the local held-out English block-composition experiment."""

        if not ENGLISH_BLOCK_TRAINING.is_file():
            raise FileNotFoundError(ENGLISH_BLOCK_TRAINING)
        profiles = {
            "smoke": {"epochs": 3, "seconds": 45},
            "full": {"epochs": 72, "seconds": 300},
        }
        profile = profile.lower()
        if profile not in profiles:
            raise ValueError("English block training profile must be smoke or full")
        seeds = [int(seed) for seed in self.state["training"]["seeds"]]
        if profile == "smoke":
            seeds = seeds[:1]
        elif len(seeds) < 3:
            raise ValueError("English block full training requires at least three seeds")
        config = profiles[profile]
        remaining = int(self.state["resources"]["compute_seconds_remaining"])
        if remaining < int(config["seconds"]):
            raise ValueError(
                f"English block {profile} reserves {config['seconds']} compute-seconds; "
                f"{remaining} remain. The owner can change it with /budget compute N."
            )

        run_id = f"english-blocks-{profile}-{int(time.time())}-{uuid.uuid4().hex[:6]}"
        output = self.artifacts / "english-blocks" / run_id
        argv = [
            sys.executable,
            "-m",
            "clay_math.english_block_training",
            "train",
            "--output",
            str(output),
            "--epochs",
            str(config["epochs"]),
            "--seeds",
            *[str(seed) for seed in seeds],
        ]
        if profile == "smoke":
            argv.append("--smoke")
        self.state["resources"]["compute_seconds_remaining"] -= int(config["seconds"])
        self._save_state()
        self.record(
            "english_block_training_started",
            {
                "run_id": run_id,
                "profile": profile,
                "argv": argv,
                "reserved_seconds": config["seconds"],
                "seeds": seeds,
            },
        )
        completed = subprocess.run(argv, cwd=str(OPCODE_ROOT), check=False)
        summary_path = output / "summary.json"
        if completed.returncode != 0 or not summary_path.is_file():
            self.record(
                "english_block_training_failed",
                {
                    "run_id": run_id,
                    "returncode": completed.returncode,
                    "summary_exists": summary_path.is_file(),
                },
            )
            raise RuntimeError(
                f"English block training failed with exit code {completed.returncode}"
            )
        return self.english_blocks_import(str(summary_path), run_id=run_id, profile=profile)

    def english_blocks_import(
        self,
        raw_summary: str,
        *,
        run_id: str | None = None,
        profile: str = "imported-local",
    ) -> dict[str, Any]:
        """Attach a completed evidence run without promoting its checkpoint."""

        summary_path = Path(raw_summary).expanduser().resolve()
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("schema") != "clay_english_block_training_v1":
            raise ValueError("not a Clay English-block training summary")
        if not summary.get("protocol", {}).get("ok"):
            raise ValueError("English-block summary failed its held-out protocol")
        if not summary.get("evaluator", {}).get("ok"):
            raise ValueError("English-block summary failed its evaluator self-test")
        summary_hash = file_sha256(summary_path)
        for existing in self.state["english_blocks"]["runs"]:
            if existing.get("summary_sha256") == summary_hash:
                return existing

        run_id = run_id or f"english-blocks-import-{int(time.time())}-{uuid.uuid4().hex[:6]}"
        receipt = {
            "run_id": run_id,
            "profile": profile,
            "summary": str(summary_path),
            "summary_sha256": summary_hash,
            "seeds": summary.get("seeds", []),
            "epochs": summary.get("epochs"),
            "parameters_per_model": summary.get("parameters_per_model"),
            "protocol": summary.get("protocol"),
            "evaluator": summary.get("evaluator"),
            "aggregates": summary.get("aggregates"),
            "capability_claim_pass": summary.get("capability_claim_pass"),
            "aligned_architecture_claim_pass": summary.get(
                "aligned_architecture_claim_pass"
            ),
            "promotion": "none; evidence receipt only",
            "external_execution_started": False,
        }
        target = self.artifacts / "english-blocks" / run_id
        sealed = geoseal_encrypt(
            canonical_json(receipt), 0.0, 0.0, tag=f"clay.train/{run_id}"
        )
        atomic_json(target / "receipt.geoseal.json", sealed)
        self.state["english_blocks"]["runs"].append(receipt)
        self.state["english_blocks"]["latest_summary"] = str(summary_path)
        self.record("english_block_training_imported", receipt)
        return receipt

    # ------------------------------------------------------------------
    # Word + context -> convergent meaning vectors -> generated explanation
    # ------------------------------------------------------------------
    def word_meanings_train(self, profile: str = "full") -> dict[str, Any]:
        """Train the local generative word-explanation model with held-out lemmas."""

        if not WORD_EXPLANATION_TRAINING.is_file():
            raise FileNotFoundError(WORD_EXPLANATION_TRAINING)
        profiles = {
            # Measured 108.5 s on the local GTX 1660 Ti with all three arms;
            # reserve 130 s so the budget reflects the executable profile.
            "smoke": {"max_synsets": 512, "epochs": 12, "eval_limit": 24, "seconds": 130},
            "full": {"max_synsets": 1600, "epochs": 18, "eval_limit": 64, "seconds": 600},
        }
        profile = profile.lower()
        if profile not in profiles:
            raise ValueError("word-meaning training profile must be smoke or full")
        seeds = [int(seed) for seed in self.state["training"]["seeds"]]
        if profile == "smoke":
            seeds = seeds[:1]
        elif len(seeds) < 3:
            raise ValueError("word-meaning full training requires at least three seeds")
        config = profiles[profile]
        remaining = int(self.state["resources"]["compute_seconds_remaining"])
        if remaining < int(config["seconds"]):
            raise ValueError(
                f"word-meaning {profile} reserves {config['seconds']} compute-seconds; "
                f"{remaining} remain. The owner can change it with /budget compute N."
            )

        run_id = f"word-meanings-{profile}-{int(time.time())}-{uuid.uuid4().hex[:6]}"
        output = self.artifacts / "word-meanings" / run_id
        argv = [
            sys.executable,
            "-m",
            "clay_math.word_explanation_training",
            "train",
            "--output",
            str(output),
            "--max-synsets",
            str(config["max_synsets"]),
            "--epochs",
            str(config["epochs"]),
            "--eval-limit",
            str(config["eval_limit"]),
            "--seeds",
            *[str(seed) for seed in seeds],
        ]
        if profile == "smoke":
            argv.append("--smoke")
        self.state["resources"]["compute_seconds_remaining"] -= int(config["seconds"])
        self._save_state()
        self.record(
            "word_meaning_training_started",
            {
                "run_id": run_id,
                "profile": profile,
                "argv": argv,
                "reserved_seconds": config["seconds"],
                "seeds": seeds,
            },
        )
        completed = subprocess.run(argv, cwd=str(OPCODE_ROOT), check=False)
        summary_path = output / "summary.json"
        if completed.returncode != 0 or not summary_path.is_file():
            self.record(
                "word_meaning_training_failed",
                {
                    "run_id": run_id,
                    "returncode": completed.returncode,
                    "summary_exists": summary_path.is_file(),
                },
            )
            raise RuntimeError(
                f"word-meaning training failed with exit code {completed.returncode}"
            )
        return self.word_meanings_import(str(summary_path), run_id=run_id, profile=profile)

    def word_meanings_import(
        self,
        raw_summary: str,
        *,
        run_id: str | None = None,
        profile: str = "imported-local",
    ) -> dict[str, Any]:
        """Attach one definition-model run without silently promoting its weights."""

        summary_path = Path(raw_summary).expanduser().resolve()
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("schema") != "clay_word_explanation_training_v1":
            raise ValueError("not a Clay word-explanation training summary")
        if not summary.get("protocol", {}).get("ok"):
            raise ValueError("word-explanation summary failed its lemma-family holdout protocol")
        if not summary.get("evaluator", {}).get("ok"):
            raise ValueError("word-explanation summary failed its evaluator self-test")
        summary_hash = file_sha256(summary_path)
        for existing in self.state["word_meanings"]["runs"]:
            if existing.get("summary_sha256") == summary_hash:
                return existing

        run_id = run_id or f"word-meanings-import-{int(time.time())}-{uuid.uuid4().hex[:6]}"
        aligned = next(
            (row for row in summary.get("runs", []) if row.get("arm") == "cross_vector"),
            None,
        )
        receipt = {
            "run_id": run_id,
            "profile": profile,
            "summary": str(summary_path),
            "summary_sha256": summary_hash,
            "seeds": summary.get("seeds", []),
            "epochs": summary.get("epochs"),
            "parameters_per_model": summary.get("parameters_per_model"),
            "tokenizer": summary.get("tokenizer"),
            "curriculum": summary.get("curriculum"),
            "protocol": summary.get("protocol"),
            "evaluator": summary.get("evaluator"),
            "aggregates": summary.get("aggregates"),
            "comparisons": summary.get("comparisons"),
            "capability_claim_pass": summary.get("capability_claim_pass"),
            "cross_vector_architecture_claim_pass": summary.get(
                "cross_vector_architecture_claim_pass"
            ),
            "candidate_checkpoint": aligned.get("checkpoint") if aligned else None,
            "candidate_checkpoint_sha256": (
                aligned.get("checkpoint_sha256") if aligned else None
            ),
            "promotion": "none; candidate can be inspected but not silently adopted",
            "external_execution_started": False,
        }
        target = self.artifacts / "word-meanings" / run_id
        sealed = geoseal_encrypt(
            canonical_json(receipt), 0.0, 0.0, tag=f"clay.train/{run_id}"
        )
        atomic_json(target / "receipt.geoseal.json", sealed)
        self.state["word_meanings"]["runs"].append(receipt)
        self.state["word_meanings"]["latest_summary"] = str(summary_path)
        self.record("word_meaning_training_imported", receipt)
        return receipt

    def word_meanings_explain(self, word: str, context: str) -> dict[str, Any]:
        """Inspect the latest candidate in closed-book mode; this is not promotion."""

        runs = self.state["word_meanings"].get("runs", [])
        if not runs:
            raise ValueError("no word-meaning checkpoint exists; run /words train first")
        checkpoint = Path(runs[-1].get("candidate_checkpoint") or "").expanduser()
        expected = runs[-1].get("candidate_checkpoint_sha256")
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if expected and file_sha256(checkpoint) != expected:
            raise ValueError("word-meaning checkpoint hash does not match its receipt")
        argv = [
            sys.executable,
            "-m",
            "clay_math.word_explanation_training",
            "explain",
            "--checkpoint",
            str(checkpoint),
            "--word",
            word,
            "--context",
            context,
        ]
        completed = subprocess.run(
            argv,
            cwd=str(OPCODE_ROOT),
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or "word explanation failed")
        result = json.loads(completed.stdout)
        self.record(
            "word_meaning_candidate_inspected",
            {
                "word_sha256": sha256_bytes(word.encode("utf-8")),
                "context_sha256": sha256_bytes(context.encode("utf-8")),
                "checkpoint_sha256": expected,
            },
        )
        return result

    # ------------------------------------------------------------------
    # Fourier flash cards -> symbol cells -> executable arithmetic blocks
    # ------------------------------------------------------------------
    def fourier_cards_train(self, profile: str = "full") -> dict[str, Any]:
        """Run the local PyTorch flash-card curriculum with equal-size controls."""

        if not FOURIER_FLASHCARD_TRAINING.is_file():
            raise FileNotFoundError(FOURIER_FLASHCARD_TRAINING)
        profiles = {
            "smoke": {"stage1_epochs": 4, "stage2_epochs": 3, "seconds": 30},
            "full": {"stage1_epochs": 52, "stage2_epochs": 44, "seconds": 180},
        }
        profile = profile.lower()
        if profile not in profiles:
            raise ValueError("Fourier-card training profile must be smoke or full")
        seeds = [int(seed) for seed in self.state["training"]["seeds"]]
        if profile == "smoke":
            seeds = seeds[:1]
        elif len(seeds) < 3:
            raise ValueError("Fourier-card full training requires at least three seeds")
        config = profiles[profile]
        remaining = int(self.state["resources"]["compute_seconds_remaining"])
        if remaining < int(config["seconds"]):
            raise ValueError(
                f"Fourier-card {profile} reserves {config['seconds']} compute-seconds; "
                f"{remaining} remain. The owner can change it with /budget compute N."
            )

        run_id = f"fourier-cards-{profile}-{int(time.time())}-{uuid.uuid4().hex[:6]}"
        output = self.artifacts / "fourier-cards" / run_id
        argv = [
            sys.executable,
            "-m",
            "clay_math.fourier_flashcard_training",
            "--output",
            str(output),
            "--seeds",
            ",".join(str(seed) for seed in seeds),
            "--stage1-epochs",
            str(config["stage1_epochs"]),
            "--stage2-epochs",
            str(config["stage2_epochs"]),
        ]
        if profile == "smoke":
            argv.append("--smoke")
        self.state["resources"]["compute_seconds_remaining"] -= int(config["seconds"])
        self._save_state()
        self.record(
            "fourier_card_training_started",
            {
                "run_id": run_id,
                "profile": profile,
                "argv": argv,
                "reserved_seconds": config["seconds"],
                "seeds": seeds,
            },
        )
        completed = subprocess.run(argv, cwd=str(OPCODE_ROOT), check=False)
        summary_path = output / "summary.json"
        if completed.returncode != 0 or not summary_path.is_file():
            self.record(
                "fourier_card_training_failed",
                {
                    "run_id": run_id,
                    "returncode": completed.returncode,
                    "summary_exists": summary_path.is_file(),
                },
            )
            raise RuntimeError(
                f"Fourier-card training failed with exit code {completed.returncode}"
            )
        return self.fourier_cards_import(str(summary_path), run_id=run_id, profile=profile)

    def fourier_cards_import(
        self,
        raw_summary: str,
        *,
        run_id: str | None = None,
        profile: str = "imported-local",
    ) -> dict[str, Any]:
        """Attach sealed evidence; algorithms stay tools and no checkpoint auto-promotes."""

        summary_path = Path(raw_summary).expanduser().resolve()
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("schema") != "clay_fourier_flashcard_training_v1":
            raise ValueError("not a Clay Fourier-card training summary")
        if not summary.get("protocol", {}).get("ok"):
            raise ValueError("Fourier-card summary failed its held-out protocol")
        if not summary.get("evaluator", {}).get("ok"):
            raise ValueError("Fourier-card summary failed its evaluator self-test")
        summary_hash = file_sha256(summary_path)
        for existing in self.state["fourier_cards"]["runs"]:
            if existing.get("summary_sha256") == summary_hash:
                return existing

        run_id = run_id or f"fourier-cards-import-{int(time.time())}-{uuid.uuid4().hex[:6]}"
        receipt = {
            "run_id": run_id,
            "profile": profile,
            "summary": str(summary_path),
            "summary_sha256": summary_hash,
            "seeds": summary.get("seeds", []),
            "stage1_epochs": summary.get("stage1_epochs"),
            "stage2_epochs": summary.get("stage2_epochs"),
            "parameters_per_model": summary.get("parameters_per_model"),
            "protocol": summary.get("protocol"),
            "evaluator": summary.get("evaluator"),
            "aggregates": summary.get("aggregates"),
            "comparisons": summary.get("comparisons"),
            "capability_claim_pass": summary.get("capability_claim_pass"),
            "fourier_drawing_claim_pass": summary.get("fourier_drawing_claim_pass"),
            "adaptive_scheduler_claim_pass": summary.get(
                "adaptive_scheduler_claim_pass"
            ),
            "artifacts": summary.get("artifacts"),
            "tool_roles": {
                "learner": "PyTorch gradient-trained flash-card core",
                "hatchery": "candidate scheduling and equal-initialization controls",
                "quine_mccluskey": "exact Boolean compression and equivalence verifier",
                "fourier": "compressed symbol target and reconstruction evaluator",
            },
            "promotion": "none; evidence receipt only",
            "external_execution_started": False,
        }
        target = self.artifacts / "fourier-cards" / run_id
        sealed = geoseal_encrypt(
            canonical_json(receipt), 0.0, 0.0, tag=f"clay.train/{run_id}"
        )
        atomic_json(target / "receipt.geoseal.json", sealed)
        self.state["fourier_cards"]["runs"].append(receipt)
        self.state["fourier_cards"]["latest_summary"] = str(summary_path)
        self.record("fourier_card_training_imported", receipt)
        return receipt

    # ------------------------------------------------------------------
    # Downstream reasoning benchmarks
    # ------------------------------------------------------------------
    @staticmethod
    def _external_gate_from_snapshot(snapshot: dict[str, Any] | None) -> dict[str, Any]:
        rule = (
            "candidate and target must have complete numeric public scores in one "
            "authenticated Kaggle CLI snapshot; candidate must be strictly higher"
        )
        if snapshot is None:
            return {"status": "NOT_REFRESHED", "rule": rule}
        candidate = snapshot.get("candidate")
        if candidate is None:
            return {"status": "NOT_RUN", "rule": rule, "snapshot_sha256": snapshot["snapshot_sha256"]}
        target_score = snapshot.get("target", {}).get("public_score")
        candidate_score = candidate.get("public_score")
        if target_score is None or candidate_score is None:
            status = "PENDING"
        else:
            status = "PASS" if float(candidate_score) > float(target_score) else "FAIL"
        return {
            "status": status,
            "rule": rule,
            "same_snapshot": True,
            "snapshot_sha256": snapshot["snapshot_sha256"],
            "candidate": candidate,
        }

    def kaggriculture_refresh_scores(
        self,
        candidate_submission_id: int | None = None,
    ) -> dict[str, Any]:
        """Read target and optional candidate scores in one authenticated CLI call."""

        argv = ["kaggle", "competitions", "submissions", "kaggriculture", "-v"]
        self.record("kaggriculture_score_refresh_started", {
            "argv": argv,
            "candidate_submission_id": candidate_submission_id,
        })
        completed = subprocess.run(argv, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or "Kaggle score refresh failed")
        rows = parse_kaggle_submission_snapshot(completed.stdout)
        by_id = {row["submission_id"]: row for row in rows}
        target_id = int(KAGGRICULTURE_TARGET["submission_id"])
        if target_id not in by_id:
            raise RuntimeError(f"target submission {target_id} is absent from Kaggle CLI snapshot")
        candidate = None
        if candidate_submission_id is not None:
            candidate_id = int(candidate_submission_id)
            if candidate_id not in by_id:
                raise RuntimeError(
                    f"candidate submission {candidate_id} is absent from the same Kaggle CLI snapshot"
                )
            candidate = by_id[candidate_id]

        snapshot = {
            "schema": "clay.reasoning-benchmark.kaggriculture.score-snapshot.v1",
            "observed_utc": utc_now(),
            "source": "authenticated-kaggle-cli",
            "snapshot_sha256": sha256_bytes(completed.stdout.encode("utf-8")),
            "target": by_id[target_id],
            "candidate": candidate,
            "same_snapshot": True,
        }
        snapshot["external_gate"] = self._external_gate_from_snapshot(snapshot)
        benchmark_state = self.state["reasoning_benchmarks"]["kaggriculture"]
        benchmark_state["score_observations"].append(snapshot)
        benchmark_state["latest_score_snapshot"] = snapshot
        benchmark_state["external_candidate_submission"] = candidate

        run_id = f"kaggriculture-score-{int(time.time())}-{uuid.uuid4().hex[:6]}"
        run_dir = self.artifacts / "benchmarks" / "kaggriculture" / run_id
        receipt_path = run_dir / "score_snapshot.json"
        atomic_json(receipt_path, snapshot)
        sealed = geoseal_encrypt(canonical_json(snapshot), 0.0, 0.0, tag=f"clay.train/{run_id}")
        atomic_json(run_dir / "score_snapshot.geoseal.json", sealed)
        self.record("kaggriculture_score_refreshed", {
            "run_id": run_id,
            "receipt": str(receipt_path),
            "receipt_sha256": file_sha256(receipt_path),
            "external_gate": snapshot["external_gate"]["status"],
        })
        return snapshot

    def kaggriculture_benchmark_status(self) -> dict[str, Any]:
        """Describe the exact external target and the deliberately weaker local gate."""

        state = self.state["reasoning_benchmarks"]["kaggriculture"]
        target = dict(KAGGRICULTURE_TARGET)
        snapshot = state.get("latest_score_snapshot")
        if snapshot is not None:
            target.update(snapshot["target"])
            target["score_snapshot_sha256"] = snapshot["snapshot_sha256"]
            target["score_observed_utc"] = snapshot["observed_utc"]
        target["local_source"] = str(KAGGRICULTURE_TARGET_SOURCE)
        target["local_source_exists"] = KAGGRICULTURE_TARGET_SOURCE.is_file()
        target["local_source_sha256"] = (
            file_sha256(KAGGRICULTURE_TARGET_SOURCE)
            if KAGGRICULTURE_TARGET_SOURCE.is_file()
            else None
        )
        external_gate = self._external_gate_from_snapshot(snapshot)
        return {
            "schema": "clay.reasoning-benchmark.kaggriculture.v1",
            "purpose": "downstream reasoning exam; never training data or an opcode-fitness target",
            "target": target,
            "local_surrogate": {
                "runner": str(KAGGRICULTURE_COMPARE),
                "rule": "fresh seeds, both seats, direct win signal greater than twice its SE",
                "not_equivalent_to_public_score": True,
                "latest_receipt": state.get("latest_local_receipt"),
            },
            "gold_replay_profile": {
                "profiler": str(KAGGRICULTURE_PROFILE),
                "replay_dir": str(KAGGRICULTURE_GOLD_DIR),
                "latest_receipt": state.get("latest_gold_receipt"),
                "descriptive_not_training_data": True,
            },
            "external_gate": external_gate,
            "benchmark_data_used_for_training": False,
        }

    def kaggriculture_gold_profile(
        self,
        candidate_name: str = "main_goldroute_flat.py",
    ) -> dict[str, Any]:
        """Profile gold trajectories and attach compatible held-out receipts."""

        if not KAGGRICULTURE_PROFILE.is_file() or not KAGGRICULTURE_GOLD_DIR.is_dir():
            raise FileNotFoundError("Kaggriculture replay profiler or gold replay directory is missing")
        candidate = Path(candidate_name)
        if not candidate.is_absolute():
            candidate = KAGGRICULTURE_ROOT / candidate
        candidate = candidate.resolve()
        if not candidate.is_file():
            raise FileNotFoundError(candidate)

        run_id = f"kaggriculture-gold-{int(time.time())}-{uuid.uuid4().hex[:6]}"
        run_dir = self.artifacts / "benchmarks" / "kaggriculture" / run_id
        profile_json = run_dir / "gold_profile.json"
        profile_md = run_dir / "gold_profile.md"
        run_dir.mkdir(parents=True, exist_ok=True)
        argv = [
            sys.executable,
            str(KAGGRICULTURE_PROFILE),
            "--replay-dir",
            str(KAGGRICULTURE_GOLD_DIR),
            "--output-json",
            str(profile_json),
            "--output-md",
            str(profile_md),
        ]
        self.record("kaggriculture_gold_profile_started", {
            "run_id": run_id,
            "candidate": candidate.name,
            "argv": argv,
        })
        completed = subprocess.run(
            argv,
            cwd=str(KAGGRICULTURE_ROOT),
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0 or not profile_json.is_file() or not profile_md.is_file():
            raise RuntimeError(completed.stderr.strip() or "Kaggriculture gold profiling failed")

        candidate_hash = file_sha256(candidate)
        compatible_hashes = {candidate_hash}
        conformance_summary = None
        if KAGGRICULTURE_CONFORMANCE.is_file():
            conformance = json.loads(KAGGRICULTURE_CONFORMANCE.read_text(encoding="utf-8"))
            if (
                conformance.get("status") == "PASS"
                and conformance.get("flat", {}).get("sha256") == candidate_hash
                and int(conformance.get("mismatch_count", -1)) == 0
            ):
                compatible_hashes.add(str(conformance["wrapper"]["sha256"]))
                conformance_summary = {
                    "path": str(KAGGRICULTURE_CONFORMANCE),
                    "sha256": file_sha256(KAGGRICULTURE_CONFORMANCE),
                    "observations_checked": conformance.get("observations_checked"),
                    "status": "PASS",
                }

        heldout = []
        for path in sorted(KAGGRICULTURE_GOLD_DIR.glob("goldroute_vs_*_holdout.json")):
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("candidate_sha256") not in compatible_hashes:
                continue
            heldout.append({
                "opponent": row.get("opponent"),
                "games": row.get("games"),
                "wins": row.get("wins"),
                "losses": row.get("losses"),
                "ties": row.get("ties"),
                "mean_margin": row.get("mean_margin"),
                "margin_se": row.get("margin_se"),
                "passes_2se": bool(row.get("passes_2se")),
                "path": str(path),
                "sha256": file_sha256(path),
            })
        heldout_pass = len(heldout) >= 3 and all(row["passes_2se"] for row in heldout)
        receipt = {
            "schema": "clay.reasoning-benchmark.kaggriculture.gold-profile.v1",
            "run_id": run_id,
            "candidate": candidate.name,
            "candidate_sha256": candidate_hash,
            "gold_profile": {
                "json": str(profile_json),
                "json_sha256": file_sha256(profile_json),
                "markdown": str(profile_md),
                "markdown_sha256": file_sha256(profile_md),
            },
            "action_conformance": conformance_summary,
            "heldout_results": heldout,
            "heldout_gate": "PASS" if heldout_pass else "INSUFFICIENT_OR_FAIL",
            "external_gate": "NOT_RUN",
            "claim_boundary": (
                "Gold replays are descriptive and held-out games are a local surrogate. "
                "Only a candidate and target read from the same Kaggle score snapshot can pass the external gate."
            ),
            "benchmark_data_used_for_training": False,
        }
        receipt_path = run_dir / "gold_reasoning_gate.json"
        atomic_json(receipt_path, receipt)
        sealed = geoseal_encrypt(canonical_json(receipt), 0.0, 0.0, tag=f"clay.train/{run_id}")
        atomic_json(run_dir / "gold_reasoning_gate.geoseal.json", sealed)
        record = {
            "run_id": run_id,
            "candidate": candidate.name,
            "receipt": str(receipt_path),
            "receipt_sha256": file_sha256(receipt_path),
            "heldout_gate": receipt["heldout_gate"],
        }
        state = self.state["reasoning_benchmarks"]["kaggriculture"]
        state["gold_runs"].append(record)
        state["latest_gold_receipt"] = str(receipt_path)
        self.record("kaggriculture_gold_profile_finished", record)
        return receipt

    def kaggriculture_local_gate(
        self,
        candidate_name: str = "main_feedback_quine.py",
        seed_count: int = 12,
        seed_start: int | None = None,
    ) -> dict[str, Any]:
        """Run a fresh local proxy against the source behind the target submission."""

        if not KAGGRICULTURE_COMPARE.is_file() or not KAGGRICULTURE_TARGET_SOURCE.is_file():
            raise FileNotFoundError("Kaggriculture compare_agents.py or main_profit.py is missing")
        seed_count = int(seed_count)
        if seed_count < 3:
            raise ValueError("Kaggriculture local gate requires at least three seeds")
        candidate = Path(candidate_name)
        if not candidate.is_absolute():
            candidate = KAGGRICULTURE_ROOT / candidate
        candidate = candidate.resolve()
        if not candidate.is_file():
            raise FileNotFoundError(candidate)

        benchmark_state = self.state["reasoning_benchmarks"]["kaggriculture"]
        if seed_start is None:
            seed_start = 20000 + 100 * len(benchmark_state["local_runs"])
        seed_start = int(seed_start)
        run_id = f"kaggriculture-local-{int(time.time())}-{uuid.uuid4().hex[:6]}"
        run_dir = self.artifacts / "benchmarks" / "kaggriculture" / run_id
        raw_receipt = run_dir / "head_to_head.json"
        run_dir.mkdir(parents=True, exist_ok=True)
        argv = [
            sys.executable,
            str(KAGGRICULTURE_COMPARE),
            "--candidate",
            str(candidate),
            "--opponent",
            str(KAGGRICULTURE_TARGET_SOURCE),
            "--seed-start",
            str(seed_start),
            "--seed-count",
            str(seed_count),
            "--workers",
            str(max(1, min(8, os.cpu_count() or 1))),
            "--output",
            str(raw_receipt),
        ]
        self.record("reasoning_benchmark_started", {
            "benchmark": "kaggriculture",
            "run_id": run_id,
            "argv": argv,
            "target_submission_id": KAGGRICULTURE_TARGET["submission_id"],
        })
        completed = subprocess.run(
            argv,
            cwd=str(KAGGRICULTURE_ROOT),
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode not in {0, 1} or not raw_receipt.is_file():
            self.record("reasoning_benchmark_failed", {
                "benchmark": "kaggriculture",
                "run_id": run_id,
                "returncode": completed.returncode,
                "receipt_exists": raw_receipt.is_file(),
                "stderr_sha256": sha256_bytes(completed.stderr.encode("utf-8")),
            })
            raise RuntimeError(
                completed.stderr.strip()
                or f"Kaggriculture local gate failed with exit code {completed.returncode}"
            )

        local = json.loads(raw_receipt.read_text(encoding="utf-8"))
        local_pass = bool(local.get("passes_2se"))
        target = dict(KAGGRICULTURE_TARGET)
        latest_snapshot = benchmark_state.get("latest_score_snapshot")
        if latest_snapshot is not None:
            target.update(latest_snapshot["target"])
            target["score_snapshot_sha256"] = latest_snapshot["snapshot_sha256"]
        receipt = {
            "schema": "clay.reasoning-benchmark.kaggriculture.local.v1",
            "run_id": run_id,
            "candidate": candidate.name,
            "candidate_sha256": file_sha256(candidate),
            "target": target,
            "target_source_sha256": file_sha256(KAGGRICULTURE_TARGET_SOURCE),
            "seed_start": seed_start,
            "seed_count": seed_count,
            "seats_per_seed": 2,
            "local_result": local,
            "local_gate": "PASS" if local_pass else "FAIL",
            "external_gate": "NOT_RUN",
            "overall": "LOCAL_PASS_EXTERNAL_UNTESTED" if local_pass else "LOCAL_FAIL",
            "claim_boundary": (
                "This is a local two-seat source head-to-head, not a Kaggle leaderboard score. "
                "The external gate requires candidate and target scores from one authenticated "
                "Kaggle CLI snapshot, with the candidate strictly higher."
            ),
            "benchmark_data_used_for_training": False,
        }
        gate_path = run_dir / "reasoning_gate.json"
        atomic_json(gate_path, receipt)
        sealed = geoseal_encrypt(canonical_json(receipt), 0.0, 0.0, tag=f"clay.train/{run_id}")
        atomic_json(run_dir / "reasoning_gate.geoseal.json", sealed)
        run_record = {
            "run_id": run_id,
            "candidate": candidate.name,
            "receipt": str(gate_path),
            "receipt_sha256": file_sha256(gate_path),
            "local_gate": receipt["local_gate"],
            "external_gate": receipt["external_gate"],
        }
        benchmark_state["local_runs"].append(run_record)
        benchmark_state["latest_local_receipt"] = str(gate_path)
        self.record("reasoning_benchmark_finished", run_record)
        return receipt

    # ------------------------------------------------------------------
    # Configuration and local conversation
    # ------------------------------------------------------------------
    def add_dataset(self, raw_path: str, role: str = "train") -> dict[str, Any]:
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        if role not in {"train", "holdout", "control"}:
            raise ValueError("dataset role must be train, holdout, or control")
        receipt: dict[str, Any] = {"path": str(path), "role": role, "kind": "dir" if path.is_dir() else "file"}
        if path.is_file():
            receipt.update({"bytes": path.stat().st_size, "sha256": file_sha256(path)})
        else:
            files = [p for p in path.rglob("*") if p.is_file()]
            receipt.update({"files": len(files), "bytes": sum(p.stat().st_size for p in files)})
        self.state["datasets"].append(receipt)
        self.record("dataset_added", receipt)
        return receipt

    def set_lane(self, lane: str) -> None:
        if lane not in LANES:
            raise ValueError(f"unknown lane {lane}; choose from {', '.join(sorted(LANES))}")
        self.state["lane"] = lane
        self.record("lane_selected", {"lane": lane})

    def set_geometry(self, names: Iterable[str]) -> list[str]:
        unknown = sorted(set(names) - set(GEOMETRY_BINDINGS))
        if unknown:
            raise ValueError(f"unknown geometry words: {', '.join(unknown)}")
        self.state["geometry"] = list(dict.fromkeys(names))
        trainable = [
            GEOMETRY_BINDINGS[name].get("lane")
            for name in self.state["geometry"]
            if GEOMETRY_BINDINGS[name]["status"].startswith("trainable")
        ]
        trainable = [lane for lane in trainable if lane in LANES]
        if trainable:
            self.state["lane"] = trainable[-1]
        self.record("geometry_selected", {"geometry": self.state["geometry"], "lane": self.state["lane"]})
        return self.state["geometry"]

    def local_reply(self, prompt: str) -> tuple[str, str]:
        model = self.state.get("active_peer")
        if not model:
            if self.state["voice"].get("latest_checkpoint"):
                try:
                    return "clay-dialogue", self.voice_say(prompt, record_event=False)
                except (ValueError, RuntimeError, OSError):
                    pass
            return "manager", self._manager_reply(prompt)
        host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
        body = {
            "model": model,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a local peer inside Clay's training studio. Help turn the owner's idea into "
                        "a concrete training example or measurable run. Never claim a run occurred unless the "
                        "provided artifacts say it did. Be concise."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }
        request = urllib.request.Request(
            f"{host}/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = json.loads(response.read().decode("utf-8"))
            text = payload.get("message", {}).get("content", "").strip()
            return model, text or self._manager_reply(prompt)
        except (OSError, urllib.error.URLError, ValueError, json.JSONDecodeError) as exc:
            return "manager", f"Local peer {model!r} was unavailable ({type(exc).__name__}). " + self._manager_reply(prompt)

    def _manager_reply(self, prompt: str) -> str:
        lower = prompt.lower()
        selected = [name for name in GEOMETRY_BINDINGS if name in lower]
        parts = [f"Goal recorded locally. Active lane: {self.state['lane']}." ]
        if selected:
            parts.append("I mapped " + ", ".join(selected) + " to existing surfaces; /plan shows what is trainable versus diagnostic.")
        parts.append("Use /accept to keep this exchange as a reversible KO/DR training pair, then /hatch 3 and /run swarm for a measured step.")
        return " ".join(parts)

    def interpret(self, text: str) -> dict[str, Any]:
        clean = redact(text.strip())
        lower = clean.lower()
        changes: dict[str, Any] = {}

        geometry = [name for name in GEOMETRY_BINDINGS if re.search(rf"\b{re.escape(name)}\w*\b", lower)]
        if geometry:
            self.set_geometry(geometry)
            changes["geometry"] = geometry

        lane_aliases = {
            "inverse routing": "inverse_routing",
            "dynamic braid": "dynamic_braid",
            "cross plane": "cross_plane_relations",
            "geodesic": "geodesic_routing_phdm",
            "packet routing": "packet_routing",
        }
        for phrase, lane in lane_aliases.items():
            if phrase in lower:
                self.set_lane(lane)
                changes["lane"] = lane
                break

        seconds = re.search(r"\b(?:for|use|at)\s+(\d+)\s*(?:seconds?|secs?|s)\b", lower)
        if seconds:
            self.state["training"]["axis"]["center"] = max(1, int(seconds.group(1)))
            changes["seconds"] = int(seconds.group(1))

        seeds = re.search(r"\b(?:use|with|run)\s+(\d+)\s+seeds?\b", lower)
        if seeds:
            count = max(1, int(seeds.group(1)))
            base = self.state["training"]["seeds"][0]
            self.state["training"]["seeds"] = [base + i for i in range(count)]
            changes["seeds"] = count

        if "i want" in lower or "goal" in lower or "train clay" in lower:
            self.state["goal"] = clean
            changes["goal"] = clean

        self._save_state()
        source, response = self.local_reply(clean)
        self.state["last_interaction"] = {"prompt": clean, "response": redact(response), "source": source}
        self.record("conversation", {
            "prompt_sha256": sha256_bytes(clean.encode("utf-8")),
            "response_sha256": sha256_bytes(response.encode("utf-8")),
            "source": source,
            "changes": changes,
        })

        voice_context = "voice" in lower or "speak" in lower or "dialogue" in lower
        run_requested = bool(
            re.search(r"\b(?:run|start)\b.*\b(?:probe|training|train|swarm)\b", lower)
            and not voice_context
        )
        voice_run_requested = bool(
            voice_context and re.search(r"\b(?:run|start|train)\b", lower)
        )
        internalization_context = bool(
            re.search(r"\b(?:internaliz\w*|multi[- ]?view|cross[- ]?view)\b", lower)
            or (
                re.search(r"\b(?:opcode|tongue|representation)\w*\b", lower)
                and re.search(r"\b(?:learn|train)\w*\b", lower)
            )
        )
        internalization_run_requested = bool(
            internalization_context and re.search(r"\b(?:run|start|train)\w*\b", lower)
        )
        internalization_profile = (
            "full" if re.search(r"\b(?:full|real|complete|192)\b", lower) else "smoke"
        )
        word_meaning_context = bool(
            re.search(r"\b(?:word|meaning|definition|explain|vocabulary)\w*\b", lower)
            and not re.search(r"\b(?:block|python|cod\w*|function|fourier|glyph)\w*\b", lower)
        )
        word_meaning_run_requested = bool(
            word_meaning_context
            and re.search(r"\b(?:run|start|train|learn|teach)\w*\b", lower)
        )
        word_meaning_profile = (
            "full" if re.search(r"\b(?:full|real|complete)\b", lower) else "smoke"
        )
        english_block_context = bool(
            re.search(r"\b(?:english|dictionary|word)\w*\b", lower)
            and re.search(r"\b(?:block|python|cod\w*|function)\w*\b", lower)
        )
        english_block_run_requested = bool(
            english_block_context and re.search(r"\b(?:run|start|train|learn)\w*\b", lower)
        )
        english_block_profile = (
            "full" if re.search(r"\b(?:full|real|complete)\b", lower) else "smoke"
        )
        fourier_card_context = bool(
            re.search(r"\b(?:fourier|harmonic|flash[- ]?cards?)\b", lower)
            or re.search(r"\b(?:draw|render)\w*\b.*\b(?:letter|symbol|glyph)\w*\b", lower)
            or re.search(r"\b(?:letter|symbol|glyph)\w*\b.*\bpython\b", lower)
        )
        fourier_card_run_requested = bool(
            fourier_card_context
            and re.search(r"\b(?:run|start|train|learn)\w*\b", lower)
        )
        fourier_card_profile = (
            "full" if re.search(r"\b(?:full|real|complete)\b", lower) else "smoke"
        )
        if fourier_card_run_requested:
            # The more specific learned-symbol route owns mixed English/Python prompts.
            english_block_run_requested = False
        if word_meaning_run_requested:
            # Definition generation owns ordinary word-understanding prompts;
            # opcode internalization remains the representation-specific lane.
            internalization_run_requested = False
        if (
            internalization_run_requested
            or word_meaning_run_requested
            or english_block_run_requested
            or fourier_card_run_requested
        ):
            run_requested = False
        return {
            "source": source,
            "response": response,
            "changes": changes,
            "run_requested": run_requested,
            "voice_run_requested": voice_run_requested,
            "internalization_run_requested": internalization_run_requested,
            "internalization_profile": internalization_profile,
            "word_meaning_run_requested": word_meaning_run_requested,
            "word_meaning_profile": word_meaning_profile,
            "english_block_run_requested": english_block_run_requested,
            "english_block_profile": english_block_profile,
            "fourier_card_run_requested": fourier_card_run_requested,
            "fourier_card_profile": fourier_card_profile,
        }

    # ------------------------------------------------------------------
    # Human-readable command surface
    # ------------------------------------------------------------------
    def status_text(self) -> str:
        ok, events, _ = self.verify_events()
        resources = self.state["resources"]
        queued = sum(c.get("status") == "queued" for c in self.state["candidates"])
        examples = sum(1 for _ in self.iter_examples())
        key_mode = "owner key" if os.environ.get("GEOSEAL_KEY") else "reproducible demo key"
        return (
            f"Clay training studio [{self.session_name}]\n"
            f"  goal:       {self.state['goal']}\n"
            f"  lane:       {self.state['lane']}\n"
            f"  geometry:   {', '.join(self.state['geometry']) or '(none)'}\n"
            f"  axis:       {self.state['training']['axis']}\n"
            f"  seeds:      {self.state['training']['seeds']}\n"
            f"  lifecycle:  {len(self.state['hatcheries'])} hatchery / {resources['larva']} larva / "
            f"{sum(d.get('status') == 'reserve' for d in self.state['drones'])} reserve drones\n"
            f"  resources:  {resources['compute_seconds_remaining']} compute-s / "
            f"{resources['storage_mb_remaining']} MB / {resources['nutrients']} nutrients\n"
            f"  corpus:     {examples} approved reversible pairs / {len(self.state['datasets'])} datasets\n"
            f"  voice:      candidate generation {self.state['voice'].get('generation')} / "
            f"active generation {self.state['voice'].get('active_generation')} / "
            f"{len(self.state['voice'].get('runs', []))} measured runs\n"
            f"  opcodes:    6 x 64 nested grids / "
            f"{len(self.state['opcode_graph'].get('runs', []))} charged-graph runs / "
            f"{len(self.state['opcode_graph'].get('hatchery_runs', []))} hatchery runs / "
            f"{len(self.state['opcode_graph'].get('internalization_runs', []))} internalization runs / "
            f"latest {self.state['opcode_graph'].get('latest_internalization_summary') or self.state['opcode_graph'].get('latest_hatchery_summary') or self.state['opcode_graph'].get('latest_summary') or '(none)'}\n"
            f"  English:    {len(self.state['english_blocks'].get('runs', []))} block/function runs / "
            f"latest {self.state['english_blocks'].get('latest_summary') or '(none)'}\n"
            f"  meanings:   {len(self.state['word_meanings'].get('runs', []))} generative definition runs / "
            f"latest {self.state['word_meanings'].get('latest_summary') or '(none)'}\n"
            f"  cards:      {len(self.state['fourier_cards'].get('runs', []))} Fourier/code runs / "
            f"latest {self.state['fourier_cards'].get('latest_summary') or '(none)'}\n"
            f"  reasoning:  Kaggriculture local runs "
            f"{len(self.state['reasoning_benchmarks']['kaggriculture'].get('local_runs', []))} / "
            f"external {self.kaggriculture_benchmark_status()['external_gate']['status']}\n"
            f"  queue:      {queued} candidates\n"
            f"  integrity:  {'OK' if ok else 'BROKEN'} ({events} chained events; {key_mode})"
        )

    def help_text(self) -> str:
        return """Talk normally, or use exact commands:
  /status                         show the live hatchery and resource ledger
  /representations                show measured tokenizer/opcode translation seams
  /goal TEXT                      set the training objective
  /lane NAME                      choose an existing real trainer lane
  /geometry NAME...               curve/interweave/twist/invert/reverse/dual/etc.
  /axis seconds CENTER STEP       set the one-dimensional hill
  /seeds N                        use N distinct seeds (3+ required for a claim)
  /data PATH [train|holdout|control]
  /peer add MODEL | use MODEL | off | list   local Ollama peers only
  /accept                         approve the last exchange as a KO/DR pair
  /teach RESPONSE                 pair RESPONSE with the last owner prompt
  /learn PATH                     owner-approve a local JSONL/JSON pair corpus
  /export                         export approved pairs to local JSONL
  /voice train [SECONDS]          train routed/control offspring across all seeds
  /voice adopt latest|RUN_ID      owner explicitly promotes a candidate lineage
  /voice say TEXT                 speak from the active lineage
  /voice say-candidate TEXT       inspect latest candidate without promotion
  /voice status | selftest
  /opcode inspect | selftest | cycle
  /opcode table [GENERATION]       write six nested 64-cell grids as JSON
  /opcode train [SECONDS] [GENERATIONS]  PyTorch charged/flat/shuffled run
  /opcode internalize smoke|full   learn held-out word meanings with paired controls
  /opcode internalize import SUMMARY.json  attach a completed local evidence run
  /opcode hatch smoke|local        evolve a six-byte layer map, then test controls
  /opcode status
  /english train smoke|full        English boxes -> word blocks -> verified Python
  /english import SUMMARY.json     attach a completed held-out run without promotion
  /english status
  /words train smoke|full          word + context -> generated human explanation
  /words import SUMMARY.json       attach a lemma-family-held-out run
  /words explain WORD CONTEXT      inspect latest candidate without promotion
  /words status
  /cards train smoke|full          English flash cards -> Fourier symbols -> Python
  /cards import SUMMARY.json       attach a sealed run without auto-promotion
  /cards status
  /benchmark kaggriculture status
  /benchmark kaggriculture refresh [CANDIDATE_SUBMISSION_ID]
                                   reads target + candidate in one live CLI snapshot
  /benchmark kaggriculture gold [CANDIDATE.py]
                                   profile gold traces + attach held-out receipts
  /benchmark kaggriculture local [CANDIDATE.py] [SEEDS] [SEED_START]
                                   local proxy only; never claims a leaderboard pass
  /inject [N]                     turn approved-example nutrients into larva
  /spawn drone [N]                mature larva into reserve workers
  /promote DRONE_ID               drone -> hatchery; keeps one spare drone
  /hatch [N]                      create 0,+step,-step... candidates
  /plan                           write plain + GeoSeal-bound execution artifacts
  /run probe | /run swarm | /run CANDIDATE_ID
  /budget compute N | storage N   owner sets the local resource envelope
  /quit

Explicit /run commands execute immediately. No generated shell commands are used."""

    def handle_command(self, line: str) -> Optional[str]:
        parts = split_cli_line(line)
        if not parts:
            return ""
        command = parts[0].lower()
        args = parts[1:]
        if command in {"/quit", "/exit"}:
            return None
        if command == "/help":
            return self.help_text()
        if command == "/status":
            return self.status_text()
        if command in {"/representations", "/topology"}:
            if args:
                raise ValueError("usage: /representations")
            return json.dumps(representation_topology_status(), indent=2)
        if command == "/goal":
            if not args:
                return self.state["goal"]
            self.state["goal"] = " ".join(args)
            self.record("goal_changed", {"goal": self.state["goal"]})
            return f"goal = {self.state['goal']}"
        if command == "/lane":
            if not args:
                return "lanes: " + ", ".join(sorted(LANES))
            self.set_lane(args[0])
            return f"lane = {args[0]}"
        if command == "/geometry":
            if not args:
                return json.dumps(GEOMETRY_BINDINGS, indent=2)
            chosen = self.set_geometry(args)
            return f"geometry = {chosen}; routed lane = {self.state['lane']}"
        if command == "/axis":
            if len(args) != 3 or args[0] != "seconds":
                raise ValueError("usage: /axis seconds CENTER STEP")
            axis = self.state["training"]["axis"]
            axis.update({"name": "seconds", "center": max(1, int(args[1])), "step": max(1, int(args[2]))})
            self.record("axis_changed", axis.copy())
            return f"axis = {axis}"
        if command == "/seeds":
            if len(args) != 1:
                raise ValueError("usage: /seeds N")
            count = max(1, int(args[0]))
            base = self.state["training"]["seeds"][0]
            self.state["training"]["seeds"] = [base + i for i in range(count)]
            self.record("seeds_changed", {"seeds": self.state["training"]["seeds"]})
            suffix = "" if count >= 3 else " (UNDERPOWERED: useful smoke, not a claim)"
            return f"seeds = {self.state['training']['seeds']}{suffix}"
        if command == "/data":
            if not args:
                raise ValueError("usage: /data PATH [train|holdout|control]")
            role = args[1] if len(args) > 1 else "train"
            return json.dumps(self.add_dataset(args[0], role), indent=2)
        if command == "/peer":
            if not args or args[0] == "list":
                return f"peers={self.state['peers']} active={self.state.get('active_peer')}"
            if args[0] == "off":
                self.state["active_peer"] = None
                self.record("peer_disabled", {})
                return "local peer off; deterministic manager remains"
            if args[0] in {"add", "use"} and len(args) == 2:
                model = args[1]
                if model not in self.state["peers"]:
                    self.state["peers"].append(model)
                self.state["active_peer"] = model
                self.record("peer_selected", {"model": model})
                return f"active local peer = {model}"
            raise ValueError("usage: /peer add MODEL | use MODEL | off | list")
        if command == "/accept":
            last = self.state.get("last_interaction")
            if not last:
                raise ValueError("there is no prior exchange to approve")
            row = self.approve(last["prompt"], last["response"], last.get("source", "manager"))
            return f"approved reversible pair {row['joint_sha256'][:12]} (nutrients +1)"
        if command == "/teach":
            last = self.state.get("last_interaction")
            if not last:
                raise ValueError("talk to Clay first so /teach has a prompt")
            if not args:
                raise ValueError("usage: /teach RESPONSE")
            response = " ".join(args)
            row = self.approve(last["prompt"], response, "owner-correction")
            return f"owner correction stored as paired joint {row['joint_sha256'][:12]}"
        if command == "/learn":
            if len(args) != 1:
                raise ValueError("usage: /learn PATH")
            return json.dumps(self.learn_file(args[0]), indent=2)
        if command == "/export":
            return f"exported: {self.export_examples()}"
        if command == "/voice":
            if not args or args[0] == "status":
                return json.dumps(self.state["voice"], indent=2)
            if args[0] == "train":
                seconds = float(args[1]) if len(args) > 1 else 5.0
                return json.dumps(self.voice_train(seconds), indent=2)
            if args[0] == "say":
                if len(args) < 2:
                    raise ValueError("usage: /voice say TEXT")
                return self.voice_say(" ".join(args[1:]))
            if args[0] == "say-candidate":
                if len(args) < 2:
                    raise ValueError("usage: /voice say-candidate TEXT")
                return self.voice_say(" ".join(args[1:]), candidate=True)
            if args[0] == "adopt":
                selector = args[1] if len(args) > 1 else "latest"
                if len(args) > 2:
                    raise ValueError("usage: /voice adopt latest|RUN_ID")
                return json.dumps(self.voice_adopt(selector), indent=2)
            if args[0] == "selftest":
                completed = subprocess.run(
                    [sys.executable, str(DIALOGUE_QUINE), "selftest"],
                    cwd=str(DIALOGUE_QUINE.parent),
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if completed.returncode != 0:
                    raise RuntimeError(completed.stderr.strip() or "dialogue selftest failed")
                return completed.stdout.strip()
            raise ValueError(
                "usage: /voice train [SECONDS] | adopt latest|RUN_ID | say TEXT | "
                "say-candidate TEXT | status | selftest"
            )
        if command == "/opcode":
            if not args or args[0] == "status":
                return json.dumps(self.state["opcode_graph"], indent=2)
            if args[0] in {"inspect", "selftest", "cycle"}:
                if len(args) != 1:
                    raise ValueError(f"usage: /opcode {args[0]}")
                return self.opcode_diagnostic(args[0])
            if args[0] == "table":
                if len(args) > 2:
                    raise ValueError("usage: /opcode table [GENERATION]")
                generation = int(args[1], 0) if len(args) == 2 else 0
                return f"opcode table = {self.opcode_diagnostic('table', generation)}"
            if args[0] == "train":
                if len(args) > 3:
                    raise ValueError("usage: /opcode train [SECONDS] [GENERATIONS]")
                seconds = float(args[1]) if len(args) >= 2 else 5.0
                generations = int(args[2]) if len(args) >= 3 else 192
                return json.dumps(self.opcode_train(seconds, generations), indent=2)
            if args[0] == "hatch":
                if len(args) > 2:
                    raise ValueError("usage: /opcode hatch [smoke|local]")
                profile = args[1].lower() if len(args) == 2 else "smoke"
                return json.dumps(self.opcode_hatch(profile), indent=2)
            if args[0] == "internalize":
                if len(args) == 3 and args[1].lower() == "import":
                    return json.dumps(self.opcode_internalize_import(args[2]), indent=2)
                if len(args) > 2:
                    raise ValueError(
                        "usage: /opcode internalize [smoke|full] | import SUMMARY.json"
                    )
                profile = args[1].lower() if len(args) == 2 else "full"
                return json.dumps(self.opcode_internalize(profile), indent=2)
            raise ValueError(
                "usage: /opcode inspect | selftest | cycle | table [GENERATION] | "
                "train [SECONDS] [GENERATIONS] | hatch [smoke|local] | "
                "internalize [smoke|full] | internalize import SUMMARY.json | status"
            )
        if command == "/english":
            if not args or args[0] == "status":
                return json.dumps(self.state["english_blocks"], indent=2)
            if args[0] == "train":
                if len(args) > 2:
                    raise ValueError("usage: /english train [smoke|full]")
                profile = args[1].lower() if len(args) == 2 else "full"
                return json.dumps(self.english_blocks_train(profile), indent=2)
            if args[0] == "import" and len(args) == 2:
                return json.dumps(self.english_blocks_import(args[1]), indent=2)
            raise ValueError(
                "usage: /english train [smoke|full] | import SUMMARY.json | status"
            )
        if command in {"/words", "/meanings"}:
            if not args or args[0] == "status":
                return json.dumps(self.state["word_meanings"], indent=2)
            if args[0] == "train":
                if len(args) > 2:
                    raise ValueError("usage: /words train [smoke|full]")
                profile = args[1].lower() if len(args) == 2 else "full"
                return json.dumps(self.word_meanings_train(profile), indent=2)
            if args[0] == "import" and len(args) == 2:
                return json.dumps(self.word_meanings_import(args[1]), indent=2)
            if args[0] == "explain" and len(args) >= 3:
                context = " ".join(args[2:]).strip()
                if context.startswith("|"):
                    context = context[1:].strip()
                return json.dumps(
                    self.word_meanings_explain(args[1], context),
                    indent=2,
                )
            raise ValueError(
                "usage: /words train [smoke|full] | import SUMMARY.json | "
                "explain WORD [|] CONTEXT | status"
            )
        if command == "/cards":
            if not args or args[0] == "status":
                return json.dumps(self.state["fourier_cards"], indent=2)
            if args[0] == "train":
                if len(args) > 2:
                    raise ValueError("usage: /cards train [smoke|full]")
                profile = args[1].lower() if len(args) == 2 else "full"
                return json.dumps(self.fourier_cards_train(profile), indent=2)
            if args[0] == "import" and len(args) == 2:
                return json.dumps(self.fourier_cards_import(args[1]), indent=2)
            raise ValueError(
                "usage: /cards train [smoke|full] | import SUMMARY.json | status"
            )
        if command == "/benchmark":
            if not args or args[0].lower() != "kaggriculture":
                raise ValueError(
                    "usage: /benchmark kaggriculture status | refresh [SUBMISSION_ID] | "
                    "gold [CANDIDATE.py] | "
                    "/benchmark kaggriculture local [CANDIDATE.py] [SEEDS] [SEED_START]"
                )
            action = args[1].lower() if len(args) >= 2 else "status"
            if action == "status" and len(args) <= 2:
                return json.dumps(self.kaggriculture_benchmark_status(), indent=2)
            if action == "refresh" and len(args) <= 3:
                candidate_submission_id = int(args[2]) if len(args) == 3 else None
                return json.dumps(
                    self.kaggriculture_refresh_scores(candidate_submission_id),
                    indent=2,
                )
            if action == "gold" and len(args) <= 3:
                candidate = args[2] if len(args) == 3 else "main_goldroute_flat.py"
                return json.dumps(self.kaggriculture_gold_profile(candidate), indent=2)
            if action == "local" and len(args) <= 5:
                candidate = args[2] if len(args) >= 3 else "main_feedback_quine.py"
                seed_count = int(args[3]) if len(args) >= 4 else 12
                seed_start = int(args[4]) if len(args) >= 5 else None
                return json.dumps(
                    self.kaggriculture_local_gate(candidate, seed_count, seed_start),
                    indent=2,
                )
            raise ValueError(
                "usage: /benchmark kaggriculture status | refresh [SUBMISSION_ID] | "
                "gold [CANDIDATE.py] | "
                "/benchmark kaggriculture local [CANDIDATE.py] [SEEDS] [SEED_START]"
            )
        if command == "/inject":
            count = int(args[0]) if args else 1
            return f"created {self.inject(count)} larva"
        if command == "/spawn":
            if not args:
                raise ValueError("usage: /spawn drone [N] | /spawn probe [N] | /spawn hatchling [N]")
            count = int(args[1]) if len(args) > 1 else 1
            if args[0] == "drone":
                return f"drones = {self.spawn_drones(count)}"
            if args[0] in {"probe", "hatchling"}:
                made = self.hatch(count)
                return "candidates = " + ", ".join(f"{c['id']}({c['value']}s)" for c in made)
            raise ValueError("spawn kind must be drone, probe, or hatchling")
        if command == "/promote":
            if len(args) != 1:
                raise ValueError("usage: /promote DRONE_ID")
            return f"new hatchery = {self.promote_drone(args[0])}"
        if command == "/hatch":
            count = int(args[0]) if args else 3
            made = self.hatch(count)
            return "candidates = " + ", ".join(f"{c['id']}({c['value']}s)" for c in made)
        if command == "/plan":
            return f"plan = {self.write_plan()}"
        if command == "/run":
            if not args:
                raise ValueError("usage: /run probe | swarm | CANDIDATE_ID")
            if args[0] == "probe":
                result = self.run_probe()
                return json.dumps(result, indent=2)
            if args[0] == "swarm":
                results = self.run_swarm()
                return json.dumps(results, indent=2)
            return json.dumps(self.run_candidate(args[0]), indent=2)
        if command == "/budget":
            if len(args) != 2 or args[0] not in {"compute", "storage"}:
                raise ValueError("usage: /budget compute N | /budget storage N")
            value = max(0, int(args[1]))
            key = "compute_seconds_remaining" if args[0] == "compute" else "storage_mb_remaining"
            self.state["resources"][key] = value
            self.record("budget_changed", {"resource": key, "value": value, "authority": "owner"})
            return f"{key} = {value}"
        raise ValueError(f"unknown command {command}; use /help")


def repl(studio: ClayTrainingStudio) -> int:
    print("Clay Training Studio — local manager, real graders, sealed receipts")
    print("Talk normally or type /help. Explicit /run starts work. Ctrl+C returns to the prompt.")
    print(studio.status_text())
    while True:
        try:
            line = input("clay.train> ").strip()
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print("\n(interrupted; session is already saved)")
            continue
        if not line:
            continue
        try:
            if line.startswith("/"):
                response = studio.handle_command(line)
                if response is None:
                    return 0
                if response:
                    print(response)
                continue
            result = studio.interpret(line)
            print(f"[{result['source']}] {result['response']}")
            if result["changes"]:
                print("[routed] " + json.dumps(result["changes"], ensure_ascii=False))
            if result["run_requested"]:
                print("[owner intent] running one measured probe")
                print(json.dumps(studio.run_probe(), indent=2))
            if result["voice_run_requested"]:
                print("[owner intent] training one measured dialogue offspring")
                print(json.dumps(studio.voice_train(), indent=2))
            if result["internalization_run_requested"]:
                profile = result["internalization_profile"]
                print(f"[owner intent] running {profile} held-out representation training")
                print(json.dumps(studio.opcode_internalize(profile), indent=2))
            if result["word_meaning_run_requested"]:
                profile = result["word_meaning_profile"]
                print(f"[owner intent] running {profile} generative word-meaning training")
                print(json.dumps(studio.word_meanings_train(profile), indent=2))
            if result["english_block_run_requested"]:
                profile = result["english_block_profile"]
                print(f"[owner intent] running {profile} English block/function training")
                print(json.dumps(studio.english_blocks_train(profile), indent=2))
            if result["fourier_card_run_requested"]:
                profile = result["fourier_card_profile"]
                print(f"[owner intent] running {profile} Fourier flash-card training")
                print(json.dumps(studio.fourier_cards_train(profile), indent=2))
        except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
        except KeyboardInterrupt:
            print("\n(run interrupted; receipts up to the interruption are saved)")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=DEFAULT_HOME)
    parser.add_argument("--session", default="default")
    parser.add_argument("--model", help="optional local Ollama peer model")
    parser.add_argument("--command", help="run one slash command or conversational instruction, then exit")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args(argv)

    studio = ClayTrainingStudio(args.home, args.session, args.model)
    if args.status:
        print(studio.status_text())
        return 0
    if args.command:
        try:
            if args.command.startswith("/"):
                result = studio.handle_command(args.command)
                if result:
                    print(result)
            else:
                result = studio.interpret(args.command)
                print(f"[{result['source']}] {result['response']}")
                if result["internalization_run_requested"]:
                    print(
                        json.dumps(
                            studio.opcode_internalize(result["internalization_profile"]),
                            indent=2,
                        )
                    )
                if result["word_meaning_run_requested"]:
                    print(
                        json.dumps(
                            studio.word_meanings_train(result["word_meaning_profile"]),
                            indent=2,
                        )
                    )
                if result["english_block_run_requested"]:
                    print(
                        json.dumps(
                            studio.english_blocks_train(result["english_block_profile"]),
                            indent=2,
                        )
                    )
                if result["fourier_card_run_requested"]:
                    print(
                        json.dumps(
                            studio.fourier_cards_train(result["fourier_card_profile"]),
                            indent=2,
                        )
                    )
            return 0
        except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    return repl(studio)


if __name__ == "__main__":
    raise SystemExit(main())
