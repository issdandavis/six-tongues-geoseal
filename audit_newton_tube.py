"""Deterministic exactness and size audit for the Newton operation tube."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from math import comb
from pathlib import Path
import random
import time

import newton_tube

ROOT = Path(__file__).resolve().parent


def metric(name: str, values: list[int]) -> dict:
    started = time.perf_counter()
    face = newton_tube.newton_face(values)
    elapsed = time.perf_counter() - started
    trits = sum(len(spelling) for spelling in face["bal3"])
    input_bits = len(values) * 8
    ideal_feature_bits = trits * math.log2(3)
    return {
        "name": name,
        "length": len(values),
        "degree": face["degree"],
        "seconds": round(elapsed, 6),
        "max_coefficient_bits": max(
            (abs(coefficient).bit_length() for coefficient in face["leading"]),
            default=0,
        ),
        "balanced_trits": trits,
        "ideal_feature_bits": round(ideal_feature_bits, 2),
        "input_bits": input_bits,
        "ideal_expansion": (
            round(ideal_feature_bits / input_bits, 3) if input_bits else None
        ),
    }


def audit(seed: int = 191026, round_trips: int = 1000, max_length: int = 128) -> dict:
    rng = random.Random(seed)
    prefix_checks = 0
    for _ in range(round_trips):
        length = rng.randrange(max_length + 1)
        values = [rng.randrange(256) for _ in range(length)]
        face = newton_tube.newton_face(values)
        if newton_tube.recover_leading(face["leading"]) != values:
            raise RuntimeError("full Newton recovery failed")
        if newton_tube.recover_active(face["active"], length) != values:
            raise RuntimeError("trimmed Newton recovery failed")
        for order, coefficient in enumerate(face["leading"]):
            prefix_coefficient = sum(
                ((-1) ** (order - index)) * comb(order, index) * values[index]
                for index in range(order + 1)
            )
            if prefix_coefficient != coefficient:
                raise RuntimeError("coefficient used a value outside its prefix")
            prefix_checks += 1
        if not all(
            newton_tube.rails_hold(coefficient, rails)
            for coefficient, rails in zip(face["leading"], face["rails"])
        ):
            raise RuntimeError("rail split failed")
        if not all(
            newton_tube.from_bal3(spelling) == coefficient
            for coefficient, spelling in zip(face["leading"], face["bal3"])
        ):
            raise RuntimeError("balanced ternary recovery failed")

    metric_rng = random.Random(seed + 1)
    readme = (ROOT / "README.md").read_bytes()
    rows = [
        metric(
            f"random-{length}",
            [metric_rng.randrange(256) for _ in range(length)],
        )
        for length in (16, 64, 128, 256, 512)
    ]
    rows.extend(
        [
            metric("ramp-256", list(range(256))),
            metric("readme-512", list(readme[:512])),
            metric("repeating-512", list((b"abc " * 128)[:512])),
            metric("zeros-512", [0] * 512),
        ]
    )
    return {
        "schema": "six-tongues.newton-operation-tube-audit.v1",
        "status": "PASS",
        "seed": seed,
        "random_round_trips": round_trips,
        "max_random_length": max_length,
        "causal_prefix_coefficients": prefix_checks,
        "readme_sha256": sha256(readme).hexdigest(),
        "metrics": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=191026)
    parser.add_argument("--round-trips", type=int, default=1000)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()
    if args.round_trips < 1 or args.max_length < 0:
        parser.error("round-trips must be positive and max-length non-negative")
    print(
        json.dumps(
            audit(args.seed, args.round_trips, args.max_length),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
