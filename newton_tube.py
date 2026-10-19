"""Operation tube: one byte board, read as several streams.

The flat board is the byte string. Operation order is the index of those
bytes. This module adds the reading that the tongue roads, the trit seam,
and the four-rail tube already imply. It does not add a fifth side. The
counts already in the body of work are six tongue roads, three trit states,
four rails, and seven semantic elements.

Roads, both ways:

* the board bytes
* the Newton leading coefficients along the operation index, kept at the
  board's length so a trailing zero edge is still a digit
* the balanced-ternary spelling of each coefficient (low place first, the
  same remainder rule as loom ``balanced_state.to_bal3``)

Views of that same road:

* polarity, the trit sign of each coefficient: -1, 0, +1
* four rails per coefficient: positive vein, negative vein, both, and the
  zero rail. ``both`` is the coefficient. The zero rail is the edge.
* the step tangent, the first difference between consecutive operations

The degree is the last order whose leading coefficient is not zero. An
internal zero stays in the coefficient vector. Only orders past the degree
are the trailing zero edge. An all-zero board of length n is that edge for
every order, and the length is what recovers it.

The atomic tokenizer is a destination. Word stamps sit beside the board.
An unresolved stamp leaves the element empty so a fallback class is not
reported as a measured element. GeoSeal, when the caller asks for it, binds
the board bytes. It is the clasp, not another alphabet.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from math import comb
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


SHAPE: dict[str, Any] = {
    "cross_section": "six tongue roads on one byte board",
    "axis": "operation index",
    "seam": "trit polarity of each Newton coefficient",
    "rails": ["positive", "negative", "both", "zero"],
    "atomic": "seven-element destination consulted beside the board",
}

_ATOMIC_MODULE: Any = None
_ATOMIC_TRIED = False


def trit_sign(coefficient: int) -> int:
    """Return the trit seam of one integer: -1, 0, or +1."""
    if coefficient > 0:
        return 1
    if coefficient < 0:
        return -1
    return 0


def coefficient_rails(coefficient: int) -> dict[str, Any]:
    """Split one coefficient into the four rails of the existing tube.

    ``both`` is the coefficient, so this view round-trips. The positive and
    negative veins cannot both carry a value. The zero rail is the edge.
    """
    positive = coefficient if coefficient > 0 else 0
    negative = coefficient if coefficient < 0 else 0
    return {
        "positive": positive,
        "negative": negative,
        "both": coefficient,
        "zero": coefficient == 0,
    }


def rails_hold(coefficient: int, rails: Mapping[str, Any]) -> bool:
    """True when the four rails are the split of this coefficient."""
    positive = rails["positive"]
    negative = rails["negative"]
    return (
        positive + negative == coefficient
        and rails["both"] == coefficient
        and bool(rails["zero"]) == (coefficient == 0)
        and not (positive and negative)
    )


def to_bal3(number: int) -> list[int]:
    """Balanced ternary, low place first. Zero is the one-digit edge [0]."""
    digits: list[int] = []
    value = int(number)
    while value != 0:
        remainder = value % 3
        value //= 3
        if remainder == 2:
            remainder = -1
            value += 1
        digits.append(remainder)
    return digits or [0]


def from_bal3(digits: Sequence[int]) -> int:
    """Invert ``to_bal3``. Place i has weight 3**i."""
    return sum(int(digit) * (3**place) for place, digit in enumerate(digits))


def difference_rows(samples: Sequence[int]) -> list[list[int]]:
    """Forward-difference triangle. Row k is the tangent of order k."""
    rows = [list(samples)]
    while len(rows[-1]) > 1:
        previous = rows[-1]
        rows.append([previous[i + 1] - previous[i] for i in range(len(previous) - 1)])
    return rows


def recover_leading(leading: Sequence[int]) -> list[int]:
    """Recover the operation sequence from Newton leading coefficients.

    a(k) = sum_j C(k, j) * leading[j]. The length of ``leading`` is the
    length of the sequence.
    """
    coefficients = [int(coefficient) for coefficient in leading]
    length = len(coefficients)
    return [
        sum(comb(index, order) * coefficients[order] for order in range(index + 1))
        for index in range(length)
    ]


def recover_active(active: Sequence[int], length: int) -> list[int]:
    """Recover a sequence from its degree-trimmed coefficients and its length.

    Orders past ``active`` are the trailing zero edge. An internal zero inside
    ``active`` is kept. The length is required: the same coefficients sampled
    at two lengths are two boards.
    """
    if length < 0:
        raise ValueError("length must be non-negative")
    if length < len(active):
        raise ValueError("length is shorter than the active coefficients")
    leading = [int(coefficient) for coefficient in active] + [0] * (length - len(active))
    return recover_leading(leading)


def newton_face(samples: Sequence[int]) -> dict[str, Any]:
    """Build the operation-order face of an integer sequence.

    Raises if the binomial recovery disagrees with the samples. That
    disagreement is a broken road, not a record to return.
    """
    values = [int(sample) for sample in samples]
    length = len(values)
    if length == 0:
        return {
            "length": 0,
            "degree": None,
            "all_edge": False,
            "samples": [],
            "leading": [],
            "polarity": [],
            "rails": [],
            "bal3": [],
            "active": [],
            "step_tangent": [],
            "closed": True,
        }

    rows = difference_rows(values)
    leading = [row[0] for row in rows]
    degree = None
    for order, coefficient in enumerate(leading):
        if coefficient != 0:
            degree = order
    recovered = recover_leading(leading)
    if recovered != values:
        raise RuntimeError("Newton face did not recover the operation sequence")
    active = [] if degree is None else leading[: degree + 1]
    if recover_active(active, length) != values:
        raise RuntimeError("trimmed Newton coefficients did not recover the sequence")

    step_row = rows[1] if length > 1 else []
    return {
        "length": length,
        "degree": degree,
        "all_edge": degree is None,
        "samples": values,
        "leading": leading,
        "polarity": [trit_sign(coefficient) for coefficient in leading],
        "rails": [coefficient_rails(coefficient) for coefficient in leading],
        "bal3": [to_bal3(coefficient) for coefficient in leading],
        "active": active,
        "step_tangent": [
            {
                "at": index,
                "delta": delta,
                "polarity": trit_sign(delta),
                "rails": coefficient_rails(delta),
            }
            for index, delta in enumerate(step_row)
        ],
        "closed": True,
    }


def operation_tube(board: bytes) -> dict[str, Any]:
    """Newton face of a byte board. Each byte is one operation."""
    face = newton_face(list(board))
    face["board_hex"] = board.hex()
    return face


def _atomic_candidates() -> list[Path]:
    configured = os.environ.get("SCBE_ATOMIC")
    candidates = []
    if configured:
        candidates.append(Path(configured))
    candidates.append(
        Path(r"C:\Users\issda\SCBE-AETHERMOORE\python\scbe\atomic_tokenization.py")
    )
    candidates.append(Path(r"C:\dev\SCBE-AETHERMOORE\python\scbe\atomic_tokenization.py"))
    return candidates


def atomic_module() -> Any:
    """Load the SCBE atomic tokenizer once, or return None when it is absent."""
    global _ATOMIC_MODULE, _ATOMIC_TRIED
    if _ATOMIC_TRIED:
        return _ATOMIC_MODULE
    _ATOMIC_TRIED = True
    for path in _atomic_candidates():
        if not path.is_file():
            continue
        spec = importlib.util.spec_from_file_location("scbe_atomic_tokenization_tube", path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        # dataclasses look the class module up in sys.modules during decoration.
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(spec.name, None)
            raise
        _ATOMIC_MODULE = module
        break
    return _ATOMIC_MODULE


def stamp_words(words: Sequence[str]) -> dict[str, Any]:
    """Destination stamps for linguistic tokens.

    These stamps are not the samples of the polynomial. An unresolved token
    keeps its class and source and leaves the element empty.
    """
    module = atomic_module()
    if module is None:
        return {
            "role": "destination",
            "present": True,
            "available": False,
            "reason": "atomic tokenizer not found",
            "stamps": [],
        }
    stamps = []
    for word in words:
        state = module.map_token_to_atomic_state(word)
        resolved = bool(state.semantic_resolved)
        stamps.append(
            {
                "token": state.token,
                "semantic_class": state.semantic_class,
                "semantic_resolved": resolved,
                "semantic_source": state.semantic_source,
                "element": state.element.symbol if resolved else None,
                "trit": state.tau.as_dict() if resolved else None,
                "negative_state": bool(state.negative_state) if resolved else None,
                "dual_state": state.dual_state if resolved else None,
            }
        )
    return {
        "role": "destination",
        "present": True,
        "available": True,
        "stamps": stamps,
    }


def stream_record(
    board: bytes,
    *,
    tongues: Optional[Mapping[str, str]] = None,
    atomic: Optional[Mapping[str, Any]] = None,
    seal: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """One board plus the streams a caller was able to attach."""
    return {
        "shape": SHAPE,
        "newton": operation_tube(board),
        "tongues": None if tongues is None else dict(tongues),
        "atomic": dict(atomic) if atomic is not None else {"role": "destination", "present": False},
        "seal": None if seal is None else dict(seal),
    }
