"""Round-trip checks for the operation tube.

The Newton checks are algebra on stated sequences. The tongue check uses the
byte codec in this repo. The atomic check uses the SCBE tokenizer when that
file is on this machine. Nothing here is a model score.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

import aethermoore
import newton_tube


ROOT = Path(__file__).resolve().parent
TRIANGULAR = bytes([1, 3, 6, 10, 15])
MIXED = bytes([1, 4, 2, 8, 3])


class NewtonFaceTests(unittest.TestCase):
    def test_triangular_degree_and_recovery(self) -> None:
        face = newton_tube.operation_tube(TRIANGULAR)
        self.assertEqual(face["samples"], [1, 3, 6, 10, 15])
        self.assertEqual(face["leading"], [1, 2, 1, 0, 0])
        self.assertEqual(face["polarity"], [1, 1, 1, 0, 0])
        self.assertEqual(face["degree"], 2)
        self.assertFalse(face["all_edge"])
        self.assertEqual(face["active"], [1, 2, 1])
        self.assertTrue(face["closed"])
        self.assertEqual(
            newton_tube.recover_active(face["active"], face["length"]),
            face["samples"],
        )
        self.assertEqual(face["step_tangent"][0]["delta"], 2)
        self.assertEqual(face["step_tangent"][0]["polarity"], 1)

    def test_sign_changes_keep_their_polarity(self) -> None:
        face = newton_tube.newton_face(list(MIXED))
        self.assertEqual(face["leading"], [1, 3, -5, 13, -32])
        self.assertEqual(face["polarity"], [1, 1, -1, 1, -1])
        self.assertEqual(face["degree"], 4)
        for coefficient, rails in zip(face["leading"], face["rails"]):
            self.assertTrue(newton_tube.rails_hold(coefficient, rails))
        for coefficient, digits in zip(face["leading"], face["bal3"]):
            self.assertEqual(newton_tube.from_bal3(digits), coefficient)

    def test_internal_zero_is_kept_and_trailing_zeros_trim(self) -> None:
        bumped = newton_tube.newton_face([1, 1, 2])
        self.assertEqual(bumped["leading"], [1, 0, 1])
        self.assertEqual(bumped["degree"], 2)
        self.assertEqual(bumped["active"], [1, 0, 1])
        self.assertEqual(bumped["step_tangent"][0]["polarity"], 0)

        flat = newton_tube.newton_face([5, 5, 5])
        self.assertEqual(flat["degree"], 0)
        self.assertEqual(flat["active"], [5])
        self.assertEqual(newton_tube.recover_active([5], 3), [5, 5, 5])

    def test_zero_board_keeps_its_length(self) -> None:
        face = newton_tube.operation_tube(bytes(4))
        self.assertEqual(face["length"], 4)
        self.assertIsNone(face["degree"])
        self.assertTrue(face["all_edge"])
        self.assertEqual(face["active"], [])
        self.assertEqual(face["polarity"], [0, 0, 0, 0])
        self.assertEqual(newton_tube.recover_active([], 4), [0, 0, 0, 0])

    def test_empty_board(self) -> None:
        face = newton_tube.operation_tube(b"")
        self.assertEqual(face["length"], 0)
        self.assertIsNone(face["degree"])
        self.assertFalse(face["all_edge"])
        self.assertEqual(face["samples"], [])
        self.assertTrue(face["closed"])

    def test_balanced_ternary_negation_is_the_sign_flip(self) -> None:
        for number in (-13, -1, 0, 1, 7, 32):
            digits = newton_tube.to_bal3(number)
            self.assertEqual(newton_tube.from_bal3(digits), number)
            self.assertEqual(newton_tube.from_bal3([-digit for digit in digits]), -number)
            self.assertEqual(newton_tube.to_bal3(0), [0])


class StreamTests(unittest.TestCase):
    def test_six_tongues_are_one_board(self) -> None:
        tongues = {tongue: aethermoore.encode(TRIANGULAR, tongue) for tongue in aethermoore.TONGUES}
        self.assertEqual(len(tongues), 6)
        self.assertEqual(len(set(tongues.values())), 6)
        for tongue, stream in tongues.items():
            self.assertEqual(aethermoore.decode(stream, tongue), TRIANGULAR)
        record = newton_tube.stream_record(TRIANGULAR, tongues=tongues)
        self.assertEqual(record["newton"]["degree"], 2)
        self.assertFalse(record["atomic"]["present"])
        self.assertIsNone(record["seal"])
        self.assertEqual(record["shape"]["rails"], ["positive", "negative", "both", "zero"])

    def test_atomic_stamps_stay_beside_the_board(self) -> None:
        stamped = newton_tube.stamp_words(["not", "run", "feature_0"])
        if not stamped["available"]:
            self.skipTest(stamped["reason"])
        negation, action, synthetic = stamped["stamps"]
        self.assertEqual(negation["element"], "Cl")
        self.assertTrue(negation["semantic_resolved"])
        self.assertTrue(negation["negative_state"])
        self.assertEqual(negation["dual_state"], 1)
        self.assertEqual(action["element"], "Na")
        self.assertEqual(action["dual_state"], 0)
        self.assertFalse(action["negative_state"])
        self.assertFalse(synthetic["semantic_resolved"])
        self.assertIsNone(synthetic["element"])
        self.assertEqual(synthetic["semantic_source"], "synthetic-or-nonlexical")

    def test_seal_binds_the_same_board(self) -> None:
        os.environ["GEOSEAL_KEY"] = "newton-tube-unit-key"
        sealed = aethermoore.geoseal_encrypt(TRIANGULAR, lat=0.0, lon=0.0, tag="tube")
        opened = aethermoore.geoseal_decrypt(sealed, expect_tag="tube")
        self.assertEqual(opened, TRIANGULAR)
        record = newton_tube.stream_record(TRIANGULAR, seal=sealed)
        self.assertEqual(record["seal"]["context"]["version"], 1)
        self.assertEqual(bytes.fromhex(record["seal"]["payload_hex"]), TRIANGULAR)
        self.assertNotIn("key", record["seal"])

    def test_streams_command_prints_the_degree(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(ROOT / "aethermoore.py"), "streams", "--hex", TRIANGULAR.hex()],
            check=False,
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        record = json.loads(completed.stdout)
        self.assertEqual(record["newton"]["degree"], 2)
        self.assertEqual(record["newton"]["leading"], [1, 2, 1, 0, 0])
        self.assertEqual(len(record["tongues"]), 6)
        for tongue, stream in record["tongues"].items():
            self.assertEqual(aethermoore.decode(stream, tongue), TRIANGULAR)
        self.assertFalse(record["atomic"]["present"])
        self.assertIsNone(record["seal"])


if __name__ == "__main__":
    unittest.main()
