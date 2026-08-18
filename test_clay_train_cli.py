from __future__ import annotations

import json
import os
import tempfile
import unittest
import subprocess
from pathlib import Path
from unittest import mock

from clay_train_cli import (
    KAGGRICULTURE_TARGET,
    ClayTrainingStudio,
    alternating_offsets,
    file_sha256,
    parse_kaggle_submission_snapshot,
    representation_topology_status,
    split_cli_line,
)


class ClayTrainingStudioTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.key = mock.patch.dict(os.environ, {"GEOSEAL_KEY": "unit-test-owner-key"})
        self.key.start()
        self.addCleanup(self.key.stop)
        self.studio = ClayTrainingStudio(Path(self.temp.name), "test")

    def test_state_and_event_chain_survive_reload(self) -> None:
        self.studio.handle_command("/goal teach Clay from owner corrections")
        ok, count, _ = self.studio.verify_events()
        self.assertTrue(ok)
        self.assertGreaterEqual(count, 2)

        reopened = ClayTrainingStudio(Path(self.temp.name), "test")
        self.assertEqual(reopened.state["goal"], "teach Clay from owner corrections")
        self.assertTrue(reopened.verify_events()[0])

    def test_windows_paths_survive_command_parsing(self) -> None:
        raw = r'/learn "C:\dev\Clay Data\pairs.jsonl"'
        self.assertEqual(split_cli_line(raw), ["/learn", r"C:\dev\Clay Data\pairs.jsonl"])

    def test_paired_joint_recovers_same_payload_both_directions(self) -> None:
        row = self.studio.paired_record("name bracelet", "two readings, one joint")
        self.assertTrue(row["verified"])
        self.assertEqual(row["forward"]["tongue"], "KO")
        self.assertEqual(row["reverse"]["tongue"], "DR")
        self.assertEqual({route["tongue"] for route in row["routes"]}, {"KO", "AV", "RU", "CA", "UM", "DR"})
        self.assertEqual(row["canonical"]["response"], "two readings, one joint")

    def test_approved_pair_is_redacted_sealed_and_exportable(self) -> None:
        row = self.studio.approve("use api_key=do-not-store", "done", "owner")
        self.assertNotIn("do-not-store", json.dumps(row))
        examples = list(self.studio.iter_examples())
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["joint_sha256"], row["joint_sha256"])
        wrapper = json.loads(self.studio.examples_path.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(wrapper["codec"], "zlib-json-v1")
        self.assertNotIn("tokens", json.dumps(wrapper))
        exported = self.studio.export_examples()
        self.assertTrue(exported.exists())
        self.assertNotIn("do-not-store", exported.read_text(encoding="utf-8"))

    def test_owner_bulk_learning_is_deduplicated_and_recoverable(self) -> None:
        source = Path(self.temp.name) / "pairs.jsonl"
        rows = [
            {"prompt": "belt", "response": "tokenizer"},
            {"prompt": "gear", "target": "digit"},
            {"prompt": "belt", "response": "tokenizer"},
        ]
        source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        receipt = self.studio.learn_file(str(source))
        self.assertEqual(receipt["accepted"], 2)
        self.assertEqual(receipt["duplicates"], 1)
        self.assertEqual(len(list(self.studio.iter_examples())), 2)

    def test_owner_can_adopt_unqualified_hash_verified_candidate(self) -> None:
        checkpoint = Path(self.temp.name) / "candidate.pt"
        checkpoint.write_bytes(b"candidate-state")
        run = {
            "run_id": "voice-test",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint),
            "generation": 2,
            "verdict": {"status": "NO_CLAIM"},
        }
        self.studio.state["voice"]["latest_checkpoint"] = str(checkpoint)
        self.studio.state["voice"]["runs"].append(run)
        self.studio.state["drones"].append({
            "id": "voice-drone",
            "source": "voice-test",
            "status": "candidate",
        })
        promotion = self.studio.voice_adopt("latest")
        self.assertEqual(promotion["basis"], "owner-override")
        self.assertEqual(self.studio.state["voice"]["active_checkpoint"], str(checkpoint))
        self.assertEqual(self.studio.state["drones"][-1]["status"], "reserve")

    def test_geometry_routes_to_existing_lane_and_keeps_status(self) -> None:
        response = self.studio.handle_command("/geometry curve interweave reverse wafer")
        self.assertIn("inverse_routing", response)
        self.assertEqual(self.studio.state["lane"], "inverse_routing")
        plan = self.studio.write_plan([])
        payload = json.loads(plan.read_text(encoding="utf-8"))
        self.assertEqual(payload["geometry_bindings"]["wafer"]["status"], "executable-diagnostic")
        self.assertEqual(payload["geometry_bindings"]["curve"]["status"], "trainable")
        self.assertEqual(
            payload["representation_topology"]["external_anchors"]["STISA"]["ownership"],
            "external-reference; do not copy into Clay",
        )

    def test_representation_topology_uses_existing_translation_seams(self) -> None:
        topology = representation_topology_status()
        self.assertEqual(topology["census"]["farm_lanes_with_five_or_more"], 4)
        self.assertEqual(len(topology["farm_translation_lanes"]), 4)
        self.assertTrue(all(row["exists"] for row in topology["farm_translation_lanes"].values()))
        stisa = topology["external_anchors"]["STISA"]
        self.assertTrue(stisa["exists"])
        self.assertEqual(len(stisa["sha256"]), 64)
        self.assertIn("SCBE-AETHERMOORE", stisa["path"])

    def test_hill_candidates_are_centre_forward_reverse_pairs(self) -> None:
        self.assertEqual(alternating_offsets(5), [0, 1, -1, 2, -2])
        made = self.studio.hatch(3)
        self.assertEqual([candidate["value"] for candidate in made], [30, 40, 20])
        self.assertEqual(self.studio.state["resources"]["larva"], 0)
        argv = self.studio.trainer_argv(made[0])
        self.assertIn("--seed-list", argv)
        self.assertNotIn("--promote", argv)
        self.assertNotIsInstance(argv, str)

    def test_drone_to_hatchery_never_consumes_last_spare(self) -> None:
        with self.assertRaisesRegex(ValueError, "one more drone"):
            self.studio.promote_drone("d0")
        made = self.studio.spawn_drones(1)
        self.assertEqual(len(made), 1)
        hatchery = self.studio.promote_drone(made[0])
        self.assertTrue(hatchery.startswith("h"))
        reserves = [d for d in self.studio.state["drones"] if d["status"] == "reserve"]
        self.assertEqual(len(reserves), 1)
        self.assertEqual(len(self.studio.state["hatcheries"]), 2)

    def test_underpowered_generation_does_not_move_axis(self) -> None:
        self.studio.state["candidates"] = [
            {"id": "centre", "generation": 1, "value": 30, "seed_metrics": [0.50, 0.51], "mean_metric": 0.505},
            {"id": "up", "generation": 1, "value": 40, "seed_metrics": [0.70, 0.71], "mean_metric": 0.705},
        ]
        verdict = self.studio._evaluate_generation(1)
        self.assertEqual(verdict["status"], "UNDERPOWERED")
        self.assertEqual(self.studio.state["training"]["axis"]["center"], 30)

    def test_three_seed_clear_step_moves_hill(self) -> None:
        self.studio.state["candidates"] = [
            {"id": "centre", "generation": 1, "value": 30, "seed_metrics": [0.50, 0.50, 0.50], "mean_metric": 0.50},
            {"id": "up", "generation": 1, "value": 40, "seed_metrics": [0.70, 0.70, 0.70], "mean_metric": 0.70},
        ]
        verdict = self.studio._evaluate_generation(1)
        self.assertEqual(verdict["status"], "HILL_STEP_ACCEPTED")
        self.assertEqual(verdict["direction"], "up")
        self.assertEqual(self.studio.state["training"]["axis"]["center"], 40)

    def test_opcode_table_is_written_as_a_durable_json_receipt(self) -> None:
        payload = {
            "schema": "clay.opcode-quine.six-grid-table.v1",
            "roundtrips": 384,
            "tables": {tongue: [] for tongue in ("KO", "AV", "RU", "CA", "UM", "DR")},
        }
        completed = subprocess.CompletedProcess([], 0, stdout=json.dumps(payload), stderr="")
        with mock.patch.object(self.studio, "_run_opcode_module", return_value=completed):
            path = Path(self.studio.opcode_diagnostic("table", 7))
        self.assertTrue(path.exists())
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["roundtrips"], 384)
        self.assertEqual(self.studio.state["opcode_graph"]["latest_table"], str(path))

    def test_opcode_training_uses_three_controls_and_records_summary(self) -> None:
        seen: dict[str, object] = {}

        def fake_run(argv, cwd, check):
            seen["argv"] = argv
            seen["cwd"] = cwd
            output = Path(argv[argv.index("--output") + 1])
            output.mkdir(parents=True, exist_ok=True)
            (output / "summary.json").write_text(json.dumps({
                "verdict": {"status": "UNDERPOWERED"},
                "charge_partition": {"under": 8, "hinge": 43, "over": 13},
                "quine_cycle": {"status": "PASS", "cycle_generations": 192},
            }), encoding="utf-8")
            return subprocess.CompletedProcess(argv, 0)

        with mock.patch("clay_train_cli.subprocess.run", side_effect=fake_run):
            receipt = self.studio.opcode_train(0.05, 2)
        argv = seen["argv"]
        self.assertIn("charged,flat,shuffled", argv)
        self.assertEqual(argv[argv.index("--generations") + 1], "2")
        self.assertEqual(receipt["verdict"]["status"], "UNDERPOWERED")
        self.assertEqual(len(self.studio.state["opcode_graph"]["runs"]), 1)

    def test_opcode_hatchery_keeps_population_four_and_records_controls(self) -> None:
        seen: dict[str, object] = {}

        def fake_run(argv, cwd, check):
            seen["argv"] = argv
            seen["cwd"] = cwd
            output = Path(argv[argv.index("--output") + 1])
            output.mkdir(parents=True, exist_ok=True)
            (output / "summary.json").write_text(json.dumps({
                "claim_boundary": "structured routes are an experiment",
                "champion": {"genome": {"id": "abc", "hex": "000000000000"}},
                "search": {"candidate_slots": 8},
                "promotion_arms": [
                    {"hatchery_variant": name}
                    for name in ("evolved", "matched_random", "fixed_charged", "flat")
                ],
                "verdict": {"status": "NO_CLAIM"},
                "offload": {"external_execution_started": False},
            }), encoding="utf-8")
            return subprocess.CompletedProcess(argv, 0)

        with mock.patch("clay_train_cli.subprocess.run", side_effect=fake_run):
            receipt = self.studio.opcode_hatch("smoke")
        argv = seen["argv"]
        self.assertIn("hatchery", argv)
        self.assertEqual(argv[argv.index("--population-size") + 1], "4")
        self.assertIn("--smoke", argv)
        self.assertEqual(receipt["verdict"]["status"], "NO_CLAIM")
        self.assertFalse(receipt["external_execution_started"])
        self.assertEqual(len(self.studio.state["opcode_graph"]["hatchery_runs"]), 1)

    def test_opcode_internalization_runs_real_module_with_three_controls(self) -> None:
        seen: dict[str, object] = {}

        def fake_run(argv, cwd, check):
            seen["argv"] = argv
            seen["cwd"] = cwd
            output = Path(argv[argv.index("--output") + 1])
            output.mkdir(parents=True, exist_ok=True)
            (output / "summary.json").write_text(
                json.dumps(
                    {
                        "protocol": {"status": "PASS", "validation_examples": 96},
                        "aggregates": {
                            "aligned": {"word_opcode_accuracy": {"mean": 0.4}}
                        },
                        "verdict": {"status": "EVIDENCE_POSITIVE"},
                    }
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(argv, 0)

        with mock.patch("clay_train_cli.subprocess.run", side_effect=fake_run):
            receipt = self.studio.opcode_internalize("smoke")
        argv = seen["argv"]
        self.assertIn("clay_math.representation_internalization", argv)
        self.assertEqual(argv[argv.index("--child-epochs") + 1], "3")
        self.assertEqual(receipt["protocol"]["status"], "PASS")
        self.assertFalse(receipt["external_execution_started"])
        self.assertEqual(len(self.studio.state["opcode_graph"]["internalization_runs"]), 1)

    def test_conversation_can_request_full_internalization(self) -> None:
        result = self.studio.interpret(
            "start full training so Clay can internalize the six representations"
        )
        self.assertTrue(result["internalization_run_requested"])
        self.assertEqual(result["internalization_profile"], "full")
        self.assertFalse(result["run_requested"])

    def test_completed_internalization_import_is_hash_deduplicated(self) -> None:
        summary = Path(self.temp.name) / "summary.json"
        summary.write_text(
            json.dumps(
                {
                    "schema": "clay.representation_internalization.experiment.v1",
                    "protocol": {"status": "PASS", "validation_examples": 96},
                    "config": {
                        "seeds": [1, 2, 3],
                        "parent_epochs": 32,
                        "child_epochs": 192,
                    },
                    "aggregates": {"aligned": {"word_opcode_accuracy": {"mean": 0.4}}},
                    "verdict": {"status": "EVIDENCE_POSITIVE"},
                }
            ),
            encoding="utf-8",
        )
        first = self.studio.opcode_internalize_import(str(summary))
        second = self.studio.opcode_internalize_import(str(summary))
        self.assertEqual(first["summary_sha256"], second["summary_sha256"])
        self.assertEqual(len(self.studio.state["opcode_graph"]["internalization_runs"]), 1)
        self.assertEqual(first["promotion"], "none; evidence receipt only")

    def test_conversation_routes_english_block_training(self) -> None:
        result = self.studio.interpret(
            "start full training so Clay learns English word blocks and Python functions"
        )
        self.assertTrue(result["english_block_run_requested"])
        self.assertEqual(result["english_block_profile"], "full")
        self.assertFalse(result["run_requested"])

    def test_completed_english_block_run_imports_once_without_promotion(self) -> None:
        summary = Path(self.temp.name) / "english-summary.json"
        summary.write_text(
            json.dumps(
                {
                    "schema": "clay_english_block_training_v1",
                    "protocol": {"ok": True, "heldout_pairs": 16},
                    "evaluator": {"ok": True, "oracle_exact": 1.0},
                    "seeds": [17, 29, 43],
                    "epochs": 72,
                    "parameters_per_model": 233152,
                    "aggregates": {
                        "aligned": {"python_execution_accuracy": {"mean": 0.63}}
                    },
                    "capability_claim_pass": True,
                    "aligned_architecture_claim_pass": False,
                }
            ),
            encoding="utf-8",
        )
        first = self.studio.english_blocks_import(str(summary))
        second = self.studio.english_blocks_import(str(summary))
        self.assertEqual(first["summary_sha256"], second["summary_sha256"])
        self.assertEqual(len(self.studio.state["english_blocks"]["runs"]), 1)
        self.assertEqual(first["promotion"], "none; evidence receipt only")
        self.assertFalse(first["aligned_architecture_claim_pass"])

    def test_english_block_train_uses_local_pytorch_module_and_three_seeds(self) -> None:
        seen: dict[str, object] = {}

        def fake_run(argv, cwd, check):
            seen["argv"] = argv
            seen["cwd"] = cwd
            output = Path(argv[argv.index("--output") + 1])
            output.mkdir(parents=True, exist_ok=True)
            (output / "summary.json").write_text(
                json.dumps(
                    {
                        "schema": "clay_english_block_training_v1",
                        "protocol": {"ok": True, "heldout_pairs": 16},
                        "evaluator": {"ok": True, "oracle_exact": 1.0},
                        "seeds": [20260805, 20260806, 20260807],
                        "epochs": 72,
                        "parameters_per_model": 233152,
                        "aggregates": {},
                        "capability_claim_pass": True,
                        "aligned_architecture_claim_pass": False,
                    }
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(argv, 0)

        with mock.patch("clay_train_cli.subprocess.run", side_effect=fake_run):
            receipt = self.studio.english_blocks_train("full")
        argv = seen["argv"]
        self.assertIn("clay_math.english_block_training", argv)
        self.assertEqual(argv[argv.index("--epochs") + 1], "72")
        self.assertEqual(len(receipt["seeds"]), 3)
        self.assertTrue(receipt["capability_claim_pass"])
        self.assertFalse(receipt["external_execution_started"])

    def test_conversation_routes_generative_word_meaning_training(self) -> None:
        result = self.studio.interpret(
            "start real training so Clay understands words and explains their meanings"
        )
        self.assertTrue(result["word_meaning_run_requested"])
        self.assertEqual(result["word_meaning_profile"], "full")
        self.assertFalse(result["internalization_run_requested"])
        self.assertFalse(result["english_block_run_requested"])
        self.assertFalse(result["run_requested"])

    def test_completed_word_meaning_run_imports_once_without_promotion(self) -> None:
        summary = Path(self.temp.name) / "word-meaning-summary.json"
        summary.write_text(
            json.dumps(
                {
                    "schema": "clay_word_explanation_training_v1",
                    "protocol": {"ok": True, "heldout_lemma_families": 24},
                    "evaluator": {"ok": True, "perfect_content_f1": 1.0},
                    "seeds": [17, 29, 43],
                    "epochs": 18,
                    "parameters_per_model": 1_250_000,
                    "aggregates": {
                        "cross_vector": {"closed_content_f1": {"mean": 0.22}}
                    },
                    "comparisons": {},
                    "capability_claim_pass": True,
                    "cross_vector_architecture_claim_pass": False,
                    "runs": [
                        {
                            "seed": 17,
                            "arm": "cross_vector",
                            "checkpoint": "candidate.pt",
                            "checkpoint_sha256": "a" * 64,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        first = self.studio.word_meanings_import(str(summary))
        second = self.studio.word_meanings_import(str(summary))
        self.assertEqual(first["summary_sha256"], second["summary_sha256"])
        self.assertEqual(len(self.studio.state["word_meanings"]["runs"]), 1)
        self.assertIn("not silently adopted", first["promotion"])
        self.assertFalse(first["cross_vector_architecture_claim_pass"])

    def test_word_meaning_train_uses_local_pytorch_module_and_three_seeds(self) -> None:
        seen: dict[str, object] = {}

        def fake_run(argv, cwd, check):
            seen["argv"] = argv
            seen["cwd"] = cwd
            output = Path(argv[argv.index("--output") + 1])
            output.mkdir(parents=True, exist_ok=True)
            (output / "summary.json").write_text(
                json.dumps(
                    {
                        "schema": "clay_word_explanation_training_v1",
                        "protocol": {"ok": True, "heldout_lemma_families": 24},
                        "evaluator": {"ok": True, "perfect_content_f1": 1.0},
                        "seeds": [20260805, 20260806, 20260807],
                        "epochs": 18,
                        "parameters_per_model": 1_250_000,
                        "aggregates": {},
                        "comparisons": {},
                        "capability_claim_pass": False,
                        "cross_vector_architecture_claim_pass": False,
                        "runs": [],
                    }
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(argv, 0)

        with mock.patch("clay_train_cli.subprocess.run", side_effect=fake_run):
            receipt = self.studio.word_meanings_train("full")
        argv = seen["argv"]
        self.assertIn("clay_math.word_explanation_training", argv)
        self.assertEqual(argv[argv.index("--max-synsets") + 1], "1600")
        self.assertEqual(argv[argv.index("--epochs") + 1], "18")
        self.assertEqual(len(receipt["seeds"]), 3)
        self.assertFalse(receipt["capability_claim_pass"])
        self.assertFalse(receipt["external_execution_started"])

    def test_conversation_routes_fourier_flashcard_training(self) -> None:
        result = self.studio.interpret(
            "start real training so Clay learns Fourier flash cards, draws symbols, and makes Python"
        )
        self.assertTrue(result["fourier_card_run_requested"])
        self.assertEqual(result["fourier_card_profile"], "full")
        self.assertFalse(result["english_block_run_requested"])
        self.assertFalse(result["run_requested"])

    def test_words_explain_strips_optional_pipe_separator(self) -> None:
        expected = {"word": "bank", "explanation": "test"}
        with mock.patch.object(
            self.studio,
            "word_meanings_explain",
            return_value=expected,
        ) as explain:
            result = json.loads(
                self.studio.handle_command(
                    "/words explain bank | The children sat by the river."
                )
            )
        self.assertEqual(result, expected)
        explain.assert_called_once_with(
            "bank", "The children sat by the river."
        )

    def test_completed_fourier_card_run_imports_once_without_promotion(self) -> None:
        summary = Path(self.temp.name) / "fourier-summary.json"
        summary.write_text(
            json.dumps(
                {
                    "schema": "clay_fourier_flashcard_training_v1",
                    "protocol": {"ok": True, "final_split": "symbol_sealed+program_sealed"},
                    "evaluator": {"ok": True, "all_36_programs_verify": True},
                    "seeds": [17, 29, 43],
                    "stage1_epochs": 52,
                    "stage2_epochs": 44,
                    "parameters_per_model": 117677,
                    "aggregates": {
                        "adaptive_fourier": {
                            "python_execution_accuracy": {"mean": 0.7083}
                        }
                    },
                    "comparisons": {"capability_vs_untrained": {}},
                    "capability_claim_pass": True,
                    "fourier_drawing_claim_pass": True,
                    "adaptive_scheduler_claim_pass": False,
                    "artifacts": {"m3_m5_table": "m3_m5_child_table.json"},
                }
            ),
            encoding="utf-8",
        )
        first = self.studio.fourier_cards_import(str(summary))
        second = self.studio.fourier_cards_import(str(summary))
        self.assertEqual(first["summary_sha256"], second["summary_sha256"])
        self.assertEqual(len(self.studio.state["fourier_cards"]["runs"]), 1)
        self.assertEqual(first["promotion"], "none; evidence receipt only")
        self.assertEqual(first["tool_roles"]["hatchery"], "candidate scheduling and equal-initialization controls")
        self.assertFalse(first["adaptive_scheduler_claim_pass"])

    def test_fourier_card_train_uses_local_pytorch_module_and_three_seeds(self) -> None:
        seen: dict[str, object] = {}

        def fake_run(argv, cwd, check):
            seen["argv"] = argv
            seen["cwd"] = cwd
            output = Path(argv[argv.index("--output") + 1])
            output.mkdir(parents=True, exist_ok=True)
            (output / "summary.json").write_text(
                json.dumps(
                    {
                        "schema": "clay_fourier_flashcard_training_v1",
                        "protocol": {"ok": True, "final_split": "symbol_sealed+program_sealed"},
                        "evaluator": {"ok": True, "all_36_programs_verify": True},
                        "seeds": [20260805, 20260806, 20260807],
                        "stage1_epochs": 52,
                        "stage2_epochs": 44,
                        "parameters_per_model": 117677,
                        "aggregates": {},
                        "comparisons": {},
                        "capability_claim_pass": True,
                        "fourier_drawing_claim_pass": True,
                        "adaptive_scheduler_claim_pass": False,
                        "artifacts": {},
                    }
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(argv, 0)

        with mock.patch("clay_train_cli.subprocess.run", side_effect=fake_run):
            receipt = self.studio.fourier_cards_train("full")
        argv = seen["argv"]
        self.assertIn("clay_math.fourier_flashcard_training", argv)
        self.assertEqual(argv[argv.index("--stage1-epochs") + 1], "52")
        self.assertEqual(argv[argv.index("--stage2-epochs") + 1], "44")
        self.assertEqual(len(argv[argv.index("--seeds") + 1].split(",")), 3)
        self.assertTrue(receipt["capability_claim_pass"])
        self.assertFalse(receipt["external_execution_started"])

    def test_kaggriculture_local_gate_cannot_claim_external_pass(self) -> None:
        candidate = Path(self.temp.name) / "candidate.py"
        candidate.write_text("def agent(obs): return {}\n", encoding="utf-8")

        def fake_run(argv, cwd, check, capture_output, text):
            output = Path(argv[argv.index("--output") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps({
                "candidate": candidate.name,
                "opponent": "main_profit.py",
                "seeds": list(range(22000, 22012)),
                "seats_per_seed": 2,
                "games": 24,
                "wins": 18,
                "losses": 6,
                "ties": 0,
                "mean_margin": 1000.0,
                "margin_se": 200.0,
                "direct_win_signal": 0.5,
                "direct_win_signal_se": 0.1,
                "passes_2se": True,
                "matches": [],
            }), encoding="utf-8")
            return subprocess.CompletedProcess(argv, 0, "", "")

        with mock.patch("clay_train_cli.subprocess.run", side_effect=fake_run):
            receipt = self.studio.kaggriculture_local_gate(str(candidate), 12, 22000)
        self.assertEqual(receipt["local_gate"], "PASS")
        self.assertEqual(receipt["external_gate"], "NOT_RUN")
        self.assertEqual(receipt["overall"], "LOCAL_PASS_EXTERNAL_UNTESTED")
        self.assertFalse(receipt["benchmark_data_used_for_training"])
        self.assertEqual(receipt["target"]["submission_id"], 55283743)
        self.assertNotIn("public_score", receipt["target"])

    def test_kaggriculture_status_requires_same_live_snapshot(self) -> None:
        status = self.studio.kaggriculture_benchmark_status()
        self.assertEqual(status["target"]["submission_id"], KAGGRICULTURE_TARGET["submission_id"])
        self.assertEqual(status["external_gate"]["status"], "NOT_REFRESHED")
        self.assertIn("one authenticated Kaggle CLI snapshot", status["external_gate"]["rule"])
        self.assertNotIn("625.5", json.dumps(status))
        self.assertTrue(status["local_surrogate"]["not_equivalent_to_public_score"])

    def test_kaggriculture_refresh_compares_candidate_and_target_from_one_response(self) -> None:
        payload = """ref,fileName,date,description,status,publicScore,privateScore
55283743,main_profit.py,2026-08-06 00:09:24,target,SubmissionStatus.COMPLETE,668.3,
60000000,main_goldroute_flat.py,2026-08-06 05:00:00,candidate,SubmissionStatus.COMPLETE,700.0,
"""
        completed = subprocess.CompletedProcess([], 0, payload, "")
        with mock.patch("clay_train_cli.subprocess.run", return_value=completed) as run:
            snapshot = self.studio.kaggriculture_refresh_scores(60000000)

        run.assert_called_once()
        self.assertEqual(snapshot["target"]["public_score"], 668.3)
        self.assertEqual(snapshot["candidate"]["public_score"], 700.0)
        self.assertTrue(snapshot["same_snapshot"])
        self.assertEqual(snapshot["external_gate"]["status"], "PASS")
        status = self.studio.kaggriculture_benchmark_status()
        self.assertEqual(status["target"]["public_score"], 668.3)
        self.assertEqual(status["external_gate"]["status"], "PASS")

    def test_kaggle_csv_parser_does_not_confuse_description_commas_for_columns(self) -> None:
        rows = parse_kaggle_submission_snapshot(
            'ref,fileName,date,description,status,publicScore,privateScore\n'
            '55283743,main_profit.py,2026-08-06,"contains, commas",SubmissionStatus.COMPLETE,677.2,\n'
        )
        self.assertEqual(rows[0]["submission_id"], 55283743)
        self.assertEqual(rows[0]["public_score"], 677.2)
        self.assertEqual(rows[0]["status"], "COMPLETE")


if __name__ == "__main__":
    unittest.main()
