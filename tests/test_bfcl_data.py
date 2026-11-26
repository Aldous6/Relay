"""Synthetic snapshot fixtures test provenance and isolation, never BFCL scores."""
from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from routerbench import bfcl_data as data

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from import_bfcl import main


def fixture_files():
    questions, references = [], []
    for index in range(200):
        identity = f"multi_turn_base_{index}"
        questions.append({"id": identity, "question": [[{"role": "user", "content": f"fixture_prompt_{index}"}]],
                          "involved_classes": ["SyntheticFixture"], "initial_config": {}})
        references.append({"id": identity, "ground_truth": [[f"fixture_private_reference_{index}"]]})
    encode = lambda values: b"".join((json.dumps(value) + "\n").encode() for value in values)
    return {"LICENSE": b"Synthetic fixture license\n", data.QUESTION_PATH: encode(questions),
            data.REFERENCE_PATH: encode(references)}


class BFCLDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "snapshot"
        self.files = fixture_files()
        self.inventory = tuple((path, data.blob_sha1(raw), len(raw)) for path, raw in self.files.items())
        self.patcher = patch.object(data, "PINNED_FILES", self.inventory)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.fetches = []

    def fetch(self, url):
        self.fetches.append(url)
        prefix = f"https://raw.githubusercontent.com/{data.REPOSITORY}/{data.COMMIT}/"
        self.assertTrue(url.startswith(prefix))
        path = url[len(prefix):]
        if path == data.REFERENCE_PATH:
            self.assertTrue((self.root / "selection.json").is_file())
        return self.files[path]

    def imported(self):
        return data.download_snapshot(self.root, fetch=self.fetch)

    def test_prepare_and_cli_default_have_no_network_or_writes(self):
        with patch.object(data, "_fetch_public") as fetch, redirect_stdout(StringIO()) as output:
            result = main(["--output", str(self.root)])
        fetch.assert_not_called()
        self.assertEqual(result["mode"], "prepare")
        self.assertEqual(result["network_calls"], 0)
        self.assertEqual(json.loads(output.getvalue()), result)
        self.assertFalse(self.root.exists())

    def test_download_preserves_exact_bytes_hashes_and_freezes_before_references(self):
        snapshot = self.imported()
        self.assertEqual(len(self.fetches), len(self.files))
        for item in snapshot["manifest"]["files"]:
            original = self.files[item["path"]]
            self.assertEqual((self.root / "upstream" / item["path"]).read_bytes(), original)
            self.assertEqual(item["git_blob_sha1"], data.blob_sha1(original))
            self.assertEqual(item["sha256"], hashlib.sha256(original).hexdigest())
        counts = {name: len(snapshot["selection"][name]) for name in data.SPLIT_COUNTS}
        self.assertEqual(counts, {"fit": 96, "calibration": 40, "development": 24, "evaluation": 40})
        self.assertEqual(snapshot["manifest"]["provider_calls"], 0)
        self.assertNotIn("fixture_prompt_", json.dumps(snapshot))
        self.assertNotIn("fixture_private_reference_", json.dumps(snapshot))

    def test_completed_snapshot_is_idempotent_and_does_not_download_again(self):
        first = self.imported()
        with patch.object(data, "_fetch_public") as fetch:
            second = data.download_snapshot(self.root)
        fetch.assert_not_called()
        self.assertEqual(first, second)

    def test_split_is_deterministic_disjoint_and_ignores_input_order(self):
        identities = [f"multi_turn_base_{index}" for index in range(200)]
        split = data.select_ids(identities)
        self.assertEqual(split, data.select_ids(identities[::-1]))
        self.assertNotEqual(split["development"], data.select_ids(identities, seed=7)["development"])
        merged = [identity for name in data.SPLIT_COUNTS for identity in split[name]]
        self.assertEqual(len(merged), len(set(merged)))
        self.assertEqual(set(merged), set(identities))
        with self.assertRaises(ValueError):
            data.select_ids([*identities[:-1], "multi_turn_miss_func_0"])
        with self.assertRaises(ValueError):
            data.select_ids(identities[:-1])

    def test_only_chosen_development_payloads_are_decoded(self):
        snapshot = self.imported()
        selected = snapshot["selection"]["development"][:2]
        actual_loads = json.loads
        decoded = []
        def traced(value, *args, **kwargs):
            text = value.decode() if isinstance(value, bytes) else value
            if "fixture_prompt_" in text or "fixture_private_reference_" in text:
                decoded.append(actual_loads(value))
            return actual_loads(value, *args, **kwargs)
        with patch.object(data.json, "loads", side_effect=traced):
            cases = data.load_selected_cases(self.root, selected, include_reference=True)
        self.assertEqual([case["id"] for case in cases], selected)
        self.assertEqual(len(decoded), 4)
        self.assertEqual({record["id"] for record in decoded}, set(selected))
        self.assertTrue(all(case["reference"] is not None for case in cases))
        access = json.loads((self.root / "access.jsonl").read_text())
        self.assertEqual(access["ids"], selected)
        self.assertTrue(access["references_decoded"])

    def test_default_access_does_not_decode_references_and_blocks_other_partitions(self):
        snapshot = self.imported()
        selected = snapshot["selection"]["development"][:1]
        with patch.object(data, "_load_records", wraps=data._load_records) as load:
            cases = data.load_selected_cases(self.root, selected)
        self.assertEqual(load.call_count, 1)
        self.assertIsNone(cases[0]["reference"])
        for part in ("fit", "calibration", "evaluation"):
            with self.subTest(part=part), patch.object(data, "_load_records") as load:
                with self.assertRaises(ValueError):
                    data.load_selected_cases(self.root, snapshot["selection"][part][:1], include_reference=True)
                load.assert_not_called()
            with self.assertRaises(ValueError):
                data.load_selected_cases(self.root, partition=part)

    def test_upstream_tamper_and_selection_tamper_fail_closed(self):
        self.imported()
        original = self.files["LICENSE"]
        target = self.root / "upstream" / "LICENSE"
        target.write_bytes(original + b"changed")
        with self.assertRaisesRegex(ValueError, "pinned Git blob"):
            data.read_snapshot(self.root)
        target.write_bytes(original)
        selection = self.root / "selection.json"
        selection.write_bytes(selection.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "selection hash"):
            data.load_selected_cases(self.root)

    def test_wrong_download_bytes_are_rejected_before_persisting(self):
        with self.assertRaisesRegex(ValueError, "pinned Git blob"):
            data.download_snapshot(self.root, fetch=lambda url: b"not upstream")
        self.assertFalse((self.root / "manifest.json").exists())
        self.assertFalse((self.root / "upstream" / "LICENSE").exists())

    def test_partial_import_can_resume_only_matching_existing_bytes(self):
        target = self.root / "upstream" / "LICENSE"
        target.parent.mkdir(parents=True)
        target.write_bytes(self.files["LICENSE"])
        self.imported()
        self.assertEqual(len(self.fetches), len(self.files) - 1)

    def test_raw_directory_is_never_an_import_destination(self):
        with self.assertRaisesRegex(ValueError, "outside data/raw"):
            data.prepare_snapshot(ROOT / "data/raw/not-allowed")

    def test_exposure_ledger_preserves_assignment_and_blocks_exposed_evaluation(self):
        snapshot = self.imported()
        selection_before = (self.root / "selection.json").read_bytes()
        ids = (snapshot["selection"]["fit"][:6] + snapshot["selection"]["calibration"][:4]
               + snapshot["selection"]["development"][:5] + snapshot["selection"]["evaluation"][:3])
        partitions = {part: sorted(set(ids) & set(snapshot["selection"][part])) for part in data.SPLIT_COUNTS}
        incident = self.root.parent / "incident.json"
        incident.write_text(json.dumps({"kind": "accidental_source_search_exposure", "commit": data.COMMIT,
            "conservative_affected_ids": ids, "partition_ids": partitions}), encoding="utf-8")
        ledger = data.register_exposure_record(self.root, incident)
        self.assertEqual(len(ledger["exposed_ids"]), 18)
        self.assertEqual(len(ledger["blocked_evaluation_ids"]), 3)
        self.assertFalse(ledger["repartitioned"])
        self.assertEqual((self.root / "selection.json").read_bytes(), selection_before)
        self.assertEqual(data.register_exposure_record(self.root, incident), ledger)
        self.assertIsNotNone(data.read_snapshot(self.root)["exposure_ledger_sha256"])
        with self.assertRaisesRegex(ValueError, "Previously exposed"):
            data.require_unexposed_evaluation_ids(self.root, partitions["evaluation"])
        untouched = snapshot["selection"]["evaluation"][3:4]
        self.assertFalse(data.require_unexposed_evaluation_ids(self.root, untouched)["independent_validation"])

    def test_missing_exposure_ledger_cannot_certify_holdout(self):
        snapshot = self.imported()
        with self.assertRaisesRegex(ValueError, "explicit exposure ledger"):
            data.require_unexposed_evaluation_ids(self.root, snapshot["selection"]["evaluation"][:1])


if __name__ == "__main__":
    unittest.main()
