"""Test leakage boundaries required for claims across unknown task families."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("dynamic_evaluation", ROOT / "tools/prepare_dynamic_evaluation.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DynamicEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.data = {name: [{"id": "a", "prompt": name+" material", "reference": "17", "grade": "exact"}]
                     for name in MODULE.SPLITS}
        self.meta = {name: [{"id": "a", "task_family": name+" task", "source_group": name+" source"}]
                     for name in MODULE.SPLITS}

    def test_open_topics_do_not_need_fixed_sector_labels_and_gold_is_not_a_predictor(self):
        counts, known = MODULE.inspect_partitions(self.data, self.meta)
        self.assertEqual(counts["evaluation"]["cases"], 1)
        self.assertEqual(len(known), 2)
        changed = copy.deepcopy(self.data)
        changed["evaluation"][0]["reference"] = "different answer"
        self.assertEqual(MODULE.inspect_partitions(changed, self.meta), (counts, known))

    def test_new_instances_of_known_family_do_not_qualify_as_unknown_family_test(self):
        self.meta["evaluation"][0]["task_family"] = self.meta["fit"][0]["task_family"]
        with self.assertRaisesRegex(ValueError, "familias"):
            MODULE.inspect_partitions(self.data, self.meta)

    def test_source_document_cannot_cross_splits_even_with_different_prompts(self):
        self.meta["calibration"][0]["source_group"] = self.meta["fit"][0]["source_group"]
        with self.assertRaisesRegex(ValueError, "fuente"):
            MODULE.inspect_partitions(self.data, self.meta)

    def test_normalized_duplicates_and_misaligned_annotations_are_rejected(self):
        self.data["evaluation"][0]["prompt"] = " FIT   MATERIAL "
        with self.assertRaisesRegex(ValueError, "duplicados"):
            MODULE.inspect_partitions(self.data, self.meta)
        self.data["evaluation"][0]["prompt"] = "new request"
        self.meta["evaluation"][0]["id"] = "wrong"
        with self.assertRaisesRegex(ValueError, "anotaciones"):
            MODULE.inspect_partitions(self.data, self.meta)

    def test_cli_freezes_artifacts_and_refuses_changed_policy_on_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            partitions = {}
            for split in MODULE.SPLITS:
                (root / (split+".jsonl")).write_text(json.dumps(self.data[split][0])+"\n", encoding="utf-8")
                (root / (split+"_meta.jsonl")).write_text(json.dumps(self.meta[split][0])+"\n", encoding="utf-8")
                partitions[split] = {"cases": split+".jsonl", "metadata": split+"_meta.jsonl"}
            artifact = root / "frozen.bin"
            artifact.write_bytes(b"model version one")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"partitions": partitions, "policy_manifest": {"paid_api_calls": False},
                                           "artifacts": ["frozen.bin"]}), encoding="utf-8")
            output = root / "evaluation"
            with patch.object(MODULE, "ROOT", root):
                MODULE.main(["--manifest", str(manifest), "--output", str(output)])
                frozen = (output / "freeze.json").read_bytes()
                MODULE.main(["--manifest", str(manifest), "--output", str(output)])
                artifact.write_bytes(b"changed model")
                with self.assertRaisesRegex(ValueError, "congelada"):
                    MODULE.main(["--manifest", str(manifest), "--output", str(output)])
            self.assertEqual((output / "freeze.json").read_bytes(), frozen)
            self.assertFalse((output / "responses.json").exists())


if __name__ == "__main__":
    unittest.main()
