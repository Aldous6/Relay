"""Frozen score lookup integrity; no checkpoint load, model call or outcomes."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("adaptive_scoring_test", ROOT / "tools/adaptive_scoring.py")
scoring = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scoring)


class CachedScoringTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "scores.json"
        self.text = "def answer():\n    return 42\n"
        self.records = [{"task_id": "a", "prompt_text": self.text},
                        {"task_id": "a", "prompt_text": self.text}]

    def cache(self, historical=False):
        reference = dict(checkpoint=scoring.CHECKPOINT, revision=scoring.REVISION,
                         weights_sha256=scoring.FILES["model.safetensors"], load_seconds=10.5)
        row = dict(task_id="a", input_tokens=600, truncated_tokens=88)
        digest = hashlib.sha256(self.text.encode()).hexdigest()
        if historical:
            reference.update(implementation="local_wrapper_published_routellm_bert_transfer", checkpoint_verified=True,
                             predictor_injected=False, tokenization=dict(max_length=512, truncation=True, padding=True, prefix=None),
                             supervision="published GPT-4/Mixtral preferences with external augmentation",
                             scoring="1 - sum(softmax(logits)[-2:]); higher prefers the strong action")
            row.update(probability=.6, routing_text_sha256=digest, retained_tokens=512,
                       input_characters=len(self.text), router_seconds=1.2)
            return dict(reference=reference, rows=[row])
        reference.update(implementation="local_published_routellm_bert_scoring", max_input_tokens=512,
                         success_probability=False, source_supervision="GPT-4/Mixtral preferences")
        row.update(score=.6, prompt_sha256=digest, routing_seconds=1.2)
        return dict(kind="frozen_historical_BERT_controls", at_utc="2026-11-18", suite_sha256="0" * 64,
                    reference=reference, thresholds=[.5, .58], truncation_policy="keep_baseline",
                    threshold_provenance="historical_public_v2_development_frozen_no_new_outcomes",
                    source_sha256={}, rows=[row])

    def load(self, value, records=None):
        self.path.write_text(json.dumps(value), encoding="utf-8")
        return scoring.cached_bert_scorer(self.path, self.records if records is None else records)

    def test_both_schemas_exact_text_lookup_and_original_compute_provenance(self):
        for historical in (False, True):
            with self.subTest(historical=historical):
                scorer = self.load(self.cache(historical))
                self.assertEqual(scorer(self.text), dict(score=.6, input_tokens=600, truncated_tokens=88))
                self.assertEqual(scorer.load_seconds, 0)
                self.assertTrue(scorer.cached)
                self.assertEqual(scorer.provenance["original_load_seconds"], 10.5)
                self.assertEqual(scorer.provenance["original_scoring_seconds"], 1.2)
                self.assertEqual(scorer.provenance["original_total_compute_seconds"], 11.7)
                self.assertEqual(scorer.provenance["load_seconds"], 10.5)
                self.assertEqual(scorer.provenance["implementation"], "local_published_routellm_bert_scoring")
                self.assertEqual(scorer.provenance["source_supervision"], "GPT-4/Mixtral preferences")
                self.assertEqual(scorer.provenance["max_input_tokens"], 512)
                self.assertEqual(scorer.provenance["original_reference"], self.cache(historical)["reference"])
                self.assertEqual(scorer.provenance["cache_sha256"], hashlib.sha256(self.path.read_bytes()).hexdigest())
                self.assertIsNone(scorer.provenance["router_cost_usd"])
                with self.assertRaisesRegex(ValueError, "exact prompt"):
                    scorer(self.text.strip())
                changed = scorer(self.text)
                changed["score"] = 0
                self.assertEqual(scorer(self.text)["score"], .6)
                changed = scorer.provenance
                changed["checkpoint"] = "changed"
                self.assertEqual(scorer.provenance["checkpoint"], scoring.CHECKPOINT)

    def test_full_tasks_use_actual_task_prompt(self):
        from relay_router.harness_episode import task_prompt
        task = dict(task_id="a", instruction="Implement", initial_source="def x(): pass\n", public_tests="assert x() == 1")
        value = self.cache()
        text = task_prompt(task)
        value["rows"][0]["prompt_sha256"] = hashlib.sha256(text.encode()).hexdigest()
        scorer = self.load(value, [task])
        self.assertEqual(scorer(text)["score"], .6)
        task["prompt_text"] = "different"
        with self.assertRaisesRegex(ValueError, "actual task"):
            self.load(value, [task])

    def test_pinned_checkpoint_metadata_cannot_change(self):
        for key in ("checkpoint", "revision", "weights_sha256"):
            value = self.cache()
            value["reference"][key] = "changed"
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "pinned BERT"):
                self.load(value)
        value = self.cache(True)
        value["reference"]["predictor_injected"] = True
        with self.assertRaisesRegex(ValueError, "unverified"):
            self.load(value)

    def test_missing_extra_duplicate_rows_and_wrong_task_hash_fail(self):
        mutations = [lambda value: value["rows"].clear(),
                     lambda value: value["rows"].append(deepcopy(value["rows"][0])),
                     lambda value: value["rows"][0].update(task_id="other"),
                     lambda value: value["rows"][0].update(prompt_sha256="0" * 64),
                     lambda value: value["rows"][0].update(accepted=True)]
        for change in mutations:
            value = self.cache()
            change(value)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.load(value)
        value = self.cache()
        records = self.records + [dict(task_id="b", prompt_text="second prompt")]
        value["rows"].append(deepcopy(value["rows"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.load(value, records)

    def test_score_truncation_and_timing_validation(self):
        for key, bad in (("score", True), ("score", -1), ("score", 1.001), ("score", float("nan")),
                         ("input_tokens", True), ("input_tokens", 0), ("truncated_tokens", -1),
                         ("truncated_tokens", 87), ("routing_seconds", -1), ("routing_seconds", float("inf"))):
            value = self.cache()
            value["rows"][0][key] = bad
            with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                self.load(value)
        value = self.cache()
        value["reference"]["load_seconds"] = -1
        with self.assertRaises(ValueError):
            self.load(value)
        value = self.cache()
        value["reference"]["load_seconds"] = 1e308
        value["rows"][0]["routing_seconds"] = 1e308
        with self.assertRaises(ValueError):
            self.load(value)
        value = self.cache()
        value["reference"]["accepted"] = True
        with self.assertRaisesRegex(ValueError, "Unexpected data"):
            self.load(value)
        value = self.cache(True)
        value["rows"][0]["retained_tokens"] = 511
        with self.assertRaises(ValueError):
            self.load(value)

    def test_same_task_or_same_prompt_cannot_have_conflicting_data(self):
        with self.assertRaisesRegex(ValueError, "different exact prompts"):
            self.load(self.cache(), [self.records[0], dict(task_id="a", prompt_text="changed")])
        value = self.cache()
        value["rows"].append(dict(value["rows"][0], task_id="b", score=.1))
        with self.assertRaisesRegex(ValueError, "inconsistent cached scores"):
            self.load(value, [self.records[0], dict(task_id="b", prompt_text=self.text)])

    def test_does_not_read_outcomes_and_rejects_ambiguous_json(self):
        record = dict(self.records[0], accepted=object(), cost_usd=object(), model=object())
        self.assertEqual(self.load(self.cache(), [record])(self.text)["score"], .6)
        self.path.write_text('{"reference":{},"reference":{},"rows":[]}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate cache JSON key"):
            scoring.cached_bert_scorer(self.path, self.records)

    def test_historical_cache_is_compatible_with_core_without_loading_ml(self):
        from relay_router.adaptive_policy import train_policy, score_prompt
        cache = self.cache(True)
        prototype = cache["rows"][0]
        cache["rows"] = []
        records = []
        for index in range(3):
            task_id, text = str(index), self.text + "# " + str(index)
            cache["rows"].append(dict(prototype, task_id=task_id, input_characters=len(text),
                                      routing_text_sha256=hashlib.sha256(text.encode()).hexdigest()))
            for model, cost in (("small", .5), ("large", 1.)):
                records.append(dict(task_id=task_id, group_id=task_id, prompt_text=text,
                                    model=model, accepted=True, cost_usd=cost, cost_basis="tariff_estimate"))
        scorer = self.load(cache, records)
        policy = train_policy(records, baseline_model="large", candidate_model="small",
                              representation="routellm-bert", scorer=scorer)
        self.assertEqual(policy["predictor"]["kind"], "constant_safe")
        self.assertEqual(score_prompt(policy, records[0]["prompt_text"], scorer=scorer), 1.)


if __name__ == "__main__":
    unittest.main()
