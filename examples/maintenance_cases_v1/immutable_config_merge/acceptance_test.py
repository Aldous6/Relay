import copy
import unittest
from module import DELETE, merge_config


class AcceptanceTests(unittest.TestCase):
    def test_null_deletion_and_type_changes(self):
        self.assertEqual(merge_config({"a": 1, "b": {"x": 1}},
                                     {"a": None, "b": [], "missing": DELETE}), {"a": None, "b": []})
        self.assertEqual(merge_config({"x": 2}, {"x": {"gone": DELETE, "keep": 1}}), {"x": {"keep": 1}})
        self.assertEqual(merge_config({}, {"a": {"b": {"z": DELETE}, "c": False}}), {"a": {"b": {}, "c": False}})

    def test_deep_detachment_in_both_directions(self):
        base = {"retained": {"list": [{"v": 1}]}, "merged": {"old": [1]}}
        patch = {"merged": {"new": [{"k": [2]}]}, "replacement": [{"n": [3]}]}
        before_base, before_patch = copy.deepcopy(base), copy.deepcopy(patch)
        result = merge_config(base, patch)
        result["retained"]["list"][0]["v"] = 99
        result["merged"]["old"].append(4)
        result["merged"]["new"][0]["k"].append(5)
        result["replacement"][0]["n"].append(6)
        self.assertEqual(base, before_base)
        self.assertEqual(patch, before_patch)
        frozen_result = copy.deepcopy(result)
        base["retained"]["list"].append(0)
        patch["replacement"][0]["n"].append(7)
        self.assertEqual(result, frozen_result)

    def test_no_mutation_when_deleting_nested_values(self):
        base = {"n": {"x": [1], "y": 2}}
        patch = {"n": {"x": DELETE, "absent": DELETE}}
        self.assertEqual(merge_config(base, patch), {"n": {"y": 2}})
        self.assertEqual(base, {"n": {"x": [1], "y": 2}})
        self.assertIs(patch["n"]["x"], DELETE)

    def test_return_is_plain_dict_and_empty_copy(self):
        original = {"x": [{"a": []}]}
        result = merge_config(original, {})
        self.assertIs(type(result), dict)
        result["x"][0]["a"].append(1)
        self.assertEqual(original, {"x": [{"a": []}]})
