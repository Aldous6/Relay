"""Author local diagnostic fixtures; never contacts providers."""
from pathlib import Path
import json
import textwrap

ROOT = Path(__file__).resolve().parent
CASES = []


def case(name, task, source, reference, public, acceptance):
    folder = ROOT / name
    folder.mkdir(parents=True, exist_ok=True)
    for filename, contents in (("module.py", source), ("test_public.py", public),
                               ("acceptance_test.py", acceptance)):
        (folder / filename).write_text(textwrap.dedent(contents).lstrip(), encoding="utf-8")
    refs = ROOT / "reference_solutions"
    refs.mkdir(exist_ok=True)
    (refs / f"{name}.py").write_text(textwrap.dedent(reference).lstrip(), encoding="utf-8")
    config = {
        "schema_version": 1, "repository": ".", "source_files": ["module.py"],
        "test_files": ["test_public.py"],
        "models": {"inspection": "groq/openai/gpt-oss-20b", "editing": "groq/openai/gpt-oss-120b"},
        "limits": {"max_repairs": 1, "max_output_tokens": 2048, "max_requests": 3,
                   "request_budget": 8000, "minimum_interval_seconds": 60,
                   "request_timeout_seconds": 90, "test_timeout_seconds": 30},
        "spending": {"account_is_free_tier": True, "max_provider_spend_usd": 0},
    }
    (folder / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    CASES.append({"task_id": name, "group_id": f"authored_v1_{name}",
                  "partition": "development", "config": f"../examples/maintenance_cases_v1/{name}/config.json",
                  "task": task, "acceptance_files": ["acceptance_test.py"]})


case("interval_difference", """Repair subtract_intervals(intervals, blocked) in module.py. Both arguments are iterables of pairs of integer endpoints representing half-open intervals [start,end). Return the union of intervals minus the union of blocked, as a sorted list of maximal nonempty (start,end) tuples: merge touching as well as overlapping output intervals. Inputs may be unsorted, repeated, overlapping, nested, empty, negative, or one-shot iterators. Ignore zero-length intervals. Reject reversed endpoints and endpoints whose type is not exactly int (including bool) with ValueError, even if the other collection is empty. Do not mutate either input. Preserve the function interface and use only the standard library.""", r'''
"""Utilities for subtracting unavailable spans from reservations."""


def _ordered(spans):
    """Prepare spans for the sweep."""
    ordered = []
    for start, end in spans:
        if start > end:
            raise ValueError("reversed span")
        ordered.append((start, end))
    return sorted(ordered)


def _cut_one(span, cut):
    start, end = span
    left, right = cut
    if right <= start or left >= end:
        return [span]
    result = []
    if start < left:
        result.append((start, left))
    if right < end:
        result.append((right, end))
    return result


def subtract_intervals(intervals, blocked):
    """Return available pieces of every input span."""
    result = []
    for span in _ordered(intervals):
        pieces = [span]
        for cut in blocked:
            next_pieces = []
            for piece in pieces:
                next_pieces.extend(_cut_one(piece, cut))
            pieces = next_pieces
        result.extend(pieces)
    return result
''', r'''
def _union(spans):
    valid = []
    for start, end in spans:
        if type(start) is not int or type(end) is not int or start > end:
            raise ValueError("invalid interval")
        if start < end:
            valid.append((start, end))
    result = []
    for start, end in sorted(valid):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def subtract_intervals(intervals, blocked):
    source, cuts = _union(intervals), _union(blocked)
    result = []
    for start, end in source:
        cursor = start
        for left, right in cuts:
            if right <= cursor:
                continue
            if left >= end:
                break
            if left > cursor:
                result.append((cursor, left))
            cursor = max(cursor, right)
            if cursor >= end:
                break
        if cursor < end:
            result.append((cursor, end))
    return result
''', r'''
import unittest
from module import subtract_intervals


class PublicTests(unittest.TestCase):
    def test_cut(self):
        self.assertEqual(subtract_intervals([(1, 9)], [(3, 5)]), [(1, 3), (5, 9)])

    def test_union(self):
        self.assertEqual(subtract_intervals([(3, 8), (1, 4)], []), [(1, 8)])

    def test_empty(self):
        self.assertEqual(subtract_intervals([], []), [])
''', r'''
import copy
import random
import unittest
from module import subtract_intervals


class AcceptanceTests(unittest.TestCase):
    def test_touching_empty_nested(self):
        self.assertEqual(subtract_intervals([(1, 2), (2, 8), (4, 4)], [(2, 3), (3, 5)]), [(1, 2), (5, 8)])
        self.assertEqual(subtract_intervals([(-9, 9), (0, 1)], [(-20, 20)]), [])
        self.assertEqual(subtract_intervals([(1, 1)], []), [])

    def test_iterators_and_immutability(self):
        source, cuts = [[0, 5], [9, 12]], [[2, 3], [10, 11]]
        original = copy.deepcopy((source, cuts))
        self.assertEqual(subtract_intervals(iter(source), iter(cuts)), [(0, 2), (3, 5), (9, 10), (11, 12)])
        self.assertEqual((source, cuts), original)

    def test_validation_not_short_circuited(self):
        for invalid in [[(3, 1)], [(True, 2)], [(1, 2.0)]]:
            for args in [(invalid, []), ([], invalid)]:
                with self.subTest(args=args), self.assertRaises(ValueError):
                    subtract_intervals(*args)

    def test_integer_coverage_and_canonical_result(self):
        rng = random.Random(173)
        for _ in range(60):
            source = [tuple(sorted((rng.randrange(-8, 9), rng.randrange(-8, 9)))) for _ in range(6)]
            cuts = [tuple(sorted((rng.randrange(-8, 9), rng.randrange(-8, 9)))) for _ in range(4)]
            expected = {x for a, b in source for x in range(a, b)} - {x for a, b in cuts for x in range(a, b)}
            result = subtract_intervals(source, cuts)
            self.assertEqual({x for a, b in result for x in range(a, b)}, expected)
            self.assertTrue(all(a < b for a, b in result))
            self.assertTrue(all(result[i][1] < result[i + 1][0] for i in range(len(result) - 1)))
''')

case("dependency_order", """Repair dependency_order(graph) in module.py. graph maps string node names to iterable string prerequisites. Include keys and prerequisites absent as keys. Return a topological ordering, choosing the lexicographically smallest currently ready node at every step (Python string ordering). Duplicate prerequisite edges count once. Reject any cycle, including a self-cycle or a cycle disconnected from ready nodes, with ValueError. Empty graph returns []. Inputs are valid strings and finite iterables, which can be generators; do not mutate graph or its containers. Preserve the interface and use only the standard library.""", r'''
"""Plan prerequisite-first work for a dependency graph."""


def _visit(node, graph, visited, result):
    if node in visited:
        return
    visited.add(node)
    for dependency in graph.get(node, ()):
        _visit(dependency, graph, visited, result)
    result.append(node)


def _all_nodes(graph):
    nodes = set(graph)
    for dependencies in graph.values():
        for node in dependencies:
            nodes.add(node)
    return nodes


def dependency_order(graph):
    """Visit dependencies before the task that needs them.

    Node names provide deterministic ordering for reproducible plans.
    The traversal supports nodes that only appear as prerequisites.
    Each node is appended once so shared prerequisites are not repeated.
    """
    result = []
    visited = set()
    nodes = _all_nodes(graph)
    for node in sorted(nodes):
        _visit(node, graph, visited, result)
    return result
''', r'''
import heapq


def dependency_order(graph):
    dependencies = {node: set(edges) for node, edges in graph.items()}
    nodes = set(dependencies)
    for edges in dependencies.values():
        nodes.update(edges)
    following = {node: [] for node in nodes}
    degree = {node: len(dependencies.get(node, ())) for node in nodes}
    for node, edges in dependencies.items():
        for edge in edges:
            following[edge].append(node)
    ready = [node for node in nodes if degree[node] == 0]
    heapq.heapify(ready)
    result = []
    while ready:
        node = heapq.heappop(ready)
        result.append(node)
        for target in following[node]:
            degree[target] -= 1
            if degree[target] == 0:
                heapq.heappush(ready, target)
    if len(result) != len(nodes):
        raise ValueError("cyclic graph")
    return result
''', r'''
import unittest
from module import dependency_order


class PublicTests(unittest.TestCase):
    def test_chain(self):
        self.assertEqual(dependency_order({"c": ["b"], "b": ["a"]}), ["a", "b", "c"])

    def test_ready_tie(self):
        self.assertEqual(dependency_order({"a": ["z"], "b": []}), ["b", "z", "a"])

    def test_empty(self):
        self.assertEqual(dependency_order({}), [])
''', r'''
import copy
import itertools
import unittest
from module import dependency_order


class AcceptanceTests(unittest.TestCase):
    def test_duplicate_edges_missing_nodes_generators(self):
        self.assertEqual(dependency_order({"d": iter(["b", "a", "b"]), "c": iter(["b"])}), ["a", "b", "c", "d"])

    def test_cycles(self):
        for graph in [{"x": ["x"]}, {"a": ["b"], "b": ["a"]}, {"z": [], "x": ["y"], "y": ["x"]}]:
            with self.subTest(graph=graph), self.assertRaises(ValueError):
                dependency_order(graph)

    def test_no_mutation_and_dynamic_ready(self):
        graph = {"z": [], "b": ["c"], "a": ["c"], "c": []}
        before = copy.deepcopy(graph)
        self.assertEqual(dependency_order(graph), ["c", "a", "b", "z"])
        self.assertEqual(graph, before)

    def test_small_dags_against_all_valid_orders(self):
        nodes = ["a", "b", "c", "d"]
        edges = [(nodes[i], nodes[j]) for i in range(4) for j in range(i)]
        for mask in range(1 << len(edges)):
            graph = {node: [] for node in nodes}
            for i, (node, dependency) in enumerate(edges):
                if mask & (1 << i):
                    graph[node].append(dependency)
            orders = [p for p in itertools.permutations(nodes)
                      if all(p.index(d) < p.index(n) for n, ds in graph.items() for d in ds)]
            self.assertEqual(dependency_order(graph), list(min(orders)))
''')

case("quoted_record_stream", """Repair parse_records(chunks, delimiter=',') in module.py. Consume a finite iterable of string chunks once and return a list of records, each a list of field strings. Chunk boundaries have no meaning, including boundaries between doubled quotes or CR and LF. Delimiter must be one character other than double quote, CR or LF; otherwise raise ValueError. Fields may be quoted only when their first character is double quote. Within quotes preserve all characters including CR/LF, interpret doubled double quotes as one quote, and allow delimiters. After a closing quote only a delimiter, record terminator or end is valid. Outside quotes records end with LF or CRLF; bare CR is invalid. Quotes inside an unquoted field, unclosed quotes and text after a closing quote are invalid: raise ValueError. Do not strip spaces. Empty input yields []; a blank record yields ['']; trailing record terminator adds no extra record; a trailing delimiter adds an empty field. Use only the standard library. No requirement to limit buffering.""", r'''
"""Read delimiter-separated records from text chunks."""


def _fields(line, delimiter):
    fields = []
    current = []
    quoted = False
    for character in line:
        if character == '"':
            quoted = not quoted
        elif character == delimiter and not quoted:
            fields.append("".join(current))
            current = []
        else:
            current.append(character)
    if quoted:
        raise ValueError("unclosed quote")
    fields.append("".join(current))
    return fields


def parse_records(chunks, delimiter=","):
    """Buffer partial lines and emit completed records."""
    result = []
    pending = ""
    for chunk in chunks:
        pending += chunk
        while "\n" in pending:
            line, pending = pending.split("\n", 1)
            result.append(_fields(line.rstrip("\r"), delimiter))
    if pending:
        result.append(_fields(pending, delimiter))
    return result
''', r'''
def parse_records(chunks, delimiter=","):
    if not isinstance(delimiter, str) or len(delimiter) != 1 or delimiter in '\"\r\n':
        raise ValueError("invalid delimiter")
    text = "".join(chunks)
    records, record, field = [], [], []
    state = "start"
    active = False
    i = 0
    while i < len(text):
        ch = text[i]
        if state == "quoted":
            if ch == '"':
                if i + 1 < len(text) and text[i + 1] == '"':
                    field.append('"')
                    i += 2
                    continue
                state = "closed"
            else:
                field.append(ch)
        elif ch == delimiter:
            record.append("".join(field))
            field, state = [], "start"
            active = True
        elif ch in "\r\n":
            if ch == "\r":
                if i + 1 >= len(text) or text[i + 1] != "\n":
                    raise ValueError("bare CR")
                i += 1
            record.append("".join(field))
            records.append(record)
            record, field, state, active = [], [], "start", False
        elif state == "closed":
            raise ValueError("text after quote")
        elif ch == '"':
            if state != "start":
                raise ValueError("quote in plain field")
            state, active = "quoted", True
        else:
            field.append(ch)
            state, active = "plain", True
        i += 1
    if state == "quoted":
        raise ValueError("unclosed quote")
    if active or record or field:
        records.append(record + ["".join(field)])
    return records
''', r'''
import unittest
from module import parse_records


class PublicTests(unittest.TestCase):
    def test_plain_chunks(self):
        self.assertEqual(parse_records(["a,b\n", "c,", "d"]), [["a", "b"], ["c", "d"]])

    def test_escaped_quote(self):
        self.assertEqual(parse_records(['"say ""hello""",x']), [['say "hello"', 'x']])

    def test_empty(self):
        self.assertEqual(parse_records([]), [])
''', r'''
import unittest
from module import parse_records


class AcceptanceTests(unittest.TestCase):
    def test_every_boundary(self):
        source = '"a,b","line1\r\nline2",\r\n"say ""hi""", x\n'
        expected = [["a,b", "line1\r\nline2", ""], ['say "hi"', " x"]]
        for split in range(len(source) + 1):
            self.assertEqual(parse_records(iter([source[:split], "", source[split:]])), expected)
        self.assertEqual(parse_records(iter(source)), expected)

    def test_empty_fields_terminators(self):
        for source, expected in [("", []), ("\n", [[""]]), ("\r\n\n", [[""], [""]]),
                                 (",", [["", ""]]), ('""', [[""]]), ("a,\n", [["a", ""]]),
                                 ("a\n", [["a"]]), ('"a\rb"', [["a\rb"]])]:
            with self.subTest(source=source):
                self.assertEqual(parse_records([source]), expected)

    def test_strict_syntax(self):
        for source in ['a"b', '"a"x', '"a" ', '"open', "x\r", "x\ry", '"a"\r']:
            with self.subTest(source=source), self.assertRaises(ValueError):
                parse_records(iter(source))

    def test_custom_and_invalid_delimiter(self):
        self.assertEqual(parse_records(['"x|y"| z|'], "|"), [["x|y", " z", ""]])
        for delimiter in ["", "xx", '"', "\r", "\n", None]:
            with self.subTest(delimiter=delimiter), self.assertRaises(ValueError):
                parse_records([], delimiter)
''')

case("rolling_window", """Repair WindowStats(width) in module.py. width must have type exactly int and be positive; otherwise ValueError. add(timestamp,value) inserts an event and returns (count,total); snapshot(now) returns (count,total) without inserting. Timestamps, now and values must have type exactly int (bool is invalid); invalid values raise ValueError. Calls must have nondecreasing time across both methods, with ties allowed; reject a time older than any prior successful call with ValueError. At time t count only events with t-width < timestamp <= t, so expire events exactly on the left boundary. Values and times may be negative. Empty state returns (0,0). Failed calls leave all state unchanged, including latest time and pending events. Keep separate instances independent and use only the standard library.""", r'''
"""Small event counter for integer time windows."""


class WindowStats:
    def __init__(self, width):
        if width <= 0:
            raise ValueError("positive width required")
        self.width = width
        self.events = []
        self.last = None

    def _expire(self, now):
        cutoff = now - self.width
        self.events = [(stamp, value) for stamp, value in self.events if stamp >= cutoff]

    def _totals(self):
        count = len(self.events)
        total = sum(value for _, value in self.events)
        return count, total

    def add(self, timestamp, value):
        """Append an event and return current totals."""
        self._expire(timestamp)
        self.events.append((timestamp, value))
        if self.last is not None and timestamp < self.last:
            raise ValueError("time moved backwards")
        self.last = timestamp
        return self._totals()

    def snapshot(self, now):
        """Inspect the most recently accumulated events."""
        return self._totals()
''', r'''
from collections import deque


class WindowStats:
    def __init__(self, width):
        if type(width) is not int or width <= 0:
            raise ValueError("invalid width")
        self.width = width
        self.events = deque()
        self.last = None
        self.total = 0

    def _advance(self, now):
        if type(now) is not int or (self.last is not None and now < self.last):
            raise ValueError("invalid time")
        while self.events and self.events[0][0] <= now - self.width:
            self.total -= self.events.popleft()[1]
        self.last = now

    def add(self, timestamp, value):
        if type(value) is not int:
            raise ValueError("invalid value")
        self._advance(timestamp)
        self.events.append((timestamp, value))
        self.total += value
        return len(self.events), self.total

    def snapshot(self, now):
        self._advance(now)
        return len(self.events), self.total
''', r'''
import unittest
from module import WindowStats


class PublicTests(unittest.TestCase):
    def test_accumulation(self):
        stats = WindowStats(5)
        self.assertEqual(stats.add(1, 3), (1, 3))
        self.assertEqual(stats.add(2, 4), (2, 7))

    def test_boundary(self):
        stats = WindowStats(5)
        stats.add(1, 3)
        self.assertEqual(stats.add(6, 4), (1, 4))

    def test_empty(self):
        self.assertEqual(WindowStats(2).snapshot(0), (0, 0))
''', r'''
import random
import unittest
from module import WindowStats


class AcceptanceTests(unittest.TestCase):
    def test_snapshot_expiry_and_time_contract(self):
        stats = WindowStats(3)
        stats.add(-2, -7)
        stats.add(-2, 4)
        self.assertEqual(stats.snapshot(0), (2, -3))
        self.assertEqual(stats.snapshot(1), (0, 0))
        with self.assertRaises(ValueError):
            stats.add(0, 10)
        self.assertEqual(stats.add(1, 2), (1, 2))

    def test_invalid_calls_atomic(self):
        for width in [True, 1.5, 0, -1, "3"]:
            with self.subTest(width=width), self.assertRaises(ValueError):
                WindowStats(width)
        stats = WindowStats(20)
        stats.add(10, 5)
        invalid = [lambda: stats.add(9, 100), lambda: stats.add(100, 2.5),
                   lambda: stats.add(True, 1), lambda: stats.snapshot(10.5),
                   lambda: stats.snapshot(9), lambda: stats.add(11, False)]
        for operation in invalid:
            with self.assertRaises(ValueError):
                operation()
            self.assertEqual(stats.snapshot(10), (1, 5))

    def test_repeated_times_and_independent_instances(self):
        left, right = WindowStats(2), WindowStats(2)
        self.assertEqual(left.add(2, 0), (1, 0))
        self.assertEqual(left.add(2, -1), (2, -1))
        self.assertEqual(right.snapshot(2), (0, 0))
        self.assertEqual(left.snapshot(4), (0, 0))

    def test_deterministic_stream(self):
        rng = random.Random(47)
        stats, events, now = WindowStats(7), [], -100
        for i in range(100):
            now += rng.randrange(4)
            if i % 3:
                value = rng.randrange(-12, 13)
                events.append((now, value))
                actual = stats.add(now, value)
            else:
                actual = stats.snapshot(now)
            live = [value for stamp, value in events if now - 7 < stamp <= now]
            self.assertEqual(actual, (len(live), sum(live)))
''')

case("immutable_config_merge", """Repair merge_config(base, patch) and preserve exported DELETE in module.py. Inputs are plain dictionaries with string keys and JSON-like values (dict, list, string, number, bool, None); patch may additionally use the singleton DELETE as a dictionary value. Merge recursively when a patch value is a dict; if the base value is not a dict, start that subtree from {}. Other patch values replace the base, including lists and None. DELETE removes a key if present and otherwise does nothing. DELETE is allowed only as a dictionary value, never inside a list (at any depth); inputs otherwise follow the stated domain. Do not mutate either input. Every dict/list anywhere in the result must be detached from both inputs, including retained base values and replaced list contents; later mutations of result or either input must not affect the others. Return a plain dict; preserve the function interface and use only the standard library.""", r'''
"""Layer application overrides onto a base configuration."""

DELETE = object()


def _copy_container(value):
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, list):
        return list(value)
    return value


def _merge_values(previous, incoming):
    if isinstance(previous, dict) and isinstance(incoming, dict):
        return merge_config(previous, incoming)
    return _copy_container(incoming)


def merge_config(base, patch):
    """Apply explicitly supplied overrides to a copied top-level dict.

    A missing override retains the existing value.
    Dictionaries combine their individual keys while lists are replaced.
    This keeps configuration assembly convenient for the caller.
    """
    result = dict(base)
    for key, value in patch.items():
        if value is None:
            result.pop(key, None)
        elif key in result:
            result[key] = _merge_values(result[key], value)
        else:
            result[key] = _copy_container(value)
    return result
''', r'''
from copy import deepcopy

DELETE = object()


def merge_config(base, patch):
    result = deepcopy(base)
    for key, value in patch.items():
        if value is DELETE:
            result.pop(key, None)
        elif isinstance(value, dict):
            previous = result.get(key)
            result[key] = merge_config(previous if isinstance(previous, dict) else {}, value)
        else:
            result[key] = deepcopy(value)
    return result
''', r'''
import unittest
from module import DELETE, merge_config


class PublicTests(unittest.TestCase):
    def test_nested_merge(self):
        self.assertEqual(merge_config({"db": {"host": "a", "port": 1}}, {"db": {"port": 2}}),
                         {"db": {"host": "a", "port": 2}})

    def test_deletion(self):
        self.assertEqual(merge_config({"obsolete": 1, "keep": 2}, {"obsolete": DELETE}), {"keep": 2})

    def test_list_replacement(self):
        self.assertEqual(merge_config({"tags": [1, 2]}, {"tags": [3]}), {"tags": [3]})
''', r'''
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
''')

case("generation_cache", """Repair GenerationCache in module.py. begin(key) returns a positive integer token unique across all begins on that instance, including after invalidate/clear. A token belongs to its key; a later begin for that key revokes the earlier token. begin does not remove an already cached value. commit(key, token, value) returns True and stores value only when token is the latest unconsumed begin for that key; that token is consumed by the successful commit. All stale, already consumed, unknown, or wrong-key tokens return False without changing any state. get(key, default=None) returns the cached value (including None) or default if absent. invalidate(key) removes its cached value and revokes any pending token, without affecting other keys. clear() removes all cached values and revokes all pending tokens, including keys never committed. Keys are strings and tokens passed to commit are integers. Instances are independent. Values may be any objects and need not be copied. Preserve these interfaces and use only the standard library.""", r'''
"""Cache guarded by generations of asynchronous refresh work."""


class GenerationCache:
    def __init__(self):
        self.values = {}
        self.generations = {}

    def begin(self, key):
        """Start a new refresh for a key."""
        generation = self.generations.get(key, 0) + 1
        self.generations[key] = generation
        return generation

    def commit(self, key, token, value):
        """Publish a completed refresh if its generation is current."""
        if self.generations.get(key) != token:
            return False
        self.values[key] = value
        return True

    def get(self, key, default=None):
        return self.values.get(key, default)

    def invalidate(self, key):
        """Remove a cached key and its generation."""
        self.values.pop(key, None)
        self.generations.pop(key, None)

    def clear(self):
        """Invalidate all currently stored values."""
        for key in list(self.values):
            self.invalidate(key)
''', r'''
class GenerationCache:
    def __init__(self):
        self.values = {}
        self.pending = {}
        self.next_token = 0

    def begin(self, key):
        self.next_token += 1
        self.pending[key] = self.next_token
        return self.next_token

    def commit(self, key, token, value):
        if key not in self.pending or self.pending[key] != token:
            return False
        del self.pending[key]
        self.values[key] = value
        return True

    def get(self, key, default=None):
        return self.values.get(key, default)

    def invalidate(self, key):
        self.values.pop(key, None)
        self.pending.pop(key, None)

    def clear(self):
        self.values.clear()
        self.pending.clear()
''', r'''
import unittest
from module import GenerationCache


class PublicTests(unittest.TestCase):
    def test_publish(self):
        cache = GenerationCache()
        self.assertTrue(cache.commit("a", cache.begin("a"), 7))
        self.assertEqual(cache.get("a"), 7)

    def test_token_consumed(self):
        cache = GenerationCache()
        token = cache.begin("a")
        self.assertTrue(cache.commit("a", token, 1))
        self.assertFalse(cache.commit("a", token, 2))
        self.assertEqual(cache.get("a"), 1)

    def test_default(self):
        self.assertEqual(GenerationCache().get("missing", 42), 42)
''', r'''
import unittest
from module import GenerationCache


class AcceptanceTests(unittest.TestCase):
    def test_out_of_order_and_old_value_retained(self):
        cache = GenerationCache()
        cache.commit("x", cache.begin("x"), "old")
        first, second = cache.begin("x"), cache.begin("x")
        self.assertEqual(cache.get("x"), "old")
        self.assertFalse(cache.commit("x", first, "stale"))
        self.assertTrue(cache.commit("x", second, "fresh"))
        self.assertEqual(cache.get("x"), "fresh")

    def test_no_aba_after_invalidate_or_clear(self):
        cache = GenerationCache()
        old = cache.begin("x")
        cache.invalidate("x")
        new = cache.begin("x")
        self.assertNotEqual(old, new)
        self.assertFalse(cache.commit("x", old, 1))
        cache.clear()
        last = cache.begin("x")
        self.assertNotIn(last, (old, new))
        self.assertFalse(cache.commit("x", new, 2))
        self.assertTrue(cache.commit("x", last, 3))

    def test_clear_pending_only_and_cross_key_tokens(self):
        cache = GenerationCache()
        a, b = cache.begin("a"), cache.begin("b")
        self.assertIs(type(a), int)
        self.assertGreater(a, 0)
        self.assertNotEqual(a, b)
        self.assertFalse(cache.commit("a", b, 7))
        self.assertTrue(cache.commit("a", a, None))
        self.assertIsNone(cache.get("a", "missing"))
        cache.clear()
        self.assertFalse(cache.commit("b", b, 2))
        self.assertEqual(cache.get("a", "missing"), "missing")

    def test_invalidation_is_per_key_and_failures_do_not_consume(self):
        cache, other = GenerationCache(), GenerationCache()
        a, b = cache.begin("a"), cache.begin("b")
        cache.invalidate("a")
        self.assertFalse(cache.commit("a", a, 1))
        self.assertFalse(cache.commit("b", -10, 2))
        self.assertTrue(cache.commit("b", b, 3))
        self.assertIsNone(other.get("b"))
        self.assertFalse(cache.commit("never", 900, 4))
''')

case("decimal_allocation", """Repair allocate_cents(total, weights) in module.py using only the standard library. total and each weight are Decimal objects or decimal strings; reject other types, invalid/nonfinite decimals, negative weights, empty weights, or weights summing to zero with ValueError. weights is a finite iterable and may be a one-shot generator. First round total to cents using round-half-to-even. Allocate the absolute integer cents proportionally: floor each exact share, then give remaining cents to the largest fractional remainders, breaking ties by earlier input index. Reapply the sign of the rounded total. Return a list of Decimal amounts, each with exponent -2, whose sum equals the rounded total. Zero weights receive zero. Negative totals use the same magnitude allocation with negative results. Avoid binary float and rounding intermediate proportions; finite decimal inputs must work independently of the caller's decimal precision. Do not mutate inputs. Preserve the interface.""", r'''
"""Allocate an invoice across weighted cost centers."""
from decimal import Decimal


def _number(value):
    return float(value)


def _shares(total, weights):
    weight_sum = sum(weights)
    return [total * weight / weight_sum for weight in weights]


def allocate_cents(total, weights):
    """Round proportional amounts and put the remainder in the last item.

    The final correction keeps the rounded entries balanced against total.
    Weight order follows the order of the supplied cost centers.
    """
    numeric_total = _number(total)
    numeric_weights = [_number(weight) for weight in weights]
    if not numeric_weights or sum(numeric_weights) == 0:
        raise ValueError("positive total weight required")
    if any(weight < 0 for weight in numeric_weights):
        raise ValueError("negative weight")
    amounts = [round(value, 2) for value in _shares(numeric_total, numeric_weights)]
    target = round(numeric_total, 2)
    correction = target - sum(amounts)
    amounts[-1] += correction
    result = []
    for amount in amounts:
        result.append(Decimal(str(round(amount, 2))))
    return result
''', r'''
from decimal import Decimal, InvalidOperation
from fractions import Fraction


def _number(value):
    if not isinstance(value, (str, Decimal)):
        raise ValueError("decimal or string required")
    try:
        parsed = Decimal(value)
    except (InvalidOperation, ValueError):
        raise ValueError("invalid decimal") from None
    if not parsed.is_finite():
        raise ValueError("nonfinite decimal")
    return Fraction(parsed)


def allocate_cents(total, weights):
    amount = _number(total)
    parsed = [_number(weight) for weight in weights]
    if not parsed or any(weight < 0 for weight in parsed) or not sum(parsed):
        raise ValueError("invalid weights")
    cents = round(amount * 100)
    total_weight = sum(parsed)
    shares = [abs(cents) * weight / total_weight for weight in parsed]
    floors = [share.numerator // share.denominator for share in shares]
    order = sorted(range(len(parsed)), key=lambda i: (-(shares[i] - floors[i]), i))
    for i in order[:abs(cents) - sum(floors)]:
        floors[i] += 1
    sign = int(cents < 0)
    return [Decimal((sign, tuple(map(int, str(value))), -2)) for value in floors]
''', r'''
import unittest
from decimal import Decimal
from module import allocate_cents


class PublicTests(unittest.TestCase):
    def test_even_shares(self):
        self.assertEqual(allocate_cents("10.00", ["1", "1"]), [Decimal("5.00"), Decimal("5.00")])

    def test_tied_remainders(self):
        self.assertEqual(allocate_cents("0.01", ["1", "1", "1"]),
                         [Decimal("0.01"), Decimal("0.00"), Decimal("0.00")])

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            allocate_cents("1.00", [])
''', r'''
import unittest
from decimal import Decimal, localcontext
from fractions import Fraction
from module import allocate_cents


class AcceptanceTests(unittest.TestCase):
    def test_half_even_negative_and_zero_weights(self):
        self.assertEqual(allocate_cents("1.005", ["1"]), [Decimal("1.00")])
        self.assertEqual(allocate_cents("1.015", ["1"]), [Decimal("1.02")])
        self.assertEqual(allocate_cents("-0.05", ["1", "1", "1"]), [Decimal("-0.02"), Decimal("-0.02"), Decimal("-0.01")])
        self.assertEqual(allocate_cents("0.03", ["0", "2", "1", "0"]),
                         [Decimal("0.00"), Decimal("0.02"), Decimal("0.01"), Decimal("0.00")])

    def test_non_tied_remainders(self):
        self.assertEqual(allocate_cents("0.07", ["1", "2", "3"]), [Decimal("0.01"), Decimal("0.02"), Decimal("0.04")])

    def test_precision_independent_exactness_and_exponents(self):
        weights = [Decimal("0.1000000000000000000000000001"), Decimal("0.1")]
        with localcontext() as context:
            context.prec = 4
            result = allocate_cents(Decimal("12345678901234567890.125"), iter(weights))
        self.assertEqual(sum(map(Fraction, result)), Fraction("12345678901234567890.12"))
        self.assertTrue(all(type(value) is Decimal and value.as_tuple().exponent == -2 for value in result))
        self.assertEqual(allocate_cents("0.01", weights), [Decimal("0.01"), Decimal("0.00")])
        self.assertEqual(weights, [Decimal("0.1000000000000000000000000001"), Decimal("0.1")])

    def test_invalid_values(self):
        for total in ["NaN", "Infinity", "bad", 1.0, 1, None]:
            with self.subTest(total=total), self.assertRaises(ValueError):
                allocate_cents(total, ["1"])
        for weights in [["-1", "2"], ["0", "0"], ["NaN"], ["Infinity"], ["bad"], [1], [True]]:
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                allocate_cents("1", weights)

    def test_small_integer_invariant(self):
        for cents in range(-21, 22):
            result = allocate_cents(str(Decimal(cents) / 100), ["3", "2", "5"])
            self.assertEqual(sum(result), Decimal(cents) / 100)
            shares = [Fraction(abs(cents) * w, 10) for w in (3, 2, 5)]
            actual = [abs(Fraction(value)) * 100 for value in result]
            self.assertTrue(all(share.numerator // share.denominator <= value <= -(-share.numerator // share.denominator)
                                for share, value in zip(shares, actual)))
''')

case("scoped_posix_paths", """Repair resolve_path(root, path) in module.py as pure POSIX string processing, with no filesystem access. root and path are strings without NUL (reject NUL with ValueError); root must be absolute or raise ValueError. Normalize root by collapsing repeated '/', removing '.', and processing '..'; attempting to go above '/' is invalid. For a relative path (including empty), start at normalized root and process components; reject '..' immediately if it would leave root, even if later components reenter. For an absolute path, normalize from '/', reject traversal above '/', then require the final normalized result to equal root or be beneath it by whole components. Absolute paths may leave/reenter root during normalization if the final result is in scope. Repeated separators and '.' are harmless. Return an absolute canonical string with no trailing slash except '/'. Backslash and percent sequences are literal filename characters, not separators or escape encodings. Root '/' permits all absolute results and relative paths that do not climb above '/'. Preserve the interface and use only the standard library.""", r'''
"""Resolve virtual resource names within a configured namespace."""


def _collapse(value):
    parts = []
    for component in value.split("/"):
        if component in ("", "."):
            continue
        if component == "..":
            if parts:
                parts.pop()
            continue
        parts.append(component)
    return "/" + "/".join(parts)


def _join(root, path):
    if path.startswith("/"):
        return path
    return root.rstrip("/") + "/" + path


def resolve_path(root, path):
    """Canonicalize the resource location and check its root prefix."""
    if not root.startswith("/"):
        raise ValueError("absolute root required")
    normalized_root = _collapse(root)
    candidate = _join(normalized_root, path)
    resolved = _collapse(candidate)
    if not resolved.startswith(normalized_root):
        raise ValueError("path outside root")
    return resolved
''', r'''
def _parts(value, initial=(), floor=0):
    result = list(initial)
    for part in value.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if len(result) <= floor:
                raise ValueError("path escapes scope")
            result.pop()
        else:
            result.append(part)
    return result


def resolve_path(root, path):
    if "\0" in root or "\0" in path or not root.startswith("/"):
        raise ValueError("invalid path")
    base = _parts(root)
    if path.startswith("/"):
        result = _parts(path)
        if result[:len(base)] != base:
            raise ValueError("outside root")
    else:
        result = _parts(path, base, len(base))
    return "/" + "/".join(result)
''', r'''
import unittest
from module import resolve_path


class PublicTests(unittest.TestCase):
    def test_relative_canonicalization(self):
        self.assertEqual(resolve_path("/srv/app", "a/./b/../c"), "/srv/app/a/c")

    def test_sibling_prefix_rejected(self):
        with self.assertRaises(ValueError):
            resolve_path("/srv/app", "/srv/application/config")

    def test_empty_path(self):
        self.assertEqual(resolve_path("/srv/app/", ""), "/srv/app")
''', r'''
import unittest
from module import resolve_path


class AcceptanceTests(unittest.TestCase):
    def test_normalized_roots_and_absolute_reentry(self):
        self.assertEqual(resolve_path("//srv/old/../app/.", "//srv/app/a//b/"), "/srv/app/a/b")
        self.assertEqual(resolve_path("/srv/app", "/srv/app/../app/x"), "/srv/app/x")
        self.assertEqual(resolve_path("/srv/app", "/srv/app/"), "/srv/app")
        self.assertEqual(resolve_path("/", "a/../"), "/")

    def test_relative_escape_rejected_at_boundary(self):
        for path in ["..", "../app/x", "a/../../app/x", "./../app"]:
            with self.subTest(path=path), self.assertRaises(ValueError):
                resolve_path("/srv/app", path)

    def test_above_root_nul_and_nonabsolute_root(self):
        for root, path in [("relative", "a"), ("/../../srv", "a"), ("/", "../x"),
                           ("/", "/../x"), ("/a\0b", "x"), ("/a", "x\0y"), ("/a", "/other")]:
            with self.subTest(root=root, path=path), self.assertRaises(ValueError):
                resolve_path(root, path)

    def test_literal_backslashes_and_percent(self):
        self.assertEqual(resolve_path("/app", r"..\secret"), r"/app/..\secret")
        self.assertEqual(resolve_path("/app", "%2e%2e/%2f/x"), "/app/%2e%2e/%2f/x")

    def test_canonical_result_is_idempotent(self):
        for root, path in [("/", "a//b/../c"), ("/a/b", "./x/../y//"), ("//a//", "b/c/..")]:
            result = resolve_path(root, path)
            self.assertEqual(resolve_path(root, result), result)
            self.assertFalse("//" in result)
            self.assertTrue(result == "/" or not result.endswith("/"))
''')


if __name__ == "__main__":
    suite = {"schema_version": 1, "scope": "diagnostic", "tasks": CASES}
    (ROOT.parents[1] / "configs/maintenance_cases_v1.json").write_text(json.dumps(suite, indent=2) + "\n", encoding="utf-8")
    provenance = {
        "schema_version": 1, "created_date": "2026-11-16", "origin": "self_authored",
        "suite": "maintenance_cases_v1", "scope": "diagnostic_development_only",
        "external_source_material": [], "task_count": len(CASES),
        "groups": "one distinct problem per group; all development; no fit/calibration/holdout",
        "limitations": ["Small authored cohort; not representative of industry maintenance tasks.",
                        "Not an independent holdout and not a standardized benchmark.",
                        "Tasks, acceptance tests and reference solutions share authorship.",
                        "Reference solutions verify specification consistency, not exhaustive correctness."],
        "provider_visible": ["task", "module.py", "test_public.py"],
        "provider_excluded": ["acceptance_test.py", "reference_solutions", "config.json", "provenance.json"],
        "acceptance_policy": "Acceptance checks run only after model editing; no acceptance failures are repair feedback.",
        "reference_policy": "Reference solutions are outside each case repository and must never be copied into grading workspaces.",
    }
    (ROOT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
