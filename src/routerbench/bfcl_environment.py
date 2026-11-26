"""Protected local BFCL multi_turn_base runtime for one pinned upstream commit.

Upstream checker and simulated classes remain unmodified Apache-2.0 files in
the verified snapshot (Gorilla authors; see its LICENSE and import manifest).
Only the execution binding is adapted: literal AST dispatch replaces eval,
instances/defaults are local, and no benchmark package/SDK is installed.
This is a local adaptation, not an official leaderboard implementation.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
from copy import deepcopy
import decimal
import hashlib
import importlib.machinery
import importlib.util
import inspect
import json
import math
from pathlib import Path
import re
import sys
import threading
import types

from .bfcl_data import COMMIT, read_snapshot


CLASSES = {"GorillaFileSystem": "gorilla_file_system", "MathAPI": "math_api", "MessageAPI": "message_api",
           "TwitterAPI": "posting_api", "TicketAPI": "ticket_api", "TradingBot": "trading_bot",
           "TravelAPI": "travel_booking", "VehicleControlAPI": "vehicle_control"}
BASE = "bfcl_eval.eval_checker.multi_turn_eval"
_LOCK = threading.RLock()
_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")
_DENIED_CALLS = {"eval", "exec", "compile", "open", "__import__", "breakpoint", "input"}
_HOST_MODULES = {"os", "sys", "subprocess", "socket", "requests", "urllib", "pathlib", "shutil"}
RESOURCE_POLICY = {"math_precision_max": 4096, "power_absolute_exponent_max": 1_000_000,
                   "integer_power_estimated_bits_max": 1_000_000, "output_utf8_bytes_max": 65536,
                   "scope": "local protected execution limits; activation is technical censoring, never model-quality failure"}


class BFCLCompatibilityError(ValueError):
    """Technical incompatibility with the protected snapshot, not model quality."""


class BFCLResourceLimit(BFCLCompatibilityError):
    """A local resource bound requires operational censoring without escalation."""

    def __init__(self, rule, limit, *, phase="preflight", method=None):
        self._details = {"category": "tool_resource_limit", "rule": rule, "limit": limit,
                         "phase": phase, "method": method, "capability_label": None}
        super().__init__(f"Protected BFCL resource bound exceeded: {rule}.")

    def details(self):
        return deepcopy(self._details)


def _literal(node, depth=0):
    if depth > 32:
        raise ValueError("Literal nesting exceeds the protected transport bound.")
    if isinstance(node, ast.Constant) and (node.value is None or type(node.value) in (str, bool, int, float)):
        if type(node.value) is float and not math.isfinite(node.value):
            raise ValueError("Nonfinite literals are unsupported.")
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _literal(node.operand, depth + 1)
        if type(value) not in (int, float):
            raise ValueError("Unary signs require numeric literals.")
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, (ast.List, ast.Tuple)):
        items = [_literal(item, depth + 1) for item in node.elts]
        return tuple(items) if isinstance(node, ast.Tuple) else items
    if isinstance(node, ast.Dict):
        result = {}
        for key, item in zip(node.keys, node.values):
            if key is None:
                raise ValueError("Dictionary expansion is not a literal.")
            key = _literal(key, depth + 1)
            if type(key) not in (str, int, float, bool) or key in result:
                raise ValueError("Duplicate or unsupported dictionary key.")
            result[key] = _literal(item, depth + 1)
        return result
    raise ValueError("Only scalar/container literals are accepted; expressions never execute.")


def decode_call(text, allowed_names):
    """Decode one bare whitelisted function call without eval or literal_eval."""
    if type(text) is not str or len(text.encode("utf-8")) > 65536:
        raise ValueError("A bounded function-call string is required.")
    try:
        tree = ast.parse(text, mode="eval")
    except (SyntaxError, RecursionError) as exc:
        raise ValueError("Invalid function-call syntax.") from exc
    if sum(1 for _ in ast.walk(tree)) > 8192:
        raise ValueError("Function-call AST exceeds the protected transport bound.")
    node = tree.body
    if (not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name)
            or not _NAME.fullmatch(node.func.id) or node.func.id not in allowed_names):
        raise ValueError("Only declared public function names are callable.")
    kwargs = {}
    for argument in node.keywords:
        if argument.arg is None or argument.arg in kwargs:
            raise ValueError("Keyword expansion or duplicate keyword.")
        kwargs[argument.arg] = _literal(argument.value)
    return {"name": node.func.id, "args": [_literal(item) for item in node.args], "arguments": kwargs}


def _json_arguments(arguments):
    def check(value, depth=0):
        if depth > 32:
            return False
        if value is None or type(value) in (str, bool, int):
            return True
        if type(value) is float:
            return math.isfinite(value)
        if type(value) is list:
            return all(check(item, depth + 1) for item in value)
        return type(value) is dict and all(type(key) is str and check(item, depth + 1) for key, item in value.items())
    if type(arguments) is not dict or not check(arguments):
        raise ValueError("Native arguments require finite JSON with string object keys.")
    if len(json.dumps(arguments, ensure_ascii=False).encode("utf-8")) > 65536:
        raise ValueError("Native arguments exceed the protected transport bound.")


def _native_schema(schema):
    if type(schema) is not dict:
        return deepcopy(schema)
    result = deepcopy(schema)
    aliases = {"dict": "object", "list": "array", "float": "number", "int": "integer", "bool": "boolean"}
    if type(result.get("type")) is str:
        result["type"] = aliases.get(result["type"], result["type"])
    for field in ("properties", "$defs", "definitions"):
        if type(result.get(field)) is dict:
            result[field] = {name: _native_schema(value) for name, value in result[field].items()}
    for field in ("items", "additionalProperties", "not"):
        if type(result.get(field)) is dict:
            result[field] = _native_schema(result[field])
    for field in ("anyOf", "allOf", "oneOf", "prefixItems"):
        if type(result.get(field)) is list:
            result[field] = [_native_schema(value) for value in result[field]]
    return result


def _audit_simulator_source(raw, path):
    """Fail closed on host-effect calls in the reviewed simulator subset."""
    tree = ast.parse(raw, filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func
        if isinstance(name, ast.Name) and name.id in _DENIED_CALLS:
            raise BFCLCompatibilityError(f"Host/code effect in simulator {path.name}:{node.lineno}.")
        while isinstance(name, ast.Attribute):
            name = name.value
        if isinstance(name, ast.Name) and name.id in _HOST_MODULES:
            raise BFCLCompatibilityError(f"Host effect in simulator {path.name}:{node.lineno}.")


class _VerifiedLoader(importlib.machinery.SourceFileLoader):
    def __init__(self, name, path, sha256):
        super().__init__(name, str(path))
        self.expected_sha256 = sha256

    def get_code(self, fullname):
        # Never trust/create a sidecar pyc in the imported snapshot.
        raw = Path(self.path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.expected_sha256:
            raise BFCLCompatibilityError("Source changed after snapshot verification.")
        return self.source_to_code(raw, self.path)


@contextmanager
def _isolated_imports(package):
    """Restore any existing bfcl_eval namespace; never import its __init__.py."""
    with _LOCK:
        previous = {name: value for name, value in sys.modules.items()
                    if name == "bfcl_eval" or name.startswith("bfcl_eval.")}
        for name in previous:
            del sys.modules[name]
        try:
            for name in ("bfcl_eval", "bfcl_eval.constants", "bfcl_eval.eval_checker", BASE, BASE + ".func_source_code"):
                module = types.ModuleType(name)
                module.__path__ = [str(package / name.replace(".", "/"))]
                module.__package__ = name
                sys.modules[name] = module
            yield
        finally:
            for name in list(sys.modules):
                if name == "bfcl_eval" or name.startswith("bfcl_eval."):
                    del sys.modules[name]
            sys.modules.update(previous)


def _bundle(metadata, involved):
    """Load only inspected official modules, with a fresh class universe."""
    package = Path(metadata["package_root"])
    hashes = {row["path"]: row["sha256"] for row in metadata["manifest"]["files"]}
    modules = {}
    def load(name, *, simulator=False):
        relative = "berkeley-function-call-leaderboard/" + name.replace(".", "/") + ".py"
        path = Path(metadata["upstream_root"]) / relative
        if simulator:
            _audit_simulator_source(path.read_bytes(), path)
        loader = _VerifiedLoader(name, path, hashes[relative])
        module = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, loader))
        sys.modules[name] = module
        loader.exec_module(module)
        modules[name] = module
        return module
    with _isolated_imports(package):
        load("bfcl_eval.constants.executable_backend_config")
        load(BASE + ".func_source_code.long_context", simulator=True)
        classes = {}
        for name in involved:
            try:
                module = load(BASE + ".func_source_code." + CLASSES[name], simulator=True)
            except ModuleNotFoundError as exc:
                raise BFCLCompatibilityError("Install the explicit BFCL optional dependency mpmath==1.3.0 for MathAPI.") from exc
            classes[name] = getattr(module, name)
        utils = load(BASE + ".multi_turn_utils")
        checker = load(BASE + ".multi_turn_checker")
    return classes, checker, utils


class BFCLRuntime:
    """Independent simulated state. Private checker results never enter tools."""

    def __init__(self, snapshot, entry):
        self._metadata = read_snapshot(snapshot)
        if (type(entry) is not dict or type(entry.get("id")) is not str
                or not entry["id"].startswith("multi_turn_base_") or type(entry.get("initial_config")) is not dict
                or type(entry.get("involved_classes")) is not list or not entry["involved_classes"]
                or any(type(name) is not str or name not in CLASSES for name in entry["involved_classes"])
                or len(set(entry["involved_classes"])) != len(entry["involved_classes"])
                or "missed_function" in entry):
            raise BFCLCompatibilityError("Only the pinned multi_turn_base class/entry contract is supported.")
        self._entry = deepcopy(entry)
        self._classes, self._checker, self._utils = _bundle(self._metadata, entry["involved_classes"])
        self._definitions = []
        folder = Path(self._metadata["package_root"]) / "bfcl_eval/data/multi_turn_func_doc"
        for name in entry["involved_classes"]:
            self._definitions.extend(json.loads(line) for line in (folder / (CLASSES[name] + ".json")).read_text(encoding="utf-8").splitlines() if line.strip())
        # Upstream populate_test_cases_with_predefined_functions does NOT filter
        # excluded_function in this commit. Preserve that observed behavior.
        self._names = {item["name"] for item in self._definitions}
        if len(self._names) != len(self._definitions) or any(not _NAME.fullmatch(name) for name in self._names):
            raise BFCLCompatibilityError("Duplicate or unsupported official tool names.")
        self._instances = self._new_instances()
        self._methods = self._method_map(self._instances)
        if not self._names <= set(self._methods):
            raise BFCLCompatibilityError("A documented function has no reviewed public method.")
        self._defaults = {}

    def _new_instances(self):
        result = {}
        for name in self._entry["involved_classes"]:
            result[name] = self._classes[name]()
            if name != "MathAPI":
                result[name]._load_scenario(deepcopy(self._entry["initial_config"].get(name, {})), long_context=False)
        return result

    @staticmethod
    def _method_map(instances):
        return {name: method for instance in instances.values()
                for name, method in inspect.getmembers(instance, predicate=inspect.ismethod) if not name.startswith("_")}

    def tools(self):
        return deepcopy(self._definitions)

    def native_tools(self):
        return [{"type": "function", "function": {"name": row["name"], "description": row["description"],
                  "parameters": _native_schema(row["parameters"])}} for row in self.tools()]

    def validate_batch(self, calls):
        if type(calls) is not list or not 1 <= len(calls) <= 128:
            raise ValueError("A nonempty bounded batch is required.")
        for call in calls:
            if (type(call) is not dict or set(call) != {"name", "arguments"}
                    or type(call["name"]) is not str or call["name"] not in self._names):
                raise ValueError("Only declared public tools may be dispatched.")
            _json_arguments(call["arguments"])
            self._resource_guard({"name": call["name"], "args": [], "arguments": call["arguments"]})

    def encode_call(self, name, arguments):
        self.validate_batch([{"name": name, "arguments": arguments}])
        text = name + "(" + ", ".join(f"{key}={value!r}" for key, value in arguments.items()) + ")"
        # Keywords must also be valid Python identifiers; no string execution.
        decode_call(text, self._names)
        return text

    def decode_call(self, text):
        return decode_call(text, self._names)

    def _resource_guard(self, call):
        """Bound expensive official arithmetic without evaluating any argument.

        Upstream power uses Python ** and precision APIs accept caller precision;
        these ceilings are local resource policy, not upstream argument ranges.
        Argument/type errors unrelated to these ceilings retain official behavior.
        """
        method = self._methods[call["name"]]
        if type(method.__self__).__name__ != "MathAPI":
            return
        try:
            values = inspect.signature(method).bind(*call["args"], **call["arguments"]).arguments
        except TypeError:
            return
        precision = values.get("precision")
        if type(precision) in (int, float) and precision > RESOURCE_POLICY["math_precision_max"]:
            raise BFCLResourceLimit("math_precision_max", RESOURCE_POLICY["math_precision_max"], method=call["name"])
        if call["name"] != "power":
            return
        base, exponent = values.get("base"), values.get("exponent")
        if type(exponent) in (int, float) and abs(exponent) > RESOURCE_POLICY["power_absolute_exponent_max"]:
            raise BFCLResourceLimit("power_absolute_exponent_max", RESOURCE_POLICY["power_absolute_exponent_max"], method="power")
        if type(base) is int and type(exponent) is int and exponent >= 0 and abs(base) > 1:
            # Conservative integer upper bound; no log/power computation needed.
            bits = abs(base).bit_length() * exponent
            if bits > RESOURCE_POLICY["integer_power_estimated_bits_max"]:
                raise BFCLResourceLimit("integer_power_estimated_bits_max", RESOURCE_POLICY["integer_power_estimated_bits_max"], method="power")

    @staticmethod
    def _check_output_scalars(value, name):
        # Avoid Python's integer-to-decimal limit becoming a spurious tool error.
        # The native interpreter limit is recorded along with the local policy.
        digit_limit = sys.get_int_max_str_digits()
        stack = [value]
        while stack:
            item = stack.pop()
            if type(item) is int and digit_limit and abs(item).bit_length() > int(digit_limit * math.log2(10)):
                raise BFCLResourceLimit("python_integer_string_digits_max", digit_limit, phase="output", method=name)
            if type(item) is dict:
                stack.extend(item.keys())
                stack.extend(item.values())
            elif type(item) in (list, tuple):
                stack.extend(item)

    def _invoke(self, call, instances, defaults):
        self._resource_guard(call)
        method = self._method_map(instances)[call["name"]]
        args, kwargs = deepcopy(call["args"]), deepcopy(call["arguments"])
        # Local adaptation: defaults omitted by the caller remain shared within
        # that runtime/method, but never with another policy or the gold world.
        try:
            bound = inspect.signature(method).bind(*args, **kwargs)
            for name, parameter in inspect.signature(method).parameters.items():
                if name not in bound.arguments and parameter.default is not inspect.Parameter.empty:
                    key = (call["name"], name)
                    if key not in defaults:
                        defaults[key] = deepcopy(parameter.default)
                    kwargs[name] = defaults[key]
        except TypeError:
            pass  # Preserve the method's own TypeError message and semantics.
        with _LOCK, decimal.localcontext():
            mp = sys.modules.get("mpmath")
            precision = mp.mp.dps if mp is not None else None
            try:
                result = method(*args, **kwargs)
                self._check_output_scalars(result, call["name"])
                if type(result) is str:
                    observation = result
                elif type(result) is dict:
                    try:
                        observation = json.dumps(result)
                    except (TypeError, ValueError):
                        observation = str(result)
                else:
                    observation = str(result)
                if len(observation.encode("utf-8")) > RESOURCE_POLICY["output_utf8_bytes_max"]:
                    raise BFCLResourceLimit("output_utf8_bytes_max", RESOURCE_POLICY["output_utf8_bytes_max"],
                                            phase="output", method=call["name"])
                public_error = type(result) is dict and bool(result.get("error"))
                return {"ok": not public_error, "observation": observation,
                        "error_type": "UpstreamToolError" if public_error else None}
            except BFCLResourceLimit:
                raise
            except Exception as exc:
                observation = f"Error during execution: {str(exc)}"
                if len(observation.encode("utf-8")) > RESOURCE_POLICY["output_utf8_bytes_max"]:
                    raise BFCLResourceLimit("output_utf8_bytes_max", RESOURCE_POLICY["output_utf8_bytes_max"],
                                            phase="output", method=call["name"])
                return {"ok": False, "observation": observation, "error_type": type(exc).__name__}
            finally:
                if mp is not None:
                    mp.mp.dps = precision

    def execute(self, name, arguments):
        try:
            text = self.encode_call(name, arguments)
        except BFCLResourceLimit:
            raise
        except ValueError:
            return {"ok": False, "observation": None, "error_type": "InvalidArguments"}
        return self._invoke(self.decode_call(text), self._instances, self._defaults)

    def execute_batch(self, calls):
        self.validate_batch(calls)
        decoded = [self.decode_call(self.encode_call(call["name"], call["arguments"])) for call in calls]
        return [self._invoke(call, self._instances, self._defaults) for call in decoded]

    def evaluate(self, decoded_turn_calls, private_groundtruth):
        """Terminal private checker on fresh model/gold worlds; no online mutation."""
        if type(private_groundtruth) is dict:
            if private_groundtruth.get("id", self._entry["id"]) != self._entry["id"]:
                raise BFCLCompatibilityError("Reference belongs to another conversation.")
            private_groundtruth = private_groundtruth.get("ground_truth")
        if (type(private_groundtruth) is not list or any(type(turn) is not list for turn in private_groundtruth)):
            raise BFCLCompatibilityError("An explicit private per-turn reference is required.")
        if type(decoded_turn_calls) is not list or len(decoded_turn_calls) != len(private_groundtruth):
            return {"valid": False, "error_type": "local_bfcl:incomplete_conversation"}
        for turn in private_groundtruth:
            try:
                for call in turn:
                    self._resource_guard(self.decode_call(call))
            except BFCLResourceLimit:
                raise
            except ValueError as exc:
                raise BFCLCompatibilityError("Reference uses a call outside the protected literal contract.") from exc
        try:
            for turn in decoded_turn_calls:
                if type(turn) is not list:
                    raise ValueError("Invalid turn.")
                for batch in turn:
                    if type(batch) is not list:
                        raise ValueError("Invalid batch.")
                    for call in batch:
                        self._resource_guard(self.decode_call(call))
        except BFCLResourceLimit:
            raise
        except ValueError:
            return {"valid": False, "error_type": "local_bfcl:invalid_call_ast"}
        # Fresh code/default universe, independent from online calls and from
        # previous grading. The actual pinned checker function is unchanged.
        classes, checker, _ = _bundle(self._metadata, self._entry["involved_classes"])
        worlds, defaults = {}, {}
        def dispatch(func_call_list, initial_config, involved_classes, model_name, test_entry_id,
                     long_context=False, is_evaL_run=False):
            if long_context or involved_classes != self._entry["involved_classes"] or initial_config != self._entry["initial_config"]:
                raise BFCLCompatibilityError("Checker requested a different simulator contract.")
            if model_name not in worlds:
                instances = {name: classes[name]() for name in involved_classes}
                for name, instance in instances.items():
                    if name != "MathAPI":
                        instance._load_scenario(deepcopy(initial_config.get(name, {})), long_context=False)
                worlds[model_name], defaults[model_name] = instances, {}
            return ([self._invoke(self.decode_call(call), worlds[model_name], defaults[model_name])["observation"]
                     for call in func_call_list], worlds[model_name])
        checker.execute_multi_turn_func_call = dispatch
        return checker.multi_turn_checker(deepcopy(decoded_turn_calls), deepcopy(private_groundtruth),
                                         deepcopy(self._entry), "multi_turn_base", "relay_local")

    def provenance(self):
        return {"commit": COMMIT, "snapshot_manifest_sha256": self._metadata["manifest_sha256"],
                "upstream_license": "Apache-2.0", "scope": "local protected multi_turn_base adaptation; not official leaderboard",
                "checker_source": "unchanged pinned multi_turn_checker.py with safe execution binding",
                "execution": "literal AST/direct public-method dispatch; original eval executor never called",
                "quoted_content": "preserved as data; no upstream regex method-name rewriting inside string literals",
                "default_isolation": "omitted defaults local per runtime/method; differs from upstream shared mutable defaults",
                "resource_policy": {**deepcopy(RESOURCE_POLICY), "python_integer_string_digits_max": sys.get_int_max_str_digits(),
                                    "output_limit_timing": "after method execution; local state may already have changed; never replay automatically"},
                "native_schema_mapping": "dict->object,list->array,float->number,int->integer,bool->boolean",
                "excluded_function": "preserved as upstream metadata; not filtered by pinned population helper"}
