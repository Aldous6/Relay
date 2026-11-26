"""Small deterministic agent environments, entirely in memory.

These are synthetic integration tasks, not a representative agent benchmark.
The repair family changes four fields of a fixed arithmetic/clipping template;
it never executes generated source code. Passing its finite test suite is not a
proof of general program correctness. Task dictionaries include private grading
material: only observation() and tools() may be sent to a model by a harness.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import random

from .features import normalize_for_dedup

FAMILIES = ("file_lookup", "aggregate_records", "repair_function")
TASK_FIELDS = {"id", "family", "source_group", "prompt", "max_steps", "initial_state", "private_reference"}
SPEC_FIELDS = {"operator", "offset", "lower", "upper"}
OPERATORS = ("add", "subtract", "multiply")


def _integer(value, lower=None, upper=None):
    return (type(value) is int and (lower is None or value >= lower)
            and (upper is None or value <= upper))


def _plain_json(value):
    if value is None or type(value) in (str, bool, int):
        return True
    if type(value) is list:
        return all(_plain_json(item) for item in value)
    if type(value) is dict:
        return all(type(key) is str and _plain_json(item) for key, item in value.items())
    return False


def _valid_spec(spec):
    return (type(spec) is dict and set(spec) == SPEC_FIELDS and spec["operator"] in OPERATORS
            and all(_integer(spec[name], -1000, 1000) for name in ("offset", "lower", "upper"))
            and spec["lower"] < spec["upper"])


def _calculate(spec, operand, value):
    if spec["operator"] == "add":
        result = value + operand
    elif spec["operator"] == "subtract":
        result = value - operand
    else:
        result = value * operand
    return max(spec["lower"], min(spec["upper"], result + spec["offset"]))


def _aggregate(records, category):
    amounts = [row["amount"] for row in records if row["category"] == category]
    return {"category": category, "count": len(amounts), "total": sum(amounts)}


def _validate_task(task):
    if type(task) is not dict or set(task) != TASK_FIELDS or not _plain_json(task):
        raise ValueError("Una tarea requiere el esquema completo de episodio, sin filas de pasos ni campos extra.")
    if task["family"] in ("table_join", "dependency_plan", "constrained_allocation"):
        from .agent_challenges import validate_challenge_task
        validate_challenge_task(task)
        return
    if (any(type(task[name]) is not str or not task[name].strip() for name in ("id", "family", "source_group", "prompt"))
            or task["family"] not in FAMILIES or not _integer(task["max_steps"], 1)):
        raise ValueError("Identidad, familia, prompt o limite de pasos invalido.")
    state, reference = task["initial_state"], task["private_reference"]
    if type(state) is not dict or type(reference) is not dict:
        raise ValueError("El estado inicial y la referencia privada deben ser objetos.")
    family = task["family"]
    if family == "file_lookup":
        valid = (set(state) == {"files"} and type(state["files"]) is dict and bool(state["files"])
                 and all(type(path) is str and path and type(content) is str for path, content in state["files"].items())
                 and set(reference) == {"path", "answer"} and type(reference["path"]) is str
                 and type(reference["answer"]) is str and bool(reference["answer"].strip())
                 and reference["path"] in state["files"]
                 and reference["answer"] in state["files"][reference["path"]])
    elif family == "aggregate_records":
        valid = (set(state) == {"records_path", "output_path", "records"}
                 and all(type(state[name]) is str and state[name] for name in ("records_path", "output_path"))
                 and state["records_path"] != state["output_path"]
                 and type(state["records"]) is list and bool(state["records"])
                 and all(type(row) is dict and set(row) == {"category", "amount"}
                         and type(row["category"]) is str and bool(row["category"])
                         and _integer(row["amount"], -1000, 1000) for row in state["records"])
                 and set(reference) == {"answer"} and type(reference["answer"]) is dict
                 and set(reference["answer"]) == {"category", "count", "total"}
                 and type(reference["answer"]["category"]) is str
                 and _integer(reference["answer"]["count"], 0) and _integer(reference["answer"]["total"])
                 and reference["answer"] == _aggregate(state["records"], reference["answer"]["category"]))
    else:
        valid = (set(state) == {"operand", "function"} and _integer(state["operand"], -1000, 1000)
                 and _valid_spec(state["function"]) and set(reference) == {"expected_spec", "test_inputs"}
                 and _valid_spec(reference["expected_spec"]) and type(reference["test_inputs"]) is list
                 and len(reference["test_inputs"]) >= 2
                 and all(_integer(value, -100000, 100000) for value in reference["test_inputs"]))
    if not valid:
        raise ValueError("El estado o la referencia no cumplen el contrato de la familia.")


def build_tasks(seed=42, per_family=4):
    """Build JSON-serializable episodes; changing the seed changes their sources."""
    if not _integer(seed) or not _integer(per_family, 1):
        raise ValueError("seed debe ser entero y per_family un entero positivo.")
    rng = random.Random(seed)
    tasks = []
    for family in FAMILIES:
        for index in range(per_family):
            identity = f"synthetic-{seed}-{family}-{index:03d}"
            tag = hashlib.sha256(identity.encode()).hexdigest()[:12]
            if family == "file_lookup":
                paths = [f"notes/{tag}-{position}.txt" for position in range(3)]
                target = rng.randrange(len(paths))
                record = f"record-{tag}-{target}"
                values = [f"CODE-{rng.randrange(100000, 1000000)}" for _ in paths]
                state = {"files": {path: f"Record: record-{tag}-{position}\nAccess code: {values[position]}\n"
                                   for position, path in enumerate(paths)}}
                reference = {"path": paths[target], "answer": values[target]}
                prompt = (f"Lista los archivos virtuales y lee el expediente Record: {record}. "
                          "Responde solamente su Access code. Debes listar y leer el expediente correcto antes de responder.")
                max_steps = 8
            elif family == "aggregate_records":
                category = rng.choice(("north", "south", "east"))
                records = [{"category": rng.choice(("north", "south", "east")), "amount": rng.randint(-20, 80)}
                           for _ in range(8)]
                records.append({"category": category, "amount": rng.randint(1, 20)})
                state = {"records_path": f"input/records-{tag}.json", "output_path": f"output/summary-{tag}.json", "records": records}
                reference = {"answer": _aggregate(records, category)}
                prompt = (f"Lee los registros de {state['records_path']}. Para category={category}, calcula count "
                          f"y la suma total de amount. Escribe en {state['output_path']} el objeto JSON "
                          "con exactamente category, count y total. Debes leer y escribir con herramientas; despues responde done.")
                max_steps = 6
            else:
                operand = rng.randint(2, 7)
                target = {"operator": OPERATORS[index % len(OPERATORS)], "offset": rng.randint(-5, 5),
                          "lower": rng.randint(-20, -8), "upper": rng.randint(8, 20)}
                broken = {**target, "operator": OPERATORS[(index + 1) % len(OPERATORS)], "offset": target["offset"] + 1}
                state = {"operand": operand, "function": broken}
                reference = {"expected_spec": target, "test_inputs": [-100, -11, -1, 0, 1, 3, 12, 100]}
                required = ", ".join(f"{key}={value}" for key, value in {"operand": operand, **target}.items())
                prompt = (f"Repara la funcion restringida del caso {tag}. Especificacion requerida: {required}. "
                          "La plantilla aplica operator a x y operand, suma offset y recorta al intervalo inclusivo [lower, upper]. "
                          "Inspecciona, modifica sus campos con patch_function y ejecuta run_tests sobre la version reparada. "
                          "Cuando este lista responde done. Es una plantilla aritmetica limitada, no codigo Python libre.")
                max_steps = 8
            task = {"id": identity, "family": family, "source_group": identity, "prompt": prompt,
                    "max_steps": max_steps, "initial_state": state, "private_reference": reference}
            _validate_task(task)
            tasks.append(task)
    return tasks


def _object_schema(properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def _matches_schema(value, schema):
    kind = schema["type"]
    if kind == "object":
        return (type(value) is dict and set(value) == set(schema["properties"])
                and all(_matches_schema(value[key], child) for key, child in schema["properties"].items()))
    if kind == "string":
        return type(value) is str and ("enum" not in schema or value in schema["enum"])
    if kind == "integer":
        return _integer(value, schema.get("minimum"), schema.get("maximum"))
    raise ValueError("Tipo de schema no soportado por el entorno sintetico.")


class _MemoryEnvironment:
    def __init__(self, task):
        _validate_task(task)
        self._task = deepcopy(task)
        self._state = deepcopy(task["initial_state"])
        self._reference = deepcopy(task["private_reference"])
        self._calls = 0
        self._listed = False
        self._read_paths = set()
        self._output = None
        self._inspected = False
        self._revision = 0
        self._tested_revision = None
        self._test_report = None

    def tools(self):
        """Return copies of public tool descriptions and strict JSON schemas."""
        family, state = self._task["family"], self._state
        if family == "file_lookup":
            specs = [("list_files", "Lista las rutas de los archivos virtuales.", {}),
                     ("read_file", "Lee un archivo virtual por su ruta exacta.", {"path": {"type": "string"}})]
        elif family == "aggregate_records":
            document = _object_schema({"category": {"type": "string"},
                                       "count": {"type": "integer", "minimum": 0, "maximum": len(state["records"])},
                                       "total": {"type": "integer", "minimum": -1000 * len(state["records"]),
                                                 "maximum": 1000 * len(state["records"])}})
            specs = [("read_records", "Lee los registros del archivo virtual de entrada.", {"path": {"type": "string"}}),
                     ("write_json", "Escribe el resultado estructurado en el archivo virtual de salida.",
                      {"path": {"type": "string"}, "document": document})]
        else:
            fields = {"operator": {"type": "string", "enum": list(OPERATORS)},
                      **{name: {"type": "integer", "minimum": -1000, "maximum": 1000}
                         for name in ("offset", "lower", "upper")}}
            specs = [("inspect_function", "Inspecciona la plantilla y sus campos actuales.", {}),
                     ("patch_function", "Sustituye los cuatro campos; lower debe ser menor que upper. No admite codigo.", fields),
                     ("run_tests", "Ejecuta comprobaciones aritmeticas deterministas de la version actual.", {})]
        return [{"name": name, "description": description, "parameters": _object_schema(properties)}
                for name, description, properties in specs]

    def observation(self):
        """Public state only; private references and unrequested files stay hidden."""
        result = {key: self._task[key] for key in ("family", "prompt", "max_steps")}
        result.update(task_id=self._task["id"], successful_tool_calls=self._calls)
        if self._task["family"] == "file_lookup":
            result["progress"] = {"listed_files": self._listed, "read_paths": sorted(self._read_paths)}
        elif self._task["family"] == "aggregate_records":
            result["progress"] = {"records_read": self._state["records_path"] in self._read_paths,
                                  "output": self._output}
        else:
            result["progress"] = {"inspected": self._inspected, "revision": self._revision,
                                  "tested_revision": self._tested_revision, "test_report": self._test_report}
        return deepcopy(result)

    def _tests(self):
        passed = sum(_calculate(self._state["function"], self._state["operand"], value)
                     == _calculate(self._reference["expected_spec"], self._state["operand"], value)
                     for value in self._reference["test_inputs"])
        total = len(self._reference["test_inputs"])
        return {"passed": passed, "total": total, "all_passed": passed == total}

    def execute(self, name, args):
        """Validate before mutation. Invalid actions leave all environment state intact."""
        def failure(kind):
            return {"ok": False, "observation": None, "error_type": kind}
        definitions = {tool["name"]: tool for tool in self.tools()}
        if type(name) is not str or name not in definitions:
            return failure("UnknownTool")
        if not _matches_schema(args, definitions[name]["parameters"]):
            return failure("InvalidArguments")
        if self._calls >= self._task["max_steps"]:
            return failure("StepLimitExceeded")
        if name == "read_file" and args["path"] not in self._state["files"]:
            return failure("UnknownVirtualPath")
        if name == "read_records" and args["path"] != self._state["records_path"]:
            return failure("UnknownVirtualPath")
        if name == "write_json" and args["path"] != self._state["output_path"]:
            return failure("UnknownVirtualPath")
        if name == "patch_function" and args["lower"] >= args["upper"]:
            return failure("InvalidBounds")
        if name == "list_files":
            self._listed = True
            value = {"paths": sorted(self._state["files"])}
        elif name == "read_file":
            self._read_paths.add(args["path"])
            value = {"path": args["path"], "content": self._state["files"][args["path"]]}
        elif name == "read_records":
            self._read_paths.add(args["path"])
            value = {"path": args["path"], "records": self._state["records"]}
        elif name == "write_json":
            self._output = deepcopy(args["document"])
            value = {"path": args["path"], "written": True}
        elif name == "inspect_function":
            self._inspected = True
            value = {"template": "clip(operator(x, operand) + offset, lower, upper)",
                     "operand": self._state["operand"], "fields": self._state["function"], "revision": self._revision}
        elif name == "patch_function":
            self._state["function"] = deepcopy(args)
            self._revision += 1
            self._tested_revision = None
            self._test_report = None
            value = {"revision": self._revision, "updated": True}
        else:
            self._test_report = self._tests()
            self._tested_revision = self._revision
            value = {**self._test_report, "revision": self._revision}
        self._calls += 1
        return {"ok": True, "observation": deepcopy(value), "error_type": None}

    @staticmethod
    def verification_error(name, result):
        """Identify failed checks using only the public tool feedback.

        A successful tool invocation can report failing arithmetic tests. This
        signal never consults the final grader or the private reference.
        """
        if (name == "run_tests" and type(result) is dict and result.get("ok") is True
                and type(result.get("observation")) is dict
                and result["observation"].get("all_passed") is False):
            return "TestFailure"
        return None

    def finish(self, answer):
        """Grade the final state; the harness ends the episode on either result.

        This method has no state side effects. A harness must not expose its
        private boolean to an agent to retry or escalate a final answer.
        """
        if type(answer) is not str:
            return False
        if self._task["family"] == "file_lookup":
            return (self._listed and self._reference["path"] in self._read_paths
                    and normalize_for_dedup(answer) == normalize_for_dedup(self._reference["answer"]))
        if normalize_for_dedup(answer) != "done":
            return False
        if self._task["family"] == "aggregate_records":
            return self._state["records_path"] in self._read_paths and self._output == self._reference["answer"]
        return (self._inspected and self._revision > 0 and self._tested_revision == self._revision
                and self._tests()["all_passed"])


def create_environment(task):
    """Create an independent reset; mutations cannot reach the task or another policy."""
    if type(task) is dict and task.get("family") in ("table_join", "dependency_plan", "constrained_allocation"):
        from .agent_challenges import create_challenge_environment
        return create_challenge_environment(task)
    return _MemoryEnvironment(task)


def validate_task_splits(splits, require_unseen_families=False):
    """Audit whole episodes; source groups may repeat only within one split.

    Optional strict mode requires an ``evaluation`` partition whose families
    occur in no other partition. Checks cover supplied annotations and normalized
    prompt equality, not semantic independence or representative generalization.
    """
    if (type(splits) is not dict or not splits or type(require_unseen_families) is not bool
            or any(type(name) is not str or not name.strip() or type(tasks) is not list or not tasks
                   for name, tasks in splits.items())):
        raise ValueError("Se requieren particiones nombradas de episodios completos no vacias.")
    ids, prompts, source_splits, families, report = set(), set(), {}, {}, {}
    for split, tasks in splits.items():
        sources = set()
        families[split] = set()
        for task in tasks:
            _validate_task(task)
            normalized = normalize_for_dedup(task["prompt"])
            if task["id"] in ids or normalized in prompts:
                raise ValueError("ID o tarea normalizada duplicada dentro o entre particiones.")
            source = task["source_group"]
            if source in source_splits and source_splits[source] != split:
                raise ValueError("Un source_group cruza particiones; mantener sus episodios y pasos juntos.")
            ids.add(task["id"])
            prompts.add(normalized)
            source_splits[source] = split
            sources.add(source)
            families[split].add(task["family"])
        report[split] = {"episodes": len(tasks), "source_groups": len(sources), "families": sorted(families[split])}
    if require_unseen_families:
        if "evaluation" not in splits or len(splits) < 2:
            raise ValueError("El modo estricto requiere evaluation y otra particion.")
        other_families = set().union(*(values for name, values in families.items() if name != "evaluation"))
        if families["evaluation"] & other_families:
            raise ValueError("Las familias de evaluation deben estar ausentes de las otras particiones.")
    return {"partitions": report, "require_unseen_families": require_unseen_families,
            "whole_episodes": True, "independence_certified": False,
            "scope": "synthetic tasks; supplied source/family labels and normalized prompt overlap only"}
