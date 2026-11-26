"""Small, deterministic development tasks with joins and constrained planning.

These are synthetic integration environments, not an independent benchmark.
Instances are generated without observing any model outcomes. Tools operate
only on copied in-memory JSON; they never execute generated source or commands.
Public checks validate public constraints. Private final grading never supplies
a new observation to the agent, including whether an allocation is optimal.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from itertools import product
import random


CHALLENGE_FAMILIES = ("table_join", "dependency_plan", "constrained_allocation")
_TASK_FIELDS = {"id", "family", "source_group", "prompt", "max_steps", "initial_state", "private_reference"}


def _integer(value, lower=0, upper=None):
    return type(value) is int and value >= lower and (upper is None or value <= upper)


def _json(value):
    if value is None or type(value) in (str, bool, int):
        return True
    if type(value) is list:
        return all(_json(item) for item in value)
    return type(value) is dict and all(type(key) is str and _json(item) for key, item in value.items())


def _names(values):
    return (type(values) is list and bool(values) and all(type(value) is str and value.strip() for value in values)
            and len(values) == len(set(values)))


def _object(properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def _array(items, minimum=0, maximum=10):
    return {"type": "array", "items": items, "minItems": minimum, "maxItems": maximum}


def _matches(value, schema):
    kind = schema["type"]
    if kind == "object":
        return (type(value) is dict and set(value) == set(schema["properties"])
                and all(_matches(value[key], child) for key, child in schema["properties"].items()))
    if kind == "array":
        return (type(value) is list and schema.get("minItems", 0) <= len(value) <= schema["maxItems"]
                and all(_matches(item, schema["items"]) for item in value))
    if kind == "string":
        return type(value) is str and ("enum" not in schema or value in schema["enum"])
    if kind == "integer":
        return _integer(value, schema.get("minimum", 0), schema.get("maximum"))
    raise ValueError("Schema de challenge no soportado.")


def _join_answer(state):
    selected = {row["id"] for row in state["customers"] if row["active"] and row["region"] == state["region"]}
    rows = [row for row in state["orders"] if row["customer_id"] in selected and row["status"] == "paid"]
    return {"order_count": len(rows), "total": sum(row["amount"] for row in rows)}


def _closure(state):
    graph = {row["id"]: row["requires"] for row in state["nodes"]}
    needed, pending = set(), list(state["requested"])
    while pending:
        node = pending.pop()
        if node not in needed:
            needed.add(node)
            pending.extend(graph[node])
    return sorted(needed)


def _plan_check(state, order):
    violations = []
    needed = set(_closure(state))
    if len(order) != len(set(order)) or set(order) != needed:
        violations.append("RequiredNodeSet")
    positions = {name: index for index, name in enumerate(order)}
    for node in state["nodes"]:
        if node["id"] in positions and any(dependency not in positions or positions[dependency] >= positions[node["id"]]
                                           for dependency in node["requires"]):
            violations.append("DependencyOrder")
            break
    return {"all_passed": not violations, "violations": violations}


def _allocation_check(state, assignments):
    jobs = {row["id"]: row for row in state["jobs"]}
    capacities = {row["id"]: row["capacity"] for row in state["machines"]}
    names = [row["job"] for row in assignments]
    if len(names) != len(set(names)) or set(names) != set(jobs):
        return {"all_passed": False, "violations": ["JobCoverage"], "total_cost": None}
    loads = {machine: 0 for machine in capacities}
    total = 0
    for row in assignments:
        job, machine = jobs[row["job"]], row["machine"]
        if machine not in capacities or machine not in job["costs"]:
            return {"all_passed": False, "violations": ["IneligibleMachine"], "total_cost": None}
        loads[machine] += job["load"]
        total += job["costs"][machine]
    violations = ["CapacityExceeded"] if any(loads[name] > capacity for name, capacity in capacities.items()) else []
    return {"all_passed": not violations, "violations": violations, "total_cost": total}


def _optimal_cost(state):
    feasible = []
    for machines in product(*(tuple(job["costs"]) for job in state["jobs"])):
        assignment = [{"job": job["id"], "machine": machine} for job, machine in zip(state["jobs"], machines)]
        check = _allocation_check(state, assignment)
        if check["all_passed"]:
            feasible.append(check["total_cost"])
    if not feasible:
        raise ValueError("La instancia de asignación debe ser factible.")
    return min(feasible)


def validate_challenge_task(task):
    """Validate before environment construction, including bounded problem size."""
    if (type(task) is not dict or set(task) != _TASK_FIELDS or not _json(task)
            or any(type(task[key]) is not str or not task[key].strip() for key in ("id", "source_group", "prompt", "family"))
            or task["family"] not in CHALLENGE_FAMILIES or not _integer(task["max_steps"], 1, 64)):
        raise ValueError("Esquema de episodio challenge inválido.")
    state, reference, family = task["initial_state"], task["private_reference"], task["family"]
    if type(state) is not dict or type(reference) is not dict:
        raise ValueError("El estado y referencia deben ser objetos.")
    valid = False
    if family == "table_join":
        valid = (set(state) == {"customers", "orders", "region"} and type(state["region"]) is str
                 and type(state["customers"]) is list and 2 <= len(state["customers"]) <= 8
                 and type(state["orders"]) is list and 2 <= len(state["orders"]) <= 16
                 and all(type(row) is dict and set(row) == {"id", "name", "region", "active"}
                         and all(type(row[key]) is str and row[key] for key in ("id", "name", "region"))
                         and type(row["active"]) is bool for row in state["customers"])
                 and all(type(row) is dict and set(row) == {"id", "customer_id", "amount", "status"}
                         and type(row["id"]) is str and type(row["customer_id"]) is str
                         and _integer(row["amount"], -1000, 1000) and row["status"] in ("paid", "cancelled")
                         for row in state["orders"]))
        if valid:
            customer_ids = [row["id"] for row in state["customers"]]
            valid = (_names(customer_ids) and _names([row["id"] for row in state["orders"]])
                     and all(row["customer_id"] in customer_ids for row in state["orders"])
                     and type(reference.get("answer")) is dict
                     and all(type(value) is int for value in reference["answer"].values())
                     and reference == {"answer": _join_answer(state)})
    elif family == "dependency_plan":
        valid = (set(state) == {"nodes", "requested"} and type(state["nodes"]) is list
                 and 2 <= len(state["nodes"]) <= 10 and _names(state["requested"])
                 and all(type(row) is dict and set(row) == {"id", "requires"} and type(row["id"]) is str
                         and row["id"] and type(row["requires"]) is list
                         and all(type(name) is str for name in row["requires"])
                         and len(row["requires"]) == len(set(row["requires"])) for row in state["nodes"]))
        if valid:
            names = [row["id"] for row in state["nodes"]]
            valid = (_names(names) and set(state["requested"]) <= set(names)
                     and all(set(row["requires"]) <= set(names) for row in state["nodes"]))
            if valid:
                resolved = set()
                for _ in names:
                    resolved.update(row["id"] for row in state["nodes"] if set(row["requires"]) <= resolved)
                valid = len(resolved) == len(names) and reference == {"required_nodes": _closure(state)}
    else:
        valid = (set(state) == {"jobs", "machines"} and type(state["jobs"]) is list and 2 <= len(state["jobs"]) <= 5
                 and type(state["machines"]) is list and 2 <= len(state["machines"]) <= 3
                 and all(type(row) is dict and set(row) == {"id", "capacity"} and type(row["id"]) is str
                         and _integer(row["capacity"], 1, 20) for row in state["machines"])
                 and all(type(row) is dict and set(row) == {"id", "load", "costs"} and type(row["id"]) is str
                         and _integer(row["load"], 1, 5) and type(row["costs"]) is dict and row["costs"]
                         and all(_integer(value, 1, 100) for value in row["costs"].values()) for row in state["jobs"]))
        if valid:
            machines = [row["id"] for row in state["machines"]]
            valid = (_names(machines) and _names([row["id"] for row in state["jobs"]])
                     and all(set(row["costs"]) <= set(machines) for row in state["jobs"]))
            if valid:
                valid = type(reference.get("optimal_cost")) is int and reference == {"optimal_cost": _optimal_cost(state)}
    if not valid:
        raise ValueError("Restricciones, referencias o estado challenge inválidos.")


def build_challenge_tasks(seed=42, per_family=4):
    """Generate new cases without outcomes, rankings, failures or model selection."""
    if type(seed) is not int or type(per_family) is not int or per_family < 1:
        raise ValueError("seed entero y per_family entero positivo requeridos.")
    tasks = []
    for family in CHALLENGE_FAMILIES:
        for index in range(per_family):
            identity = f"challenge-{seed}-{family}-{index:03d}"
            tag = hashlib.sha256(identity.encode()).hexdigest()[:8]
            rng = random.Random(identity)
            if family == "table_join":
                region = rng.choice(("north", "south"))
                other = "south" if region == "north" else "north"
                customers = [{"id": f"c{i}", "name": name, "region": area, "active": active}
                             for i, (name, area, active) in enumerate((("Alex", region, True), ("Alex", region, False),
                                                                      ("Sam", other, True), ("Lee", region, True)))]
                orders = [{"id": f"o{i}", "customer_id": f"c{customer}", "amount": rng.randint(-8, 60), "status": status}
                          for i, (customer, status) in enumerate(((0, "paid"), (0, "cancelled"), (1, "paid"),
                                                                  (2, "paid"), (3, "paid"), (3, "paid")))]
                rng.shuffle(customers)
                rng.shuffle(orders)
                state = {"customers": customers, "orders": orders, "region": region}
                reference = {"answer": _join_answer(state)}
                prompt = (f"Caso {tag}: lee customers y orders. Une por customers.id=orders.customer_id, nunca por name. "
                          f"Cuenta y suma amount de órdenes status=paid de clientes active=true y region={region}. "
                          "Incluye importes negativos. Guarda exactamente order_count y total con submit_summary; responde done.")
            elif family == "dependency_plan":
                names = ["fetch", "parse", "clean", "validate", "publish", "archive", "audit"]
                nodes = [{"id": names[i], "requires": dependencies} for i, dependencies in enumerate(
                    ([], ["fetch"], ["parse"], ["clean"], ["validate"], [rng.choice(("parse", "clean"))], []))]
                requested = ["publish"] + (["archive"] if index % 2 else [])
                rng.shuffle(nodes)
                state = {"nodes": nodes, "requested": requested}
                reference = {"required_nodes": _closure(state)}
                prompt = (f"Caso {tag}: lee read_graph. Planifica solo requested y todos sus prerrequisitos transitivos, "
                          "cada nodo una vez y siempre después de sus dependencias. Entre órdenes válidos cualquiera sirve. "
                          "Guarda el array order con submit_plan, ejecuta check_plan y responde done cuando sea válido.")
            else:
                machines = [{"id": f"m{i}", "capacity": 0} for i in range(3)]
                jobs = []
                for i in range(3 + index % 2):
                    load, base = rng.randint(1, 3), i % len(machines)
                    machines[base]["capacity"] += load
                    eligible = {machines[base]["id"], machines[(base + 1) % 3]["id"]}
                    jobs.append({"id": f"j{i}", "load": load, "costs": {name: rng.randint(1, 9) for name in sorted(eligible)}})
                for machine in machines:
                    machine["capacity"] = max(1, machine["capacity"] + rng.randint(0, 1))
                rng.shuffle(jobs)
                rng.shuffle(machines)
                state = {"jobs": jobs, "machines": machines}
                reference = {"optimal_cost": _optimal_cost(state)}
                prompt = (f"Caso {tag}: lee read_allocation. Asigna cada job a una máquina presente en sus costs sin "
                          "exceder capacity con la suma de load. Minimiza la suma de costs; cualquier óptimo sirve. "
                          "Guarda assignments como objetos job/machine con submit_assignment, ejecuta check_assignment "
                          "y responde done. La comprobación pública informa factibilidad y costo, no optimalidad.")
            task = {"id": identity, "family": family, "source_group": identity, "prompt": prompt,
                    "max_steps": 8, "initial_state": state, "private_reference": reference}
            validate_challenge_task(task)
            tasks.append(task)
    return tasks


class ChallengeEnvironment:
    def __init__(self, task):
        validate_challenge_task(task)
        self._task = deepcopy(task)
        self._state = deepcopy(task["initial_state"])
        self._reference = deepcopy(task["private_reference"])
        self._read = set()
        self._calls, self._revision = 0, 0
        self._output, self._tested_revision, self._test_report = None, None, None

    def tools(self):
        family = self._task["family"]
        if family == "table_join":
            specs = [("read_table", "Lee una de las dos tablas públicas.", {"name": {"type": "string", "enum": ["customers", "orders"]}}),
                     ("submit_summary", "Guarda el resumen calculado.", {"order_count": {"type": "integer", "minimum": 0, "maximum": 16},
                                                                         "total": {"type": "integer", "minimum": -16000, "maximum": 16000}})]
        elif family == "dependency_plan":
            specs = [("read_graph", "Lee nodos, dependencias y objetivos requested.", {}),
                     ("submit_plan", "Guarda el orden propuesto de nodos.", {"order": _array({"type": "string"}, 1)}),
                     ("check_plan", "Comprueba el plan contra las restricciones públicas.", {})]
        else:
            specs = [("read_allocation", "Lee cargas, costos elegibles y capacidades.", {}),
                     ("submit_assignment", "Guarda una máquina por trabajo.", {"assignments": _array(_object({"job": {"type": "string"},
                                                                                                              "machine": {"type": "string"}}), 1, 5)}),
                     ("check_assignment", "Comprueba factibilidad y calcula costo; no revela optimalidad.", {})]
        return [{"name": name, "description": description, "parameters": _object(properties)} for name, description, properties in specs]

    def observation(self):
        family = self._task["family"]
        read = self._read == {"customers", "orders"} if family == "table_join" else bool(self._read)
        return deepcopy({"family": family, "task_id": self._task["id"], "prompt": self._task["prompt"],
                         "max_steps": self._task["max_steps"], "successful_tool_calls": self._calls,
                         "progress": {"records_read": read, "revision": self._revision, "tested_revision": self._tested_revision,
                                      "output": self._output, "test_report": self._test_report}})

    def execute(self, name, arguments):
        def failure(error):
            return {"ok": False, "observation": None, "error_type": error}
        definitions = {tool["name"]: tool for tool in self.tools()}
        if type(name) is not str or name not in definitions:
            return failure("UnknownTool")
        if not _matches(arguments, definitions[name]["parameters"]):
            return failure("InvalidArguments")
        if self._calls >= self._task["max_steps"]:
            return failure("StepLimitExceeded")
        if name in ("check_plan", "check_assignment") and self._output is None:
            return failure("MissingSubmission")
        if name == "read_table":
            table = arguments["name"]
            self._read.add(table)
            result = {"name": table, "rows": self._state[table]}
        elif name in ("read_graph", "read_allocation"):
            self._read.add(name)
            result = self._state
        elif name.startswith("submit_"):
            self._output = deepcopy(arguments)
            self._revision += 1
            self._tested_revision, self._test_report = None, None
            result = {"written": True, "revision": self._revision}
        else:
            self._test_report = (_plan_check(self._state, self._output["order"]) if name == "check_plan"
                                 else _allocation_check(self._state, self._output["assignments"]))
            self._tested_revision = self._revision
            result = {**self._test_report, "revision": self._revision}
        self._calls += 1
        return {"ok": True, "observation": deepcopy(result), "error_type": None}

    @staticmethod
    def verification_error(name, result):
        if (name in ("check_plan", "check_assignment") and type(result) is dict and result.get("ok") is True
                and type(result.get("observation")) is dict and result["observation"].get("all_passed") is False):
            return "ConstraintViolation"
        return None

    def finish(self, answer):
        """Terminal private grade only; no mutation, feedback or implicit actions."""
        if type(answer) is not str or answer.strip().lower() != "done" or self._output is None:
            return False
        family = self._task["family"]
        if family == "table_join":
            return self._read == {"customers", "orders"} and self._output == self._reference["answer"]
        if not self._read or self._tested_revision != self._revision or not self._test_report["all_passed"]:
            return False
        if family == "dependency_plan":
            return sorted(self._output["order"]) == self._reference["required_nodes"]
        return self._test_report["total_cost"] == self._reference["optimal_cost"]


def create_challenge_environment(task):
    return ChallengeEnvironment(task)
