"""Scripted integration fixture, NOT an LLM, trained router or quality benchmark.

It solves only this tiny tool protocol using public observations. Fixed token
counts are fictitious counters for testing accounting, never measured tokens.
"""
import json
import re

from relay_router.agent import ModelReply


class ScriptedToolFixture:
    def __init__(self, weak="simulated/weak"):
        self.weak = weak
        self.calls = 0

    def __call__(self, model, request):
        self.calls += 1
        tools = [entry["content"] for entry in request.history
                 if entry["role"] == "tool" and isinstance(entry["content"], dict)
                 and entry["content"].get("ok") is True]
        by_name = {}
        for entry in tools:
            by_name.setdefault(entry["name"], []).append(entry["observation"])
        family = ("aggregate_records" if "read_records" in request.state.available_tools else
                  "repair_function" if "inspect_function" in request.state.available_tools else "file_lookup")
        if model == self.weak and not request.state.previous_errors and request.state.step == 0 and family != "file_lookup":
            # Deliberately inject exactly one syntax failure to exercise retries.
            content = "{invalid fixture JSON"
        else:
            action = self._next(request.prompt, family, by_name)
            content = json.dumps(action, ensure_ascii=False)
        return ModelReply(content, usage={"input_tokens": 100, "output_tokens": 20, "cached_input_tokens": 0})

    @staticmethod
    def _next(prompt, family, observations):
        def tool(name, **arguments):
            return {"type": "tool", "name": name, "arguments": arguments}
        def final(answer):
            return {"type": "final", "answer": answer}
        if family == "file_lookup":
            if "list_files" not in observations:
                return tool("list_files")
            record = re.search(r"Record: (record-[\w-]+)", prompt).group(1)
            read = observations.get("read_file", [])
            for row in read:
                if f"Record: {record}\n" in row["content"]:
                    return final(re.search(r"Access code: ([^\n]+)", row["content"]).group(1))
            paths = observations["list_files"][-1]["paths"]
            remaining = [p for p in paths if p not in {r["path"] for r in read}]
            return tool("read_file", path=remaining[0]) if remaining else final("not found")
        if family == "aggregate_records":
            if "read_records" not in observations:
                path = re.search(r"registros de ([^ ]+)\.", prompt).group(1)
                return tool("read_records", path=path)
            if "write_json" not in observations:
                category = re.search(r"category=(\w+)", prompt).group(1)
                amounts = [r["amount"] for r in observations["read_records"][-1]["records"] if r["category"] == category]
                path = re.search(r"Escribe en ([^ ]+) el", prompt).group(1)
                return tool("write_json", path=path, document={"category": category, "count": len(amounts), "total": sum(amounts)})
            return final("done")
        if family == "repair_function":
            if "inspect_function" not in observations:
                return tool("inspect_function")
            if "patch_function" not in observations:
                fields = dict(re.findall(r"(operator|offset|lower|upper)=([\w-]+)", prompt))
                return tool("patch_function", **{k: v if k == "operator" else int(v) for k, v in fields.items()})
            if "run_tests" not in observations:
                return tool("run_tests")
            return final("done")
        raise ValueError("El fixture no implementa esa familia.")
