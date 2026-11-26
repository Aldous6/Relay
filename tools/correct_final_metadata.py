"""Correct an inherited development label without recomputing final metrics."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    directory = ROOT / "reports/final"
    path = directory / "summary.json"
    audit = directory / "scope_metadata_correction.json"
    if audit.exists():
        print("La corrección de metadatos ya está registrada; no se modifica nada.")
        return
    original = path.read_bytes()
    before = json.loads(original)
    after = copy.deepcopy(before)
    changed = []
    old = "paired prompt bootstrap on exploratory development; historical selected-call cost only; no router overhead or confirmatory superiority claim"
    new = "held-out paired prompt bootstrap; primary or secondary status is declared by policy role; historical selected-call cost excludes router compute and retries"
    for name, policy in after["policies"].items():
        comparison = policy["comparison_vs_fixed"]
        if comparison["scope"] != old:
            raise ValueError("Etiqueta inesperada; no se aplica una corrección genérica.")
        comparison["scope"] = new
        changed.append(f"policies.{name}.comparison_vs_fixed.scope")
    # Every field except the four textual scope labels must remain identical.
    check = copy.deepcopy(after)
    for name in check["policies"]:
        check["policies"][name]["comparison_vs_fixed"]["scope"] = old
    assert check == before
    archive = directory / "summary.before_scope_metadata_correction.json"
    with archive.open("xb") as stream:
        stream.write(original)
    updated = (json.dumps(after, ensure_ascii=False, indent=2, allow_nan=False)+"\n").encode("utf-8")
    temporary = path.with_suffix(".json.part")
    temporary.write_bytes(updated)
    temporary.replace(path)
    record = {"reason": "A shared bootstrap helper inherited a development-only description in an already held-out evaluation.",
              "changed_fields": changed, "before_sha256": hashlib.sha256(original).hexdigest(),
              "after_sha256": hashlib.sha256(updated).hexdigest(),
              "original_archive": archive.name, "metrics_predictions_routes_or_selection_changed": False,
              "before": old, "after": new}
    with audit.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print("Corregidas únicamente cuatro etiquetas de alcance; métricas intactas.")


if __name__ == "__main__":
    main()
