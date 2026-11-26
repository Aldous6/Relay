"""Explicit two-turn local integration check using a synthetic document."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    from routerbench.demo import RelayDemoService

    output = ROOT / "reports/live_documents_smoke.json"
    if output.exists():
        raise FileExistsError("Se conserva la evidencia existente; no se repite generación automáticamente.")
    service = RelayDemoService(ROOT)
    document = "El proyecto se llama Bruma. El presupuesto aprobado es de 17 millones de pesos."
    first_prompt = "Según el documento, ¿cómo se llama el proyecto? Responde solo con el nombre."
    first = service.respond(first_prompt, [], document)
    history = [{"role": "user", "content": first_prompt,
                "model_content": "DOCUMENTO DE REFERENCIA:\n"+document+"\n\nCONSULTA:\n"+first_prompt},
               {"role": "assistant", "content": first["answer"]}]
    second_prompt = "¿Cuántos millones de pesos tiene de presupuesto? Responde solo con el número."
    second = service.respond(second_prompt, history)
    record = {"executed_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "two synthetic turns verify document/history integration; not a live quality benchmark",
              "paid_api_calls": False, "document": document,
              "first_prompt": first_prompt, "first_result": first,
              "second_prompt": second_prompt, "second_result": second,
              "second_turn_document_reattached": False,
              "expected_values_present": {"project": "bruma" in first["answer"].casefold(),
                                          "budget": "17" in second["answer"]}}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"evidence": str(output), "answers": [first["answer"], second["answer"]],
                      "expected_values_present": record["expected_values_present"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
