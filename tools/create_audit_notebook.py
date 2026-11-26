"""Create a small audit notebook from the inspected OpenAI skill template."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cell(kind, source):
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def main():
    notebook_path = ROOT / "notebooks/01_auditoria.ipynb"
    helper = ROOT / ".agents/skills/jupyter-notebook/scripts/new_notebook.py"
    subprocess.run([sys.executable, str(helper), "--kind", "experiment", "--title", "Auditoría RouterBench 0-shot",
                    "--out", str(notebook_path)], check=True)
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    notebook["cells"] = [
        cell("markdown", "# Auditoría de RouterBench 0-shot\n\nRevisar los artefactos iniciales y su trazabilidad. No entrenar ni evaluar modelos. Las cifras de la propuesta son antecedentes por contrastar.\n"),
        cell("code", "from pathlib import Path\nimport json\nimport sys\nimport pandas as pd\n\nroot = next(path for path in (Path.cwd(), *Path.cwd().parents) if (path / 'configs/experiment.json').is_file())\nsys.path.insert(0, str(root / 'src'))\nfrom routerbench.data import sha256_file\nfrom routerbench.features import FEATURE_NAMES\n\nconfig = json.loads((root / 'configs/experiment.json').read_text(encoding='utf-8'))\nprint(config['experiment_id'])\n"),
        cell("markdown", "La comprobación se limita a integridad, soporte y separación de prompts. No usar estos datos para escoger hiperparámetros a partir de resultados del test.\n"),
        cell("code", "audit = json.loads((root / 'reports/audit.json').read_text(encoding='utf-8'))\nassert sha256_file(root / config['dataset']['path']) == config['dataset']['sha256']\nprint({key: audit[key] for key in ['rows', 'columns', 'models_count', 'evaluations', 'duplicate_extra_rows']})\nprint(audit['thresholds']['0.75'])\n"),
        cell("markdown", "La deduplicación conserva la primera fila original sin consultar scores. Los no elegibles permanecen para la simulación futura de la política completa.\n"),
        cell("code", "manifest = json.loads((root / 'data/processed/manifest.json').read_text(encoding='utf-8'))\nfeatures = pd.read_parquet(root / 'data/processed/features.parquet')\nrows = pd.read_parquet(root / 'data/processed/rows.parquet')\nassert tuple(features.columns) == FEATURE_NAMES\nassert len(features) == len(rows) == manifest['deduplicated_rows']\nassert not features.isna().any().any()\nassert rows.groupby('prompt_group')['split'].nunique().max() == 1\nfor name, checksum in manifest['files_sha256'].items():\n    assert sha256_file(root / 'data/processed' / name) == checksum\nassert manifest['final_test_evaluated'] is False\nprint(features.shape)\nprint(manifest['counts'])\n"),
        cell("markdown", "La auditoría nueva obtiene 34,977 elegibles tras deduplicar; la propuesta dice 34,976. Once grupos duplicados tienen resultados distintos. La diferencia se conserva y se explica en `reports/auditoria_inicial.md`.\n\nSiguiente paso: comparar modelos dentro del train y congelar decisiones antes de evaluar desempeño en test. Consultar `docs/protocolo.md`.\n"),
    ]
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
