"""Pin public task sources before running any models; never touch data/raw."""
import argparse
import ast
import gzip
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
QUIX = "4257f44b0ff1181dedaedee6a447e133219fcebf"
HUMAN = "b9453e8f1865df0a0057a1383767caa400acfbcf"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def download(url, path):
    if path.exists():
        return path.read_bytes()
    request = urllib.request.Request(url, headers={"User-Agent": "Relay-public-pilot/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        content = response.read(8 * 1024 * 1024 + 1)
    if len(content) > 8 * 1024 * 1024:
        raise ValueError("Source exceeds expected size")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return content


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/imported/harness-public-v1")
    args = parser.parse_args()
    folder = args.output
    folder.mkdir(parents=True, exist_ok=True)
    sources = [
        (f"https://codeload.github.com/jkoppel/QuixBugs/zip/{QUIX}", "quixbugs.zip"),
        (f"https://raw.githubusercontent.com/openai/human-eval/{HUMAN}/data/HumanEval.jsonl.gz", "HumanEval.jsonl.gz"),
        (f"https://raw.githubusercontent.com/openai/human-eval/{HUMAN}/LICENSE", "HumanEval-LICENSE"),
    ]
    provenance = []
    for url, name in sources:
        content = download(url, folder / name)
        provenance.append({"url": url, "file": name, "sha256": digest(content), "bytes": len(content)})
    archive = zipfile.ZipFile(io.BytesIO((folder / "quixbugs.zip").read_bytes()))
    for member in archive.namelist():
        parts = member.split("/", 1)
        if len(parts) < 2:
            continue
        relative = parts[1]
        if (relative.startswith(("python_programs/", "correct_python_programs/", "json_testcases/", "python_testcases/"))
                or relative in {"LICENSE", "README.md", "legal_notes.txt"}):
            target = folder / "quixbugs" / relative
            if relative.endswith("/"):
                continue
            if not target.resolve().is_relative_to((folder / "quixbugs").resolve()):
                raise ValueError("Unsafe archive member")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(member))
    humaneval = [json.loads(line) for line in gzip.decompress((folder / "HumanEval.jsonl.gz").read_bytes()).decode().splitlines()]
    (folder / "humaneval.json").write_text(json.dumps(humaneval, indent=2), encoding="utf-8")
    manifest = {"kind": "public_sources_before_model_runs", "sources": provenance,
                "quixbugs_revision": QUIX, "humaneval_revision": HUMAN,
                "licenses": "MIT texts retained with respective sources",
                "limitation": "Public historical benchmarks; training contamination is possible; not customer production traffic."}
    (folder / "sources.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"directory": str(folder), "humaneval_tasks": len(humaneval),
                      "quixbugs_json_files": len(list((folder / "quixbugs/json_testcases").glob("*.json")))}))


if __name__ == "__main__":
    main()
