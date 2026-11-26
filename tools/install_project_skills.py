"""Install only selected, pinned skills using the official local installer.

Requires network access and write permission for .agents/skills. Does not execute
any downloaded script or install Python dependencies. Existing skills are never
overwritten. The official installer remains responsible for archive validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installer", type=Path, default=Path.home() / ".codex/skills/.system/skill-installer/scripts/install-skill-from-github.py")
    args = parser.parse_args()
    if not args.installer.is_file():
        raise FileNotFoundError("No se encontró el instalador oficial. Proporciona --installer.")
    lock = json.loads((ROOT / "configs/skills-lock.json").read_text(encoding="utf-8"))
    destination = ROOT / ".agents/skills"
    licenses = ROOT / "references/skill-licenses"
    licenses.mkdir(parents=True, exist_ok=True)
    installed = []
    for repo in lock["repositories"]:
        revision = repo["revision"]
        if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
            raise ValueError("Se requiere un commit SHA-1 completo.")
        # Multiple paths use a single download. Never execute scripts from it.
        subprocess.run([sys.executable, str(args.installer), "--repo", repo["repository"],
                        "--ref", revision, "--path", *repo["skills"], "--dest", str(destination),
                        "--method", "download"], check=True)
        license_url = f"https://raw.githubusercontent.com/{repo['repository']}/{revision}/{repo['license_path']}"
        license_target = licenses / (repo["repository"].replace("/", "--") + ".txt")
        with urlopen(Request(license_url, headers={"User-Agent": "routerbench-icd-skill-provenance"}), timeout=60) as response:
            license_target.write_bytes(response.read())
        for source in repo["skills"]:
            name = source.rsplit("/", 1)[1]
            folder = destination / name
            skill_text = (folder / "SKILL.md").read_text(encoding="utf-8")
            if not skill_text.startswith("---"):
                raise ValueError(f"Skill sin frontmatter: {name}")
            files = sorted(path for path in folder.rglob("*") if path.is_file())
            installed.append({"name": name, "repository": repo["repository"], "revision": revision,
                              "source_path": source, "local_path": folder.relative_to(ROOT).as_posix(),
                              "license_source": license_url,
                              "files_sha256": {path.relative_to(folder).as_posix(): digest(path) for path in files},
                              "modified": False})
    report = {"installed_at_utc": datetime.now(timezone.utc).isoformat(), "scope": "project",
              "dependencies_installed": False, "downloaded_code_executed": False, "skills": installed}
    (ROOT / "reports/skills-installed.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Instaladas y registradas: " + ", ".join(skill["name"] for skill in installed))


if __name__ == "__main__":
    main()
