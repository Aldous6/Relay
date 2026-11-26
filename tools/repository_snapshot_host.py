"""Local OpenHands file tools and isolated offline checks for prepared repository cases."""
from __future__ import annotations

import argparse
import getpass
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile
from typing import Literal

ROOT = Path(__file__).resolve().parents[1]
THIS_FILE = Path(__file__).resolve()
sys.path.insert(0, str(ROOT / "src"))
from relay_router.cascade import _manifest
from relay_router.cascade_verifier import CheckCommand, CommandVerifier

PUBLIC = "relay_public_check.py"
IMAGE_RE = re.compile(r"[a-z0-9][a-z0-9./:_-]*@sha256:[0-9a-f]{64}")
_REGISTERED = False
PRIVATE_GRADER = THIS_FILE.with_name("repository_public_checks.py")
OFFICIAL_PRIVATE = ROOT / "data/imported/relay-repository-acceptance-v1"
PUBLIC_CASES = ROOT / "data/imported/relay-repository-public-v1"
IMAGE_MANIFEST = ROOT / "data/imported/relay-repository-v1/image-manifest.json"
_GROQ_CREDENTIAL = None


def prepare_acquisition():
    """Check local presence/shape, without validating an account over the network."""
    global _GROQ_CREDENTIAL
    if _GROQ_CREDENTIAL is not None:
        return
    value = os.environ.get("GROQ_API_KEY")
    if not value and sys.stdin.isatty():
        value = getpass.getpass("Clave Groq para esta adquisición (entrada oculta): ")
    if not value or not value.isascii() or not value.isprintable() or any(c.isspace() for c in value):
        raise ValueError("Acquisition requires GROQ_API_KEY or an interactive credential")
    _GROQ_CREDENTIAL = value

# Only this constant runs in the image. Task text and model output never enter argv.
CONTAINER_CHECK = """
import json, os, pathlib, shutil, signal, subprocess, sys
base, revision, editable_json, mode = sys.argv[1:]
signal.alarm(9000 if mode == 'official-inputs' else 240)
editable = json.loads(editable_json)
source, target = pathlib.Path('/relay/input'), pathlib.Path('/tmp/work')
changed = []
if base:
    root = pathlib.Path(base)
    if not (root / '.git').exists():
        roots = [p for p in root.iterdir() if p.is_dir() and not p.is_symlink() and (p / '.git').exists()]
        if len(roots) != 1:
            sys.exit(2)
        base = str(roots[0])
    head = subprocess.check_output(['git', '-c', 'safe.directory=' + base, '-C', base,
                                    'rev-parse', '--verify', 'HEAD'], text=True).strip()
    if head != revision:
        sys.exit(2)
    shutil.copytree(base, target, symlinks=True, ignore=shutil.ignore_patterns('.git'))
else:
    target.mkdir()
for path in source.rglob('*'):
    if path.is_file() and path.relative_to(source).as_posix() != 'relay_public_check.py':
        dest = target / path.relative_to(source)
        if not dest.resolve().is_relative_to(target) or dest.is_symlink():
            sys.exit(2)
        relative = path.relative_to(source).as_posix()
        if base and relative in editable and (not dest.is_file() or path.read_bytes() != dest.read_bytes()):
            changed.append(relative)
        dest.parent.mkdir(exist_ok=True, parents=True)
        shutil.copyfile(path, dest)
os.chdir(target)
os.environ['HOME'] = '/tmp'
os.environ['RELAY_GRADER_SOURCE_FILES'] = editable_json
os.environ['RELAY_GRADER_CHANGED_FILES'] = json.dumps(changed)
cmd = [sys.executable, '-I', '-B', '/relay/check.py']
if mode == 'official-inputs':
    cmd.append('/relay/private.json')
code = subprocess.call(cmd)
sys.exit(code if code in (0, 10) else 2)
"""


def _case(task):
    root = os.environ.get("RELAY_REPOSITORY_CASES") or PUBLIC_CASES
    case_id = task["task_id"]
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,80}", case_id):
        raise ValueError("Invalid case ID")
    folder = Path(root).resolve(strict=True) / case_id
    config = json.loads((folder / "case.json").read_text(encoding="utf-8"))
    if (config["task_id"], config["source"], config["revision"]) != (case_id, task["source"], task["revision"]):
        raise ValueError("Case identity or repository revision differs from cohort")
    if not IMAGE_RE.fullmatch(config["image"]):
        raise ValueError("Require a local Linux image pinned by repository digest")
    base = config.get("base_path", "")
    if base and (not re.fullmatch(r"/[a-zA-Z0-9_./-]+", base) or ".." in PurePosixPath(base).parts
                 or base == "/" or not re.fullmatch(r"[a-f0-9]{40}", config["revision"])):
        raise ValueError("An image overlay requires a fixed absolute checkout and full commit")
    return folder, config


def _public(task, inspect_image):
    folder, config = _case(task)
    snapshot = folder / "snapshot"
    paths = _manifest(snapshot)
    editable = config["editable_paths"]
    if (not isinstance(editable, list) or not editable or len(set(editable)) != len(editable)
            or not set(editable) <= paths.keys() or PUBLIC in editable or PUBLIC not in paths):
        raise ValueError("Declare editable source files and a protected relay_public_check.py")
    if any(".git" in PurePosixPath(p).parts for p in paths):
        raise ValueError("Snapshot must not contain Git history or configuration")
    if inspect_image:
        inspection = subprocess.run([*_docker(), "image", "inspect", "--format", "{{.Os}}", config["image"]],
                                    stdin=subprocess.DEVNULL, capture_output=True, timeout=10, check=True)
        if inspection.stdout.strip() != b"linux":
            raise ValueError("The pinned image must be locally available for Linux")
    command = CheckCommand("public", (sys.executable, "-I", "-B", str(THIS_FILE), "check",
                                      "--image", config["image"], "--revision", config["revision"],
                                      "--base-path", config.get("base_path", ""),
                                      "--editable-json", json.dumps(editable, separators=(",", ":"))),
                           timeout_seconds=300, failure_returncodes=(10,))
    verifier = CommandVerifier.capture(snapshot, checks=[command], protected_files=sorted(paths.keys() - set(editable)))
    return snapshot, verifier, folder, config


def prepare_public(task):
    snapshot, verifier, _, _ = _public(task, inspect_image=True)
    return snapshot, verifier


def _official_entry(task, config):
    manifest = json.loads((OFFICIAL_PRIVATE / "manifest.json").read_bytes())
    entries = [r for r in manifest["cases"] if r["task_id"] == task["task_id"]]
    images = json.loads(IMAGE_MANIFEST.read_bytes())
    pinned = [r for r in images["tasks"] if r["task_id"] == task["task_id"]]
    if (config["official_instance_id"] != task["task_id"] or len(entries) != 1
            or entries[0]["base_commit"] != task["revision"] or entries[0]["repository"] != task["group_id"]
            or config["image"].split("@")[0].removeprefix("docker.io/") != entries[0]["docker_image"]
            or len(pinned) != 1 or pinned[0]["repository_revision"] != task["revision"]
            or pinned[0]["image"] != config["image"] or images["cohort_sha256"] != manifest["cohort_sha256"]):
        raise ValueError("Official case, commit or image does not match the public work order")
    return entries[0]


def _prepare(task, inspect_image):
    snapshot, verifier, folder, config = _public(task, inspect_image)
    pins = {"case": folder / "case.json", "host": THIS_FILE}
    if "official_instance_id" in config:
        _official_entry(task, config)
        pins.update(private_check=PRIVATE_GRADER, private_records=OFFICIAL_PRIVATE / "official_cases.jsonl.gz",
                    private_manifest=OFFICIAL_PRIVATE / "manifest.json", image_manifest=IMAGE_MANIFEST)
    else:
        private = (folder / "private_check.py").resolve(strict=True)
        if not private.is_file() or private.is_relative_to(snapshot.resolve()):
            raise ValueError("Private checker must be a separate trusted file")
        pins["private_check"] = private
    return snapshot, verifier, pins


def prepare(task):
    return _prepare(task, inspect_image=True)


def _docker():
    # A repository file named docker.exe must never become the host executable.
    binary = "docker.exe" if os.name == "nt" else "docker"
    candidates = (Path(directory) / binary for directory in os.get_exec_path()
                  if Path(directory).is_absolute() and Path(directory).resolve() != Path.cwd().resolve())
    exe = next((str(path.resolve()) for path in candidates if path.is_file() and os.access(path, os.X_OK)), None)
    if exe is None:
        raise FileNotFoundError("Docker CLI unavailable")
    # Ignore remote Docker contexts and DOCKER_HOST.
    endpoint = "npipe:////./pipe/docker_engine" if os.name == "nt" else "unix:///var/run/docker.sock"
    return [exe, "--host", endpoint]


def _mount(source, target):
    value = str(source.resolve(strict=True))
    if any(c in value for c in ',\r\n"'):
        raise ValueError("Unsupported bind-mount path")
    return f"type=bind,src={value},dst={target},readonly"


def _isolated_check(workspace, checker, image, revision, base, editable, private_record=None):
    try:
        docker = _docker()
    except FileNotFoundError:
        return 2
    args = [*docker, "run", "--rm", "--init", "--pull=never", "--network=none", "--read-only",
            "--cap-drop=ALL", "--security-opt=no-new-privileges", "--user=65534:65534",
            "--pids-limit=128", "--memory=2g", "--cpus=2", "--tmpfs=/tmp:rw,nosuid,nodev,size=768m,mode=1777",
            "--log-driver=none", "--mount", _mount(workspace, "/relay/input"),
            "--mount", _mount(checker, "/relay/check.py")]
    if private_record is not None:
        args.extend(["--mount", _mount(private_record, "/relay/private.json")])
    args.extend(["--entrypoint", "python", image, "-I", "-B", "-c", CONTAINER_CHECK,
                 base, revision, json.dumps(editable, separators=(",", ":")),
                 "official-inputs" if private_record is not None else "standard"])
    try:
        code = subprocess.run(args, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, timeout=9030 if private_record else 270, check=False).returncode
    except (subprocess.TimeoutExpired, OSError):
        return 2
    return code if code in (0, 10) else 2


def private_verdict(task, selected_workspace):
    folder, config = _case(task)
    args = (config["image"], config["revision"], config.get("base_path", ""), config["editable_paths"])
    if "official_instance_id" in config:
        entry = _official_entry(task, config)
        with gzip.open(OFFICIAL_PRIVATE / "official_cases.jsonl.gz", "rb") as records:
            matching = [line for line in records if hashlib.sha256(line.rstrip(b"\n")).hexdigest() == entry["row_sha256"]]
        if len(matching) != 1:
            raise ValueError("Sealed private case record is missing or changed")
        source = json.loads(matching[0])
        if source["instance_id"] != task["task_id"] or source["base_commit"] != task["revision"]:
            raise ValueError("Wrong private instance")
        with tempfile.TemporaryDirectory() as temp:
            record = Path(temp) / "input.json"
            fields = ("test_patch", "FAIL_TO_PASS", "PASS_TO_PASS", "log_parser")
            record.write_text(json.dumps({key: source[key] for key in fields}), encoding="utf-8")
            code = _isolated_check(Path(selected_workspace), PRIVATE_GRADER, *args, private_record=record)
    else:
        code = _isolated_check(Path(selected_workspace), folder / "private_check.py", *args)
    return {0: "passed", 10: "failed", 2: "inconclusive"}[code]


def _file_operation(workspace, editable, action):
    relative = action.path
    paths = _manifest(workspace)
    if action.operation == "list":
        matches = sorted(p for p in paths if p.startswith(relative))
        return "\n".join(matches[:80])[:5000] or "(no files)"
    if relative not in paths:
        raise ValueError("Unknown public file")
    path = workspace / relative
    if path.stat().st_size > 512_000:
        raise ValueError("File exceeds the text editor limit")
    original = path.read_bytes().decode("utf-8")
    if action.operation == "read":
        lines = original.splitlines(keepends=True)
        if not 1 <= action.line <= max(1, len(lines)):
            raise ValueError("Invalid starting line")
        return "".join(lines[action.line - 1: action.line + 79])[:5000]
    if relative not in editable or not action.find or original.count(action.find) != 1:
        raise ValueError("Replace a unique nonempty string in an editable file")
    changed = original.replace(action.find, action.replacement, 1).encode("utf-8")
    if len(changed) > 512_000:
        raise ValueError("Edit exceeds the text editor limit")
    path.write_bytes(changed)
    return "Saved " + relative


def _file_tool(workspace, editable):
    from openhands.sdk.tool import Action, Observation, Tool, ToolDefinition, ToolExecutor, register_tool
    global _REGISTERED
    if not _REGISTERED:
        class FileAction(Action):
            operation: Literal["list", "read", "replace"]
            path: str = ""
            line: int = 1
            find: str = ""
            replacement: str = ""

        class FileObservation(Observation):
            pass

        class Executor(ToolExecutor):
            def __init__(self, root, allowed):
                self.root, self.allowed = Path(root), allowed

            def __call__(self, action, conversation=None):
                try:
                    return FileObservation.from_text(_file_operation(self.root, self.allowed, action))
                except (ValueError, UnicodeError, OSError) as exc:
                    return FileObservation.from_text(type(exc).__name__ + ": invalid file operation", is_error=True)

        class RepositoryFiles(ToolDefinition[FileAction, FileObservation]):
            @classmethod
            def create(cls, conv_state=None, **params):
                return [cls(action_type=FileAction, observation_type=FileObservation,
                            description="List public paths by prefix, read from a 1-based line, or replace one exact string in an editable UTF-8 file.",
                            executor=Executor(params["root"], params["editable"]))]

        register_tool("RelayRepositoryFiles", RepositoryFiles)
        _REGISTERED = True
    return Tool(name="RelayRepositoryFiles", params={"root": str(workspace), "editable": list(editable)})


def build_agent(task, context):
    from openhands.sdk import Agent, LLM
    from relay_router.integrations.groq_recovery import BASE_URL, _MODEL_RATES
    _, config = _case(task)
    cost_in, cost_out = _MODEL_RATES[context.model]
    prepare_acquisition()
    llm = LLM(model=context.model, api_key=_GROQ_CREDENTIAL, base_url=BASE_URL,
              num_retries=0, api_mode="chat", stream=False, timeout=90, temperature=0,
              max_output_tokens=1024, reasoning_effort="low", log_completions=False, caching_prompt=False,
              capability_overrides={"supports_prompt_cache_key": False, "supports_prompt_cache_retention": False},
              input_cost_per_token=cost_in, output_cost_per_token=cost_out)
    return Agent(llm=llm, tools=[_file_tool(context.workspace, config["editable_paths"])],
                 include_default_tools=["FinishTool"], condenser=None, critic=None, mcp_config={},
                 system_prompt="Resolve the issue using the repository file tool. Read the relevant public code, "
                               "apply changes, then finish. Editable paths: " + ", ".join(config["editable_paths"]))


def preflight(cohort, *, inspect_images=False):
    rows = []
    for task in cohort["tasks"]:
        try:
            _prepare(task, inspect_image=inspect_images)
            rows.append({"task_id": task["task_id"], "prepared": True})
        except (KeyError, OSError, ValueError, subprocess.SubprocessError) as exc:
            rows.append({"task_id": task["task_id"], "prepared": False, "reason": str(exc)})
    return {"complete": bool(rows) and all(row["prepared"] for row in rows),
            "local_images_inspected": inspect_images, "cases": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check")
    check.add_argument("--image", required=True)
    check.add_argument("--revision", required=True)
    check.add_argument("--base-path", default="")
    check.add_argument("--editable-json", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--cohort", required=True, type=Path)
    pre.add_argument("--inspect-images", action="store_true")
    pre.add_argument("--official-acceptance", type=Path, help="Also verify the separate private SWE-Live dossier")
    args = parser.parse_args()
    if args.command == "preflight":
        result = preflight(json.loads(args.cohort.read_text(encoding="utf-8")), inspect_images=args.inspect_images)
        if args.official_acceptance:
            sys.path.insert(0, str(THIS_FILE.parent))
            from prepare_repository_acceptance import verify
            inputs = verify(args.cohort, args.official_acceptance)
            result["official_acceptance_inputs"] = {"verified_cases": len(inputs["cases"]),
                                                    "jsonl_sha256": inputs["jsonl_sha256"], "graded": False}
        print(json.dumps(result, indent=2))
        return 0 if result["complete"] else 2
    if not IMAGE_RE.fullmatch(args.image):
        return 2
    workspace = Path.cwd()
    return _isolated_check(workspace, workspace / PUBLIC, args.image, args.revision, args.base_path,
                           json.loads(args.editable_json))


if __name__ == "__main__":
    raise SystemExit(main())
