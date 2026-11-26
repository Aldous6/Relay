"""Pinned, read-only BFCL acquisition and deterministic base-conversation splits.

The importer downloads bytes only with explicit authorization. It never imports
upstream code, runs benchmark prompts, evaluates answers, or loads reserved case
payloads into the returned objects. Raw upstream files are retained unchanged.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen

REPOSITORY = "ShishirPatil/gorilla"
COMMIT = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
PREFIX = "berkeley-function-call-leaderboard/bfcl_eval/"
QUESTION_PATH = PREFIX + "data/BFCL_v4_multi_turn_base.json"
REFERENCE_PATH = PREFIX + "data/possible_answer/BFCL_v4_multi_turn_base.json"
DEFAULT_SEED = 20261113
DEFAULT_DEVELOPMENT_GROUPS = 24
SPLIT_COUNTS = {"fit": 96, "calibration": 40, "development": 24, "evaluation": 40}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SNAPSHOT = PROJECT_ROOT / "data" / "imported" / "bfcl" / COMMIT

# Metadata obtained from GitHub's Git Trees API at the pinned commit, not from
# task outcomes. Git blob SHA-1 validates exact bytes including their newlines.
PINNED_FILES = (
    ("berkeley-function-call-leaderboard/bfcl_eval/__init__.py", "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391", 0),
    ("berkeley-function-call-leaderboard/bfcl_eval/constants/__init__.py", "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391", 0),
    ("berkeley-function-call-leaderboard/bfcl_eval/constants/category_mapping.py", "492142b986438fe8a5a6c38931e567bcd45c0d9b", 2049),
    ("berkeley-function-call-leaderboard/bfcl_eval/constants/default_prompts.py", "1f5e9e1e8119b2277bee2198214ee9d6aaf6f6e2", 9118),
    ("berkeley-function-call-leaderboard/bfcl_eval/constants/enums.py", "7400315f045419d4a8a28525f3999c1844f5e9af", 896),
    ("berkeley-function-call-leaderboard/bfcl_eval/constants/eval_config.py", "813d6de70333b6137f5fb62e116956b13bb339d3", 1648),
    ("berkeley-function-call-leaderboard/bfcl_eval/constants/executable_backend_config.py", "03a8a43fd6b75dd0961504edc6704a728b74db5b", 1939),
    ("berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_multi_turn_base.json", "656fbd5e56969373353cd58b880234a6cc4edc99", 390425),
    ("berkeley-function-call-leaderboard/bfcl_eval/data/multi_turn_func_doc/gorilla_file_system.json", "b9a6d2d314febe8ccf201887ef87cfe081ffbf4e", 13953),
    ("berkeley-function-call-leaderboard/bfcl_eval/data/multi_turn_func_doc/math_api.json", "52a911035a2b71390027e02af70c1abb904c71b3", 9158),
    ("berkeley-function-call-leaderboard/bfcl_eval/data/multi_turn_func_doc/message_api.json", "a178734ad47f269571c877e6e433b26171516024", 6027),
    ("berkeley-function-call-leaderboard/bfcl_eval/data/multi_turn_func_doc/posting_api.json", "2a8737a376640ba304c313d23d63049927f823eb", 10573),
    ("berkeley-function-call-leaderboard/bfcl_eval/data/multi_turn_func_doc/ticket_api.json", "f8281657b64c4b6410a8590cc82b9584bc39729f", 6648),
    ("berkeley-function-call-leaderboard/bfcl_eval/data/multi_turn_func_doc/trading_bot.json", "7dce37326156f65efc55df30db26064d68e8dc8c", 12608),
    ("berkeley-function-call-leaderboard/bfcl_eval/data/multi_turn_func_doc/travel_booking.json", "95008945fa5a9bc104a77647e4990d1e36ce7ec5", 16979),
    ("berkeley-function-call-leaderboard/bfcl_eval/data/multi_turn_func_doc/vehicle_control.json", "7139dfc3bf2646be8f71ac383760f28899b17f85", 17545),
    ("berkeley-function-call-leaderboard/bfcl_eval/data/possible_answer/BFCL_v4_multi_turn_base.json", "3ee85fcd5e8cccca24e2347e464b58fd1b9f02a1", 78931),
    ("berkeley-function-call-leaderboard/bfcl_eval/eval_checker/__init__.py", "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391", 0),
    ("berkeley-function-call-leaderboard/bfcl_eval/eval_checker/multi_turn_eval/__init__.py", "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391", 0),
    ("berkeley-function-call-leaderboard/bfcl_eval/eval_checker/multi_turn_eval/func_source_code/__init__.py", "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391", 0),
    ("berkeley-function-call-leaderboard/bfcl_eval/eval_checker/multi_turn_eval/func_source_code/gorilla_file_system.py", "6d269dd5fcf809d514eb531263f1e0e14246993c", 32759),
    ("berkeley-function-call-leaderboard/bfcl_eval/eval_checker/multi_turn_eval/func_source_code/long_context.py", "938105bea5a07e46ee15615ab11d1c0988124b5d", 190330),
    ("berkeley-function-call-leaderboard/bfcl_eval/eval_checker/multi_turn_eval/func_source_code/math_api.py", "e1b43b3817a3d8cbedc011bb36d4d9253fd6a05a", 12252),
    ("berkeley-function-call-leaderboard/bfcl_eval/eval_checker/multi_turn_eval/func_source_code/message_api.py", "3fb461dfbf60ecb9b6cc832162c97368d69ec1ae", 12534),
    ("berkeley-function-call-leaderboard/bfcl_eval/eval_checker/multi_turn_eval/func_source_code/posting_api.py", "ccd7566181bc36889d235c2ad62a064a6e16ba00", 12537),
    ("berkeley-function-call-leaderboard/bfcl_eval/eval_checker/multi_turn_eval/func_source_code/ticket_api.py", "6ce72a1cd23a11d814841b46a39d173ccc4e47bb", 9677),
    ("berkeley-function-call-leaderboard/bfcl_eval/eval_checker/multi_turn_eval/func_source_code/trading_bot.py", "e745788b15d729194ef39844f5eb1a3a67a4d55a", 23795),
    ("berkeley-function-call-leaderboard/bfcl_eval/eval_checker/multi_turn_eval/func_source_code/travel_booking.py", "f357ccdabdedc5fed0019392101a7de1f524ef77", 36647),
    ("berkeley-function-call-leaderboard/bfcl_eval/eval_checker/multi_turn_eval/func_source_code/vehicle_control.py", "ab7ef900d1a4d0120638b79c95441710eb8f5282", 29559),
    ("berkeley-function-call-leaderboard/bfcl_eval/eval_checker/multi_turn_eval/multi_turn_checker.py", "34b36715cdfaa4464ba1be137af4d30e85da8efb", 13592),
    ("berkeley-function-call-leaderboard/bfcl_eval/eval_checker/multi_turn_eval/multi_turn_utils.py", "dfcf6512a2eeb592820f3a0fa0fb564a117facc4", 5234),
    ("berkeley-function-call-leaderboard/bfcl_eval/model_handler/base_handler.py", "a1025e9cbf5d86bcdc62cd0564006284eaffb00a", 43697),
    ("berkeley-function-call-leaderboard/bfcl_eval/model_handler/utils.py", "7abd9a242f27c72399dbdbb0f2749b0464e1a0b8", 33694),
    ("berkeley-function-call-leaderboard/bfcl_eval/utils.py", "552a10f36ec10840773f75e50fed01ca19b4bfc6", 36012),
    ("berkeley-function-call-leaderboard/CONTRIBUTING.md", "12607f801f449e1e2135558deaada90fe2d49ba0", 9293),
    ("berkeley-function-call-leaderboard/pyproject.toml", "56f78da1277e4e76d36a3e5c6188437bc363e193", 2013),
    ("berkeley-function-call-leaderboard/README.md", "39fc8a708a2c4cd29eeb37f946f7612cab9e2353", 17311),
    ("berkeley-function-call-leaderboard/TEST_CATEGORIES.md", "2da513fdc9b1d28bcabd3e00d22d234204fbebcb", 3551),
    ("LICENSE", "261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64", 11357),
)
_ID = re.compile(rb'^\s*\{\s*"id"\s*:\s*"(multi_turn_base_[0-9]+)"\s*[,}]')
_CASE_ID = re.compile(r"multi_turn_base_[0-9]+\Z")


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2) + "\n").encode("utf-8")


def _read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate metadata JSON key.")
            result[key] = value
        return result
    def invalid(_):
        raise ValueError("Nonfinite metadata JSON value.")
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=invalid)


def blob_sha1(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()


def _verify_bytes(raw, expected_sha, expected_size):
    if len(raw) != expected_size or blob_sha1(raw) != expected_sha:
        raise ValueError("Upstream content does not match the pinned Git blob.")


def _safe_root(directory):
    root = Path(directory).resolve()
    if root == PROJECT_ROOT or root.is_relative_to((PROJECT_ROOT / "data" / "raw").resolve()):
        raise ValueError("Use a dedicated imported snapshot directory outside data/raw.")
    return root


def _write_new(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)


def _raw_url(path):
    return f"https://raw.githubusercontent.com/{REPOSITORY}/{COMMIT}/{path}"


def prepare_snapshot(directory=DEFAULT_SNAPSHOT):
    """Pure plan: no network, no task/reference decoding, no filesystem writes."""
    root = _safe_root(directory)
    return {"mode": "prepare", "snapshot": str(root), "repository": REPOSITORY, "commit": COMMIT,
            "category": "multi_turn_base", "version_prefix": "BFCL_v4", "files": len(PINNED_FILES),
            "download_bytes": sum(size for _, _, size in PINNED_FILES), "provider_calls": 0,
            "network_calls": 0, "partition_groups": dict(SPLIT_COUNTS), "seed": DEFAULT_SEED,
            "reference_access": "selection is sealed from IDs before reference download/decoding",
            "scope": "exploratory import plan; not benchmark execution or independent validation"}


def _fetch_public(url):
    request = Request(url, headers={"User-Agent": "Relay-BFCL-pinned-import/1"})
    with urlopen(request, timeout=40) as response:
        if response.geturl() != url:
            raise ValueError("Unexpected redirect while fetching pinned upstream content.")
        return response.read()


def _id_lines(raw):
    """Extract the leading ID only; do not JSON-decode unselected case contents."""
    seen = set()
    for line in raw.splitlines():
        if not line.strip():
            continue
        match = _ID.match(line)
        if match is None:
            raise ValueError("BFCL records require one JSON object per line with a leading base-case ID.")
        identity = match.group(1).decode("ascii")
        if identity in seen:
            raise ValueError("Duplicate BFCL conversation ID.")
        seen.add(identity)
        yield identity, line


def select_ids(ids, *, seed=DEFAULT_SEED):
    """Outcome-independent assignment; base IDs are never guessed across variants."""
    ids = list(ids)
    if (type(seed) is not int or seed < 0 or len(ids) != sum(SPLIT_COUNTS.values()) or len(ids) != len(set(ids))
            or any(type(value) is not str or _CASE_ID.fullmatch(value) is None for value in ids)):
        raise ValueError("Select exactly 200 unique base IDs with a valid seed for the frozen partition sizes.")
    ordered = sorted(ids, key=lambda value: (hashlib.sha256(f"{COMMIT}:{seed}:{value}".encode()).hexdigest(), value))
    partitions, start = {}, 0
    for name, count in SPLIT_COUNTS.items():
        partitions[name] = ordered[start:start + count]
        start += count
    return {"schema_version": 1, "commit": COMMIT, "category": "multi_turn_base", "seed": seed,
            "selection_rule": "sha256(commit:seed:exact_base_id); no outcomes, prompts, families or lengths",
            **partitions,
            "group_rule": "one exact base conversation; no imported variants or inferred variant equivalence",
            "source_groups": {identity: f"bfcl:{COMMIT}:{identity}" for identity in sorted(ids)},
            "independent_validation": False}


def download_snapshot(directory=DEFAULT_SNAPSHOT, *, fetch=None):
    """Download only pinned public bytes; verify and retain every original file.

    Incomplete imports may resume only when existing bytes match the inventory.
    A completed snapshot is immutable and is validated instead of overwritten.
    """
    root = _safe_root(directory)
    if (root / "manifest.json").exists():
        return read_snapshot(root)
    root.mkdir(parents=True, exist_ok=True)
    fetch = _fetch_public if fetch is None else fetch
    files = []
    # Freeze IDs before fetching references; no private answers drive assignment.
    ordered = sorted(PINNED_FILES, key=lambda item: (item[0] == REFERENCE_PATH, item[0]))
    selection = None
    for path, expected_sha, expected_size in ordered:
        local = root / "upstream" / path
        if local.exists():
            raw = local.read_bytes()
        else:
            raw = fetch(_raw_url(path))
            _verify_bytes(raw, expected_sha, expected_size)
            _write_new(local, raw)
        _verify_bytes(raw, expected_sha, expected_size)
        if path == QUESTION_PATH:
            selection = select_ids(identity for identity, _ in _id_lines(raw))
            frozen = _json_bytes(selection)
            target = root / "selection.json"
            if target.exists():
                if target.read_bytes() != frozen:
                    raise ValueError("Existing selection differs from the deterministic ID assignment.")
            else:
                _write_new(target, frozen)
        if path == REFERENCE_PATH and selection is None:
            raise ValueError("Select conversation groups before reading references.")
        files.append({"path": path, "bytes": expected_size, "git_blob_sha1": expected_sha,
                      "sha256": hashlib.sha256(raw).hexdigest(), "url": _raw_url(path)})
    if selection is None:
        raise ValueError("The pinned inventory lacks the base conversation file.")
    manifest = {"schema_version": 1, "repository": REPOSITORY, "commit": COMMIT,
        "category": "multi_turn_base", "version_prefix": "BFCL_v4", "license": "Apache-2.0",
        "files": sorted(files, key=lambda item: item["path"]),
        "selection_sha256": hashlib.sha256((root / "selection.json").read_bytes()).hexdigest(),
        "scope": "unchanged upstream bytes; local ID split, no model generation or heldout inspection",
        "reserved_payloads_decoded": False, "provider_calls": 0,
        "provenance": "Git Trees API inventory pinned before download; exact Git blob verification"}
    _write_new(root / "manifest.json", _json_bytes(manifest))
    return read_snapshot(root)


def read_snapshot(directory=DEFAULT_SNAPSHOT):
    """Verify exact pinned bytes; return metadata only, never task payloads."""
    root = _safe_root(directory)
    manifest = _read_json(root / "manifest.json")
    if (manifest.get("schema_version") != 1 or manifest.get("commit") != COMMIT
            or manifest.get("repository") != REPOSITORY or manifest.get("category") != "multi_turn_base"
            or type(manifest.get("files")) is not list):
        raise ValueError("Snapshot identity or manifest schema is unsupported.")
    declared = {item.get("path"): item for item in manifest["files"] if type(item) is dict}
    if len(declared) != len(manifest["files"]) or set(declared) != {path for path, _, _ in PINNED_FILES}:
        raise ValueError("Snapshot inventory differs from the pinned upstream subset.")
    for path, expected_sha, expected_size in PINNED_FILES:
        item = declared[path]
        if (item.get("git_blob_sha1") != expected_sha or item.get("bytes") != expected_size
                or item.get("url") != _raw_url(path)):
            raise ValueError("Snapshot file provenance differs from the pinned inventory.")
        raw = (root / "upstream" / path).read_bytes()
        _verify_bytes(raw, expected_sha, expected_size)
        if item.get("sha256") != hashlib.sha256(raw).hexdigest():
            raise ValueError("Snapshot SHA-256 mismatch.")
    selection_raw = (root / "selection.json").read_bytes()
    if manifest.get("selection_sha256") != hashlib.sha256(selection_raw).hexdigest():
        raise ValueError("Snapshot selection hash mismatch.")
    selection = _read_json(root / "selection.json")
    ids = [identity for identity, _ in _id_lines((root / "upstream" / QUESTION_PATH).read_bytes())]
    if selection != select_ids(ids):
        raise ValueError("Snapshot selection is not the frozen outcome-independent assignment.")
    ledger_path = root / "exposure_ledger.json"
    ledger = _read_json(ledger_path) if ledger_path.exists() else None
    if ledger is not None:
        _validate_exposure_ledger(ledger, selection)
    return {"root": str(root), "upstream_root": str(root / "upstream"),
            "package_root": str(root / "upstream" / "berkeley-function-call-leaderboard"),
            "manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
            "manifest": manifest, "selection": selection, "exposure_ledger": ledger,
            "exposure_ledger_sha256": hashlib.sha256(ledger_path.read_bytes()).hexdigest() if ledger is not None else None}


def _exposure_partitions(ids, selection):
    ids = set(ids)
    if ids - set(selection["source_groups"]):
        raise ValueError("Exposure record contains unknown conversation IDs.")
    return {name: sorted(ids & set(selection[name])) for name in SPLIT_COUNTS}


def _validate_exposure_ledger(ledger, selection):
    if (type(ledger) is not dict or ledger.get("schema_version") != 1 or ledger.get("commit") != COMMIT
            or type(ledger.get("exposed_ids")) is not list
            or any(type(value) is not str for value in ledger["exposed_ids"])
            or len(set(ledger["exposed_ids"])) != len(ledger["exposed_ids"])):
        raise ValueError("Invalid BFCL exposure ledger.")
    expected = _exposure_partitions(ledger["exposed_ids"], selection)
    if ledger.get("partition_ids") != expected or ledger.get("blocked_evaluation_ids") != expected["evaluation"]:
        raise ValueError("Exposure ledger must preserve the original partitions and block exposed evaluation IDs.")
    if ledger.get("source_groups") != {identity: selection["source_groups"][identity] for identity in sorted(ledger["exposed_ids"])}:
        raise ValueError("Exposure source groups differ from the frozen assignment.")


def register_exposure_record(directory, incident_path):
    """Retain an explicit incident sidecar without changing the original split.

    The v1 ledger handles one immutable incident. A further distinct incident
    requires a separately reviewed ledger revision; it is never silently merged.
    """
    snapshot = read_snapshot(directory)
    incident_path = Path(incident_path)
    incident_raw = incident_path.read_bytes()
    incident = _read_json(incident_path)
    if (type(incident) is not dict or incident.get("commit") != COMMIT
            or incident.get("kind") != "accidental_source_search_exposure"
            or type(incident.get("conservative_affected_ids")) is not list
            or any(type(value) is not str for value in incident["conservative_affected_ids"])):
        raise ValueError("Expected an explicit source-search incident for this pinned commit.")
    ids = incident["conservative_affected_ids"]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Exposure IDs must be nonempty and unique.")
    partitions = _exposure_partitions(ids, snapshot["selection"])
    provided = incident.get("partition_ids")
    if type(provided) is not dict or set(provided) != set(SPLIT_COUNTS) or any(
            type(provided[name]) is not list or sorted(provided[name]) != partitions[name] for name in SPLIT_COUNTS):
        raise ValueError("Incident partition assignment differs from the frozen ID split.")
    ledger = {"schema_version": 1, "commit": COMMIT, "incident_kind": incident["kind"],
              "source_record_path": str(incident_path.resolve()), "source_record_sha256": hashlib.sha256(incident_raw).hexdigest(),
              "exposed_ids": sorted(ids), "partition_ids": partitions,
              "source_groups": {identity: snapshot["selection"]["source_groups"][identity] for identity in sorted(ids)},
              "blocked_evaluation_ids": partitions["evaluation"], "repartitioned": False,
              "scope": "Public-question/initial-state exposure; private references not inspected; affected groups are not untouched holdout."}
    _validate_exposure_ledger(ledger, snapshot["selection"])
    target = Path(snapshot["root"]) / "exposure_ledger.json"
    if target.exists():
        if _read_json(target) != ledger:
            raise ValueError("Existing exposure ledger is immutable; review a new incident revision explicitly.")
    else:
        _write_new(target, _json_bytes(ledger))
    return ledger


def require_unexposed_evaluation_ids(directory, ids):
    """Metadata-only holdout guard; this function does not load evaluation cases."""
    snapshot = read_snapshot(directory)
    ids = list(ids)
    if (not ids or any(type(value) is not str for value in ids) or len(ids) != len(set(ids))
            or set(ids) - set(snapshot["selection"]["evaluation"])):
        raise ValueError("IDs must belong uniquely to the original evaluation partition.")
    ledger = snapshot["exposure_ledger"]
    if ledger is None:
        raise ValueError("An explicit exposure ledger is required before claiming untouched evaluation groups.")
    if set(ids) & set(ledger["blocked_evaluation_ids"]):
        raise ValueError("Previously exposed evaluation groups cannot be treated as untouched holdout.")
    return {"ids": ids, "known_exposure_overlap": False,
            "exposure_ledger_sha256": snapshot["exposure_ledger_sha256"], "independent_validation": False}


def _load_records(raw, ids):
    selected = {}
    for identity, line in _id_lines(raw):
        if identity in ids:
            value = json.loads(line)
            if type(value) is not dict or value.get("id") != identity:
                raise ValueError("Selected BFCL record violates its ID contract.")
            selected[identity] = value
    if set(selected) != set(ids):
        raise ValueError("A selected conversation is absent from the upstream file.")
    return selected


def load_selected_cases(directory=DEFAULT_SNAPSHOT, ids=None, *, partition="development", include_reference=False):
    """Return selected development records, with private references only on opt-in.

    Reserved cases cannot be loaded by this exploratory loader. A future reserved
    evaluator needs a separate explicit protocol, not a boolean bypass.
    """
    snapshot = read_snapshot(directory)
    if partition != "development" or type(include_reference) is not bool:
        raise ValueError("Only the frozen development partition may be loaded in this exploratory stage.")
    allowed = snapshot["selection"]["development"]
    chosen = list(allowed if ids is None else ids)
    if (not chosen or len(chosen) != len(set(chosen)) or any(type(value) is not str for value in chosen)
            or set(chosen) - set(allowed)):
        raise ValueError("Select unique IDs wholly inside the frozen development partition.")
    root = Path(snapshot["root"])
    # Validate IDs before reading/decoding any private references.
    entries = _load_records((root / "upstream" / QUESTION_PATH).read_bytes(), set(chosen))
    references = _load_records((root / "upstream" / REFERENCE_PATH).read_bytes(), set(chosen)) if include_reference else {}
    _record_access(root, chosen, include_reference)
    return [{"id": identity, "source_group": snapshot["selection"]["source_groups"][identity],
             "entry": deepcopy(entries[identity]), "reference": deepcopy(references.get(identity))}
            for identity in chosen]


def _record_access(root, ids, include_reference):
    """Append-only exposure record; snapshot and imported files remain immutable."""
    event = {"time_utc": datetime.now(timezone.utc).isoformat(), "partition": "development",
             "ids": list(ids), "references_decoded": include_reference, "purpose": "development_loader_access"}
    path = root / "access.jsonl"
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(event, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")

