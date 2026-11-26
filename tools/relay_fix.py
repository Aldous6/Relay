"""Product shortcut for direct maintenance with a bounded strong-model repair.

Uses the existing CLI's explicit file scope, hidden credential entry, reviewable
diff and separate application step. No extra permissions or provider retries.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from relay_router.maintenance_cli import main

if __name__ == "__main__":
    arguments = sys.argv[1:]
    if arguments and arguments[0] in {"check", "apply"}:
        raise SystemExit(main(arguments))
    if arguments and arguments[0] == "run":
        arguments = arguments[1:]
    # The measured cheap-first cascade did not save cost. Keep it explicitly
    # selectable with --mode verified; use the observed stronger baseline here.
    raise SystemExit(main(["run", "--mode", "activity", "--inspection-mode", "direct",
                          "--output-mode", "prompt_json", *arguments]))
