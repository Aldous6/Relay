"""Host unittest runner with distinct exit codes for unusable checks.

0: checks pass; 1: executed tests fail; 2: configuration/import failure,
insufficient non-skipped tests, or no tests. Run only trusted project tests.
This is an optional operational checker, not an independent quality oracle.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import unittest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-directory", default="tests")
    parser.add_argument("--pattern", default="test_*.py")
    parser.add_argument("--min-tests", type=int, default=1)
    args = parser.parse_args(argv)
    if args.min_tests < 1:
        parser.error("--min-tests must be positive")
    root = Path.cwd().resolve()
    start = (root / args.start_directory).resolve()
    if not start.is_relative_to(root) or not start.is_dir():
        print("Check directory is unavailable or outside the workspace.", file=sys.stderr)
        return 2
    # Explicit host project import path, even with python -I. The verifier
    # strips provider credentials from the subprocess environment separately.
    sys.path.insert(0, str(root))
    if (root / "src").is_dir():
        sys.path.insert(0, str(root / "src"))
    loader = unittest.TestLoader()
    try:
        suite = loader.discover(str(start), pattern=args.pattern)
    except (ImportError, OSError, ValueError):
        print("Could not discover the declared tests.", file=sys.stderr)
        return 2
    if loader.errors or suite.countTestCases() < args.min_tests:
        print("Test loading failed or fewer than the declared minimum were discovered.", file=sys.stderr)
        return 2
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    active_tests = result.testsRun - len(result.skipped) - len(result.expectedFailures)
    if active_tests < args.min_tests:
        print("Too few non-skipped, non-expected-failure tests ran.", file=sys.stderr)
        return 2
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
