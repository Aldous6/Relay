"""Prove core imports work without third-party packages or a repository cwd."""
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "src"


class RuntimeImportTests(unittest.TestCase):
    def run_python(self, body, *, no_site=True):
        code = "import sys\nsys.path.insert(0, " + repr(str(SOURCE)) + ")\n" + textwrap.dedent(body)
        arguments = [sys.executable, "-I"]
        if no_site:
            arguments.append("-S")
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(arguments + ["-c", code], cwd=directory,
                                    capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_package_import_is_dependency_free_and_introspectable(self):
        self.run_python('''
            import relay_router
            assert {"Relay", "TaskSession", "RelayRouter", "RoutingDecision", "load_bundle"} <= set(dir(relay_router))
            assert "relay_router.client" not in sys.modules
            assert "relay_router.workflow" not in sys.modules
            for prefix in ("pandas", "numpy", "joblib", "sklearn", "openhands", "litellm"):
                assert not any(name == prefix or name.startswith(prefix + ".") for name in sys.modules), prefix
        ''')

    def test_workflow_exports_work_without_site_packages(self):
        self.run_python('''
            from relay_router import Relay, TaskSession
            from relay_router.workflow import Relay as DirectRelay, TaskSession as DirectSession
            import relay_router
            assert Relay is DirectRelay is relay_router.Relay
            assert TaskSession is DirectSession is relay_router.TaskSession
            for prefix in ("pandas", "numpy", "joblib", "sklearn", "openhands", "litellm"):
                assert not any(name == prefix or name.startswith(prefix + ".") for name in sys.modules), prefix
            assert "relay_router.client" not in sys.modules
        ''')

    def test_historical_exports_keep_original_class_and_function_identity(self):
        self.run_python('''
            from relay_router import RelayRouter, RoutingDecision, load_bundle
            from relay_router.client import RelayRouter as DirectRouter, RoutingDecision as DirectDecision, load_bundle as direct_load
            import relay_router
            assert RelayRouter is DirectRouter is relay_router.RelayRouter
            assert RoutingDecision is DirectDecision is relay_router.RoutingDecision
            assert load_bundle is direct_load is relay_router.load_bundle
            assert "relay_router.workflow" not in sys.modules
        ''', no_site=False)

    def test_unknown_attribute_and_stdlib_submodule_do_not_load_legacy_dependencies(self):
        self.run_python('''
            import relay_router
            try:
                relay_router.not_a_relay_export
            except AttributeError as error:
                assert "not_a_relay_export" in str(error)
            else:
                raise AssertionError("Unknown exports must fail")
            from relay_router import errors
            assert errors.BackendTimeoutError.category == "timeout"
            assert "relay_router.client" not in sys.modules
            assert "pandas" not in sys.modules
        ''')


if __name__ == "__main__":
    unittest.main()
