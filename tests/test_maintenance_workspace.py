"""Real temporary-file and unittest checks for the maintenance work copy."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from relay_router.maintenance_workspace import Workspace


SOURCE = "def add(a, b):\n    return a - b\n"
TEST = """import unittest
from calculator import add

class CalculatorTests(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)
"""


class MaintenanceWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / "repo"
        self.root.mkdir()
        (self.root / "calculator.py").write_bytes(SOURCE.encode())
        (self.root / "test_calculator.py").write_bytes(TEST.encode())
        self.run_dir = self.base / "run"

    def workspace(self):
        workspace = Workspace(self.root, self.run_dir, ["calculator.py"], ["test_calculator.py"])
        workspace.prepare()
        return workspace

    def fix(self, workspace):
        workspace.apply_edits([{"path": "calculator.py", "old_text": "return a - b", "new_text": "return a + b"}])

    def test_actual_failure_patch_verification_and_reviewed_application(self):
        workspace = self.workspace()
        baseline = workspace.verify(Path(sys.executable), 10)
        self.assertEqual(baseline["status"], "test_failed")
        self.assertEqual(baseline["tests_run"], 1)
        self.assertFalse(baseline["passed"])
        self.fix(workspace)
        self.assertIn("-    return a - b", workspace.diff())
        self.assertEqual((self.root / "calculator.py").read_text(), SOURCE)
        result = workspace.verify(Path(sys.executable), 10)
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["status"], "passed")
        reopened = Workspace.reopen(self.run_dir)
        applied = reopened.apply_to_original()
        self.assertEqual(applied["files"], ["calculator.py"])
        self.assertIn("return a + b", (self.root / "calculator.py").read_text())
        self.assertEqual((workspace.originals / "calculator.py").read_text(), SOURCE)

    def test_copies_only_authorized_python_files_and_reads_only_them(self):
        (self.root / ".env").write_text("TOKEN=never-copy")
        (self.root / "private.py").write_text("SECRET = 'do not read'")
        workspace = self.workspace()
        self.assertEqual({p.name for p in workspace.worktree.iterdir()}, {"calculator.py", "test_calculator.py"})
        with self.assertRaises(ValueError):
            workspace.read_file("private.py")
        with self.assertRaises(ValueError):
            workspace.read_file("../repo/calculator.py")

    def test_rejects_path_escape_sensitive_paths_and_aliases(self):
        for name in ["../calculator.py", "/tmp/calculator.py", "C:/calculator.py", "src/../calculator.py",
                     "src//calculator.py", "data/example.py", ".git/a.py", "src/.hidden/a.py",
                     "outputs/a.py", "x\\a.py", "secret.env", "src./a.py"]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                Workspace(self.root, self.run_dir, [name], ["test_calculator.py"])
        with self.assertRaises(ValueError):
            Workspace(self.root, self.run_dir, ["calculator.py"], ["calculator.py"])

    def test_does_not_follow_symlink(self):
        target = self.base / "outside.py"
        target.write_text("secret = True")
        link = self.root / "link.py"
        try:
            link.symlink_to(target)
        except OSError:
            self.skipTest("OS does not permit symlink creation for this user")
        with self.assertRaises(ValueError):
            Workspace(self.root, self.run_dir, ["link.py"], ["test_calculator.py"]).prepare()

    def test_rejects_hardlinked_source(self):
        try:
            os.link(self.root / "calculator.py", self.root / "alias.py")
        except OSError:
            self.skipTest("OS does not permit hardlinks here")
        with self.assertRaises(ValueError):
            self.workspace()

    @unittest.skipUnless(os.name == "nt", "Windows junction contract")
    def test_does_not_read_through_windows_junction(self):
        import _winapi
        target = self.base / "outside"
        target.mkdir()
        (target / "calculator.py").write_text("secret = True")
        junction = self.root / "src"
        _winapi.CreateJunction(str(target), str(junction))
        try:
            workspace = Workspace(self.root, self.run_dir, ["src/calculator.py"], ["test_calculator.py"])
            with self.assertRaises(ValueError):
                workspace.validate_inputs()
        finally:
            junction.rmdir()
        self.assertEqual((target / "calculator.py").read_text(), "secret = True")

    def test_input_preflight_does_not_create_output(self):
        workspace = Workspace(self.root, self.run_dir, ["calculator.py"], ["test_calculator.py"])
        result = workspace.validate_inputs()
        self.assertEqual(result["total_bytes"], len(SOURCE.encode()) + len(TEST.encode()))
        self.assertEqual(len(result["files"]), 2)
        self.assertFalse(self.run_dir.exists())

    def test_nonempty_output_is_not_overwritten(self):
        self.run_dir.mkdir()
        prior = self.run_dir / "report.json"
        prior.write_text("retain")
        with self.assertRaises(ValueError):
            self.workspace()
        self.assertEqual(prior.read_text(), "retain")
        self.assertFalse((self.run_dir / "worktree").exists())

    def test_package_initializers_must_be_explicitly_copied(self):
        package = self.root / "src" / "samplepkg"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("")
        (package / "calculator.py").write_text(SOURCE)
        workspace = Workspace(self.root, self.run_dir, ["src/samplepkg/calculator.py"], ["test_calculator.py"])
        with self.assertRaisesRegex(ValueError, "src/samplepkg/__init__.py"):
            workspace.validate_inputs()
        self.assertFalse(self.run_dir.exists())

    def test_installed_or_external_package_cannot_create_false_green(self):
        package = self.root / "src" / "samplepkg"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("")
        (package / "calculator.py").write_text(SOURCE)
        external = self.base / "external_packages"
        (external / "samplepkg").mkdir(parents=True)
        (external / "samplepkg" / "__init__.py").write_text("")
        (external / "samplepkg" / "calculator.py").write_text(SOURCE.replace("a - b", "a + b"))
        test = ("import sys\nsys.path.insert(0, " + repr(str(external)) + ")\n" +
                TEST.replace("from calculator import add", "from samplepkg.calculator import add"))
        (self.root / "test_calculator.py").write_text(test)
        workspace = Workspace(self.root, self.run_dir,
                              ["src/samplepkg/__init__.py", "src/samplepkg/calculator.py"], ["test_calculator.py"])
        workspace.prepare()
        result = workspace.verify(Path(sys.executable), 10)
        self.assertFalse(result["passed"])
        self.assertEqual(result["status"], "import_provenance_error")
        self.assertIn("samplepkg.calculator", result["result"]["import_provenance_errors"])

    def test_src_package_is_verified_inside_copy(self):
        package = self.root / "src" / "samplepkg"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("")
        (package / "calculator.py").write_text(SOURCE)
        (self.root / "test_calculator.py").write_text(TEST.replace("from calculator import add", "from samplepkg.calculator import add"))
        workspace = Workspace(self.root, self.run_dir,
                              ["src/samplepkg/__init__.py", "src/samplepkg/calculator.py"], ["test_calculator.py"])
        workspace.prepare()
        self.assertEqual(workspace.verify(Path(sys.executable), 10)["status"], "test_failed")
        workspace.apply_edits([{"path": "src/samplepkg/calculator.py", "old_text": "a - b", "new_text": "a + b"}])
        result = workspace.verify(Path(sys.executable), 10)
        self.assertTrue(result["passed"], result)

    def test_all_edits_are_validated_before_mutation(self):
        workspace = self.workspace()
        with self.assertRaises(ValueError):
            workspace.apply_edits([
                {"path": "calculator.py", "old_text": "return a - b", "new_text": "return a + b"},
                {"path": "test_calculator.py", "old_text": "5", "new_text": "-1"},
            ])
        self.assertEqual(workspace.read_file("calculator.py"), SOURCE)
        with self.assertRaises(ValueError):
            workspace.apply_edits([{"path": "calculator.py", "old_text": "a", "new_text": "b"}])
        self.assertEqual(workspace.diff(), "")

    def test_preserves_crlf_and_indent(self):
        raw = SOURCE.replace("\n", "\r\n").encode()
        (self.root / "calculator.py").write_bytes(raw)
        workspace = self.workspace()
        workspace.apply_edits([{"path": "calculator.py", "old_text": "return a - b",
                                "new_text": "value = a + b\n    return value"}])
        result = (workspace.worktree / "calculator.py").read_bytes()
        self.assertNotIn(b"\n", result.replace(b"\r\n", b""))
        self.assertIn(b"    value = a + b\r\n    return value", result)
        self.assertEqual((workspace.originals / "calculator.py").read_bytes(), raw)

    def eol_workspace(self, raw, name):
        (self.root / "calculator.py").write_bytes(raw)
        self.run_dir = self.base / name
        return self.workspace()

    def test_eol_lf_old_text_matches_crlf_source_without_changing_indentation(self):
        logical = "def add(a, b):\n\tvalue = a - b\n\treturn value\n"
        raw = logical.replace("\n", "\r\n").encode()
        workspace = self.eol_workspace(raw, "lf-old-crlf-file")
        workspace.apply_edits([{"path": "calculator.py", "old_text": logical,
                                "new_text": logical.replace("a - b", "a + b")}])
        expected = raw.replace(b"a - b", b"a + b")
        self.assertEqual((workspace.worktree / "calculator.py").read_bytes(), expected)
        self.assertEqual((workspace.originals / "calculator.py").read_bytes(), raw)
        self.assertEqual((self.root / "calculator.py").read_bytes(), raw)

    def test_eol_crlf_old_text_matches_lf_source_and_keeps_lf(self):
        raw = SOURCE.encode()
        workspace = self.eol_workspace(raw, "crlf-old-lf-file")
        workspace.apply_edits([{"path": "calculator.py", "old_text": SOURCE.replace("\n", "\r\n"),
                                "new_text": SOURCE.replace("a - b", "a + b").replace("\n", "\r\n")}])
        self.assertEqual((workspace.worktree / "calculator.py").read_bytes(), raw.replace(b"a - b", b"a + b"))

    def test_eol_new_crlf_fragment_is_adapted_to_lf_file(self):
        workspace = self.eol_workspace(SOURCE.encode(), "crlf-new-lf-file")
        workspace.apply_edits([{"path": "calculator.py", "old_text": "    return a - b",
                                "new_text": "    value = a + b\r\n    return value"}])
        self.assertEqual((workspace.worktree / "calculator.py").read_bytes(),
                         b"def add(a, b):\n    value = a + b\n    return value\n")

    def test_eol_equivalence_does_not_add_a_final_newline(self):
        logical = SOURCE.rstrip("\n")
        raw = logical.replace("\n", "\r\n").encode()
        workspace = self.eol_workspace(raw, "no-final-newline")
        workspace.apply_edits([{"path": "calculator.py", "old_text": logical,
                                "new_text": logical.replace("a - b", "a + b")}])
        self.assertEqual((workspace.worktree / "calculator.py").read_bytes(), raw.replace(b"a - b", b"a + b"))

    def test_eol_equivalent_ambiguous_and_overlapping_matches_are_rejected(self):
        for index, raw in enumerate((b"x\r\ny\r\nx\r\ny\r\n", b"x\r\nx\r\nx\r\n")):
            with self.subTest(index=index):
                workspace = self.eol_workspace(raw, f"ambiguous-{index}")
                old = "x\ny" if index == 0 else "x\nx"
                with self.assertRaises(ValueError):
                    workspace.apply_edits([{"path": "calculator.py", "old_text": old, "new_text": "z"}])
                self.assertEqual((workspace.worktree / "calculator.py").read_bytes(), raw)

    def test_eol_matching_never_normalizes_indentation_spaces_or_code(self):
        raw = SOURCE.replace("\n", "\r\n").encode()
        for index, old in enumerate((SOURCE.replace("    return", "\treturn"),
                                     SOURCE.replace("    return", "  return"),
                                     SOURCE.replace("a - b", "a-b"),
                                     SOURCE.replace("a - b", "a + b"))):
            with self.subTest(index=index):
                workspace = self.eol_workspace(raw, f"different-text-{index}")
                with self.assertRaises(ValueError):
                    workspace.apply_edits([{"path": "calculator.py", "old_text": old, "new_text": SOURCE}])
                self.assertEqual((workspace.worktree / "calculator.py").read_bytes(), raw)

    def test_eol_mixed_or_bare_cr_fragments_are_not_silently_normalized(self):
        raw = SOURCE.replace("\n", "\r\n").encode()
        mixed = "def add(a, b):\r\n    return a - b\n"
        bare_cr = "def add(a, b):\r    return a - b\r"
        edits = [(mixed, SOURCE), (bare_cr, SOURCE),
                 (SOURCE.replace("\n", "\r\n"), mixed.replace("a - b", "a + b")),
                 (SOURCE.replace("\n", "\r\n"), bare_cr.replace("a - b", "a + b"))]
        for index, (old, new) in enumerate(edits):
            with self.subTest(index=index):
                workspace = self.eol_workspace(raw, f"unsafe-fragment-{index}")
                with self.assertRaises(ValueError):
                    workspace.apply_edits([{"path": "calculator.py", "old_text": old, "new_text": new}])
                self.assertEqual((workspace.worktree / "calculator.py").read_bytes(), raw)

    def test_eol_mixed_file_rejects_normalization_but_allows_exact_single_line_change(self):
        raw = b"def add(a, b):\r\n    return a - b\n"
        workspace = self.eol_workspace(raw, "mixed-file")
        with self.assertRaises(ValueError):
            workspace.apply_edits([{"path": "calculator.py", "old_text": SOURCE,
                                    "new_text": SOURCE.replace("a - b", "a + b")}])
        self.assertEqual((workspace.worktree / "calculator.py").read_bytes(), raw)
        workspace.apply_edits([{"path": "calculator.py", "old_text": "a - b", "new_text": "a + b"}])
        self.assertEqual((workspace.worktree / "calculator.py").read_bytes(), raw.replace(b"a - b", b"a + b"))

    def test_eol_conversion_keeps_batch_validation_atomic(self):
        raw = SOURCE.replace("\n", "\r\n").encode()
        workspace = self.eol_workspace(raw, "atomic-eol")
        with self.assertRaises(ValueError):
            workspace.apply_edits([
                {"path": "calculator.py", "old_text": SOURCE, "new_text": SOURCE.replace("a - b", "a + b")},
                {"path": "calculator.py", "old_text": "    return value", "new_text": "    return 0"},
            ])
        self.assertEqual((workspace.worktree / "calculator.py").read_bytes(), raw)

    def test_application_requires_verified_unchanged_content(self):
        workspace = self.workspace()
        self.fix(workspace)
        with self.assertRaises(ValueError):
            workspace.apply_to_original()
        self.assertTrue(workspace.verify(Path(sys.executable), 10)["passed"])
        workspace.apply_edits([{"path": "calculator.py", "old_text": "a + b", "new_text": "a * b"}])
        with self.assertRaises(ValueError):
            workspace.apply_to_original()
        with self.assertRaises(ValueError):
            Workspace.reopen(self.run_dir).apply_to_original()

    def test_does_not_overwrite_concurrent_user_edit_or_test_change(self):
        workspace = self.workspace()
        self.fix(workspace)
        self.assertTrue(workspace.verify(Path(sys.executable), 10)["passed"])
        user_text = TEST + "\n# User adds another test\n"
        (self.root / "test_calculator.py").write_text(user_text)
        with self.assertRaises(ValueError):
            workspace.apply_to_original()
        self.assertEqual((self.root / "calculator.py").read_text(), SOURCE)

    def test_changed_snapshot_and_verification_are_rejected(self):
        workspace = self.workspace()
        self.fix(workspace)
        self.assertTrue(workspace.verify(Path(sys.executable), 10)["passed"])
        verification_path = self.run_dir / "verification.json"
        verification = json.loads(verification_path.read_text())
        verification["tests_run"] = 999
        verification_path.write_text(json.dumps(verification))
        with self.assertRaises(ValueError):
            Workspace.reopen(self.run_dir)

    def test_empty_suite_is_not_a_success(self):
        (self.root / "test_calculator.py").write_text("import unittest\n")
        result = self.workspace().verify(Path(sys.executable), 10)
        self.assertFalse(result["passed"])
        self.assertEqual(result["status"], "no_tests")

    def test_import_error_is_distinguished_from_failed_assertion(self):
        (self.root / "test_calculator.py").write_text("import this_dependency_does_not_exist\n")
        result = self.workspace().verify(Path(sys.executable), 10)
        self.assertEqual(result["status"], "runner_error")

    def test_only_skipped_suite_is_not_a_success(self):
        (self.root / "test_calculator.py").write_text(TEST.replace("    def test_add", "    @unittest.skip('not exercised')\n    def test_add"))
        result = self.workspace().verify(Path(sys.executable), 10)
        self.assertEqual(result["status"], "no_executed_tests")
        self.assertFalse(result["passed"])

    def test_test_subprocess_does_not_inherit_provider_credentials(self):
        (self.root / "test_calculator.py").write_text("""import unittest, os
class EnvironmentTests(unittest.TestCase):
    def test_no_provider_key(self):
        self.assertNotIn('GROQ_API_KEY', os.environ)
        self.assertNotIn('OPENAI_API_KEY', os.environ)
""")
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-only-placeholder", "OPENAI_API_KEY": "test-only-placeholder"}):
            result = self.workspace().verify(Path(sys.executable), 10)
        self.assertTrue(result["passed"], result)

    def test_timeout_is_not_reported_as_test_failure_or_success(self):
        (self.root / "test_calculator.py").write_text("import time\ntime.sleep(30)\n")
        result = self.workspace().verify(Path(sys.executable), .3)
        self.assertEqual(result["status"], "timeout")
        self.assertFalse(result["passed"])

    def test_tests_modifying_sources_cannot_approve_application(self):
        (self.root / "test_calculator.py").write_text("""import unittest
from pathlib import Path
class MutationTests(unittest.TestCase):
    def test_mutates(self):
        Path('calculator.py').write_text('changed = True')
        self.assertTrue(True)
""")
        result = self.workspace().verify(Path(sys.executable), 10)
        self.assertEqual(result["status"], "workspace_modified_by_tests")
        self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
