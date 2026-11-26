import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_bfcl_tokenizer", ROOT / "tools/prepare_bfcl_tokenizer.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class PrepareTokenizerTests(unittest.TestCase):
    def test_default_only_lists_missing_files_without_network_or_writes(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, "urlopen") as network:
            target = Path(directory) / "not-created"
            result = runner.prepare_assets(target)
            self.assertFalse(target.exists())
        network.assert_not_called()
        self.assertFalse(result["ready"])
        self.assertEqual(result["provider_calls"], 0)
        self.assertEqual(len(result["files"]), 10)
        self.assertTrue(all(row["status"] == "missing" for row in result["files"]))

    def test_bad_download_never_writes_an_asset(self):
        observed = []
        def fetch(url, **kwargs):
            observed.append(url)
            return io.BytesIO(b"corrupted public bytes")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                runner.prepare_assets(directory, download=True, opener=fetch)
            self.assertEqual(list(Path(directory).rglob("*")), [])
        self.assertEqual(len(observed), 1)
        self.assertTrue(observed[0].startswith("https://huggingface.co/openai/"))

    @unittest.skipUnless((runner.DEFAULT_ASSET_ROOT / "o200k_base.tiktoken").exists(), "Pinned local tokenizer assets unavailable; no download in tests.")
    def test_explicit_download_is_reproducible_and_verified_cache_needs_no_network(self):
        source = runner.DEFAULT_ASSET_ROOT
        provenance = json.loads(runner.DEFAULT_PROVENANCE.read_text(encoding="utf-8"))
        by_url = {row["url"]: row["path"] for row in provenance["assets"]}
        observed = []
        def fetch(url, **kwargs):
            observed.append(url)
            return io.BytesIO((source / by_url[url]).read_bytes())
        with tempfile.TemporaryDirectory() as directory:
            result = runner.prepare_assets(directory, download=True, opener=fetch)
            self.assertTrue(result["ready"])
            self.assertEqual(len(observed), 10)
            with patch.object(runner, "urlopen") as network:
                repeated = runner.prepare_assets(directory, download=True)
            network.assert_not_called()
            self.assertTrue(repeated["ready"])
            self.assertEqual(result["estimator"], repeated["estimator"])
            self.assertFalse(list(Path(directory).rglob("*.verified-download")))

    def test_cli_default_is_prepare_verify(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            result = runner.main(["--asset-root", directory])
        self.assertEqual(result["mode"], "prepare_verify")
        self.assertFalse(result["ready"])


if __name__ == "__main__":
    unittest.main()
