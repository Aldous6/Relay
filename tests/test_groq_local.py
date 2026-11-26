from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock
from contextlib import redirect_stdout
from io import StringIO
from relay_router.agent import ModelReply
from relay_router.errors import BackendTransportError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
spec = importlib.util.spec_from_file_location('groq_local_tested', ROOT / 'tools/groq_local.py')
local = importlib.util.module_from_spec(spec)
spec.loader.exec_module(local)


class LocalGroqTests(unittest.TestCase):
    def test_three_controls_prepare_without_credentials_or_api(self):
        config = local.configuration(False)
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(local, 'ROOT', Path(directory)), patch.object(local, 'configuration', return_value=deepcopy(config)), \
             patch.object(local.getpass, 'getpass') as key, patch.object(local, 'run_api') as run:
            local.main(['--prepare-only', '--controls'])
            saved = json.loads(Path(run.call_args.args[0][1]).read_text())
            self.assertEqual(saved['policies'], ['always_strong', 'always_weak', 'cheap_then_escalate'])
            self.assertEqual(saved['primary_policy'], 'cheap_then_escalate')
            self.assertFalse(saved['spending']['account_is_free_tier'])
            self.assertNotIn('--execute', run.call_args.args[0])
            key.assert_not_called()

    def test_failed_request_replay_uses_verified_journal_without_modifying_it(self):
        original = ROOT / 'outputs/groq-local/20261112T192236690025Z/run/journal.json'
        if not original.exists(): self.skipTest('Local archived diagnostic is not distributed')
        before = original.read_bytes()
        generator = Mock(side_effect=BackendTransportError())
        generator.manifest.return_value = {'litellm_version': 'offline-fixture'}
        with tempfile.TemporaryDirectory() as directory, \
             patch('routerbench.agent_litellm.LiteLLMEpisodeGenerator', return_value=generator), \
             redirect_stdout(StringIO()):
            result = local.diagnose(local.configuration(True), Path(directory), replay_last_failure=True)
        generator.assert_called_once()
        request = generator.call_args.args[1]
        self.assertGreater(len(request.history), 0)
        self.assertIn('replayed_request', result)
        self.assertEqual(original.read_bytes(), before)

    def test_diagnostic_makes_one_call_without_starting_benchmark_or_saving_key(self):
        config = local.configuration(True)
        for fail in (False, True):
            generator = Mock(return_value=ModelReply('{"type":"final","answer":"ok"}'))
            generator.manifest.return_value = {'litellm_version': 'offline-fixture'}
            if fail:
                generator.side_effect = BackendTransportError()
            with tempfile.TemporaryDirectory() as directory, \
                 patch.object(local, 'ROOT', Path(directory)), patch.object(local, 'configuration', return_value=deepcopy(config)), \
                 patch.object(local.sys.stdin, 'isatty', return_value=True), patch('builtins.input', return_value='FREE'), \
                 patch.object(local.getpass, 'getpass', return_value='fixture-only-secret'), \
                 patch('routerbench.agent_litellm.LiteLLMEpisodeGenerator', return_value=generator), \
                 patch.object(local, 'run_api') as benchmark, redirect_stdout(StringIO()) as output:
                result = local.main(['--diagnose'])
                generator.assert_called_once()
                benchmark.assert_not_called()
                self.assertEqual(result['outcome'], 'failed' if fail else 'response_received')
                self.assertNotIn('fixture-only-secret', output.getvalue())
                files = list(Path(directory).rglob('*.json'))
                self.assertEqual(len(files), 2)
                self.assertTrue(all('fixture-only-secret' not in path.read_text() for path in files))

    def test_key_is_only_in_process_and_is_restored_even_on_failure(self):
        config = local.configuration(True)
        for fail in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                def runner(arguments):
                    self.assertEqual(os.environ['GROQ_API_KEY'], 'fixture-new-key')
                    saved = json.loads(Path(arguments[1]).read_text())
                    self.assertNotIn('fixture-new-key', json.dumps(saved))
                    self.assertIn('--execute', arguments)
                    if fail: raise RuntimeError('fixture failure')
                with patch.object(local, 'ROOT', Path(directory)), patch.object(local, 'configuration', return_value=deepcopy(config)), \
                     patch.object(local.sys.stdin, 'isatty', return_value=True), patch('builtins.input', return_value='FREE'), \
                     patch.object(local.getpass, 'getpass', return_value='fixture-new-key'), \
                     patch.object(local, 'run_api', side_effect=runner), patch.dict(os.environ, {'GROQ_API_KEY': 'fixture-old-key'}):
                    if fail:
                        with self.assertRaises(RuntimeError): local.main([])
                    else: local.main([])
                    self.assertEqual(os.environ['GROQ_API_KEY'], 'fixture-old-key')

    def test_free_plan_confirmation_and_real_terminal_required(self):
        with patch.object(local.sys.stdin, 'isatty', return_value=False), patch.object(local.getpass, 'getpass') as key:
            with self.assertRaises(RuntimeError): local.main([])
            key.assert_not_called()
        with patch.object(local.sys.stdin, 'isatty', return_value=True), patch('builtins.input', return_value='no'), patch.object(local, 'run_api') as run:
            with self.assertRaises(RuntimeError): local.main([])
            run.assert_not_called()

    def test_diagnostic_configuration_has_no_inline_secret_or_fake_prices(self):
        config = local.configuration()
        self.assertFalse(config['spending']['account_is_free_tier'])
        self.assertEqual(config['pricing'], {})
        self.assertEqual(config['per_family'], 1)
        self.assertEqual(config['policies'], ['always_strong'])
        self.assertEqual(config['pacing']['min_interval_seconds'], 2.1)
        self.assertEqual(config['pacing']['tokens_per_minute'], 8000)
        self.assertEqual(config['generation_options']['action_format'], 'native_tools')
