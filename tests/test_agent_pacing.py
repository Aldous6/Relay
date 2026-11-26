import unittest
from contextlib import redirect_stdout
from io import StringIO
from relay_router.agent import AgentRequest, AgentStateV8, ModelReply
from relay_router.errors import BackendContextError
from routerbench.agent_pacing import PacedEpisodeGenerator


class Generator:
    def __init__(self): self.calls = []
    def manifest(self): return {"fixture": True}
    def __call__(self, model, request):
        self.calls.append(model)
        return ModelReply('{}', {"input_tokens": 2})


class QuotaGenerator(Generator):
    def __init__(self, now, usage, duration=0):
        super().__init__()
        self.now, self.duration, self.fail = now, duration, False
        self.reply = ModelReply('{}', usage)

    def request_input_bytes(self, request):
        return 388  # Plus 100 output and 512 reserve gives 1,000 per request.

    def __call__(self, model, request):
        self.calls.append((model, self.now[0]))
        self.now[0] += self.duration
        if self.fail:
            raise RuntimeError('fixture transport failure')
        return self.reply


class PacingTests(unittest.TestCase):
    def quota_case(self, usage, *, duration=0, tokens_per_minute=1500, window_seconds=61):
        now, sleeps = [0.], []
        def sleep(seconds):
            sleeps.append(seconds)
            now[0] += seconds
        generate = QuotaGenerator(now, usage, duration)
        paced = PacedEpisodeGenerator(generate, min_interval_seconds=2.1, max_request_budget=2000,
            max_output_tokens=100, tokens_per_minute=tokens_per_minute, window_seconds=window_seconds,
            clock=lambda: now[0], sleep=sleep)
        request = AgentRequest('task', (), AgentStateV8(0, 0, (), (), 3))
        return now, sleeps, generate, paced, request

    def test_consistent_usage_releases_reservation_but_keeps_request_interval(self):
        usage = {'input_tokens': 20, 'output_tokens': 80, 'total_tokens': 100}
        _, sleeps, generate, paced, request = self.quota_case(usage)
        paced('a', request)
        reply = paced('a', request)
        self.assertEqual(generate.calls, [('a', 0.), ('a', 2.1)])
        self.assertEqual(sum(sleeps), 2.1)
        self.assertEqual(reply.usage['provider_metadata']['quota_wait_seconds'], 2.1)
        self.assertNotIn('provider_metadata', generate.reply.usage)
        self.assertEqual(paced.manifest()['harness_pacing']['tokens_per_minute_per_model'], 1500)

    def test_unknown_invalid_or_inconsistent_usage_keeps_full_reservation(self):
        for usage in ({}, {'input_tokens': 1}, {'total_tokens': True}, {'total_tokens': -1},
                      {'total_tokens': 20.0}, {'total_tokens': 20, 'output_tokens': 21},
                      {'total_tokens': 20, 'input_tokens': 1, 'output_tokens': 2},
                      {'total_tokens': 20, 'input_tokens': False}):
            with self.subTest(usage=usage):
                _, sleeps, generate, paced, request = self.quota_case(usage)
                paced('a', request)
                paced('a', request)
                self.assertEqual(generate.calls, [('a', 0.), ('a', 61.)])
                self.assertEqual(sum(sleeps), 61.)
                self.assertLessEqual(max(sleeps), 10.)

    def test_failed_calls_reserve_until_window_after_completion_without_retry(self):
        now, sleeps, generate, paced, request = self.quota_case({}, duration=5)
        generate.fail = True
        with self.assertRaises(RuntimeError):
            paced('a', request)
        self.assertEqual(generate.calls, [('a', 0.)])
        self.assertEqual(now[0], 5.)
        generate.fail = False
        paced('a', request)
        self.assertEqual(generate.calls, [('a', 0.), ('a', 66.)])
        self.assertEqual(sum(sleeps), 61.)

    def test_known_usage_is_retained_until_window_after_slow_completion(self):
        _, sleeps, generate, paced, request = self.quota_case(
            {'input_tokens': 400, 'output_tokens': 200, 'total_tokens': 600}, duration=5)
        paced('a', request)
        paced('a', request)
        self.assertEqual(generate.calls, [('a', 0.), ('a', 66.)])
        self.assertEqual(sum(sleeps), 61.)

    def test_token_windows_are_separate_by_model_and_configurable(self):
        _, sleeps, generate, paced, request = self.quota_case({}, window_seconds=120)
        paced('a', request)
        paced('b', request)
        paced('a', request)
        paced('b', request)
        self.assertEqual(generate.calls, [('a', 0.), ('b', 0.), ('a', 120.), ('b', 120.)])
        self.assertEqual(sum(sleeps), 120.)

    def test_multiple_reservations_expire_only_as_needed_for_the_next_request(self):
        _, _, generate, paced, request = self.quota_case({'total_tokens': 400}, tokens_per_minute=1900)
        for _ in range(4):
            paced('a', request)
        self.assertEqual(generate.calls, [('a', 0.), ('a', 2.1), ('a', 4.2), ('a', 61.)])

    def test_request_that_cannot_fit_token_window_is_rejected_before_send(self):
        _, sleeps, generate, paced, request = self.quota_case({}, tokens_per_minute=999)
        with self.assertRaises(BackendContextError):
            paced('a', request)
        self.assertEqual(generate.calls, [])
        self.assertEqual(sleeps, [])

    def test_invalid_token_limits_or_short_windows_are_rejected(self):
        for quota, window in ((True, 61), (0, 61), (8000.0, 61), (8000, 60),
                              (8000, float('nan')), (8000, True)):
            with self.subTest(quota=quota, window=window), self.assertRaises(ValueError):
                self.quota_case({}, tokens_per_minute=quota, window_seconds=window)

    def test_schema_and_tool_history_budget_is_used_before_sending(self):
        generate = Generator()
        generate.request_input_bytes = lambda request: 9000
        paced = PacedEpisodeGenerator(generate, min_interval_seconds=61, max_request_budget=8000,
                                       max_output_tokens=100)
        request = AgentRequest('short text but large tools', (), AgentStateV8(0, 0, (), (), 3))
        with self.assertRaises(BackendContextError):
            paced('model', request)
        self.assertEqual(generate.calls, [])

    def test_console_progress_shows_wait_and_send_without_prompt(self):
        now = [0.]
        def sleep(seconds): now[0] += seconds
        paced = PacedEpisodeGenerator(Generator(), min_interval_seconds=61, max_request_budget=8000,
            max_output_tokens=100, clock=lambda: now[0], sleep=sleep, show_progress=True)
        request = AgentRequest('PRIVATE PROMPT', (), AgentStateV8(0, 0, (), (), 2))
        with redirect_stdout(StringIO()) as output:
            paced('model', request)
            paced('model', request)
        self.assertIn('Esperando 61 s', output.getvalue())
        self.assertIn('Faltan', output.getvalue())
        self.assertEqual(output.getvalue().count('[API]'), 2)
        self.assertNotIn('PRIVATE PROMPT', output.getvalue())

    def test_paces_each_model_without_retry_or_mutating_usage(self):
        now, sleeps = [0.], []
        def sleep(seconds):
            sleeps.append(seconds)
            now[0] += seconds
        generate = Generator()
        paced = PacedEpisodeGenerator(generate, min_interval_seconds=61, max_request_budget=8000,
                                       max_output_tokens=100, clock=lambda: now[0], sleep=sleep)
        request = AgentRequest('task', (), AgentStateV8(0, 0, (), (), 3))
        paced('a', request)
        paced('b', request)
        reply = paced('a', request)
        self.assertEqual(generate.calls, ['a', 'b', 'a'])
        self.assertEqual(sum(sleeps), 61)
        self.assertLessEqual(max(sleeps), 10)
        self.assertEqual(reply.usage['provider_metadata']['quota_wait_seconds'], 61)
        self.assertTrue(paced.manifest()['harness_pacing']['wait_in_generation_and_episode_time'])

    def test_oversized_input_is_not_trimmed_or_sent(self):
        generate = Generator()
        paced = PacedEpisodeGenerator(generate, min_interval_seconds=61, max_request_budget=50, max_output_tokens=10)
        request = AgentRequest('task', (), AgentStateV8(0, 0, (), (), 3))
        with self.assertRaises(BackendContextError): paced('a', request)
        self.assertEqual(generate.calls, [])
