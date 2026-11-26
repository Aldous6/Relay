"""Explicit harness pacing; never retries, changes a model or trims input."""
from copy import deepcopy
import math
import time

from relay_router.agent import ModelReply
from relay_router.errors import BackendContextError
from .agent_context import render_messages


def _known_total(usage):
    """Only reconcile a reservation with internally consistent reported usage."""
    total = usage.get("total_tokens")
    if type(total) is not int or total < 0:
        return None
    parts = [usage.get("input_tokens"), usage.get("output_tokens")]
    known = [value for value in parts if value is not None]
    if any(type(value) is not int or value < 0 for value in known):
        return None
    if sum(known) > total or len(known) == 2 and sum(known) != total:
        return None
    return total


class PacedEpisodeGenerator:
    def __init__(self, generate, *, min_interval_seconds, max_request_budget,
                 max_output_tokens, clock=time.monotonic, sleep=time.sleep, show_progress=False,
                 tokens_per_minute=None, window_seconds=61):
        if (type(min_interval_seconds) not in (int, float) or not math.isfinite(min_interval_seconds)
                or min_interval_seconds < 0 or type(max_request_budget) is not int or max_request_budget < 1
                or type(max_output_tokens) is not int or max_output_tokens < 1):
            raise ValueError("Declarar intervalo y presupuesto positivos de pacing.")
        if (tokens_per_minute is not None and
                (type(tokens_per_minute) is not int or tokens_per_minute < 1)):
            raise ValueError("tokens_per_minute requiere un entero positivo o None.")
        if (type(window_seconds) not in (int, float) or not math.isfinite(window_seconds)
                or window_seconds < 61):
            raise ValueError("La ventana de cuota requiere al menos 61 segundos.")
        self.generate = generate
        self.interval = min_interval_seconds
        self.budget = max_request_budget
        self.output = max_output_tokens
        self.clock, self.sleep = clock, sleep
        self.show_progress = show_progress
        self._last_started = {}
        self.tokens_per_minute = tokens_per_minute
        self.window = window_seconds
        self._reservations = {}

    def manifest(self):
        return {**deepcopy(self.generate.manifest()), "harness_pacing": {
            "minimum_start_interval_seconds_per_model": self.interval,
            "request_budget_utf8_bytes_plus_output_and_reserve": self.budget,
            "output_token_reservation": self.output, "template_reserve": 512,
            "tokens_per_minute_per_model": self.tokens_per_minute,
            "token_window_seconds": self.window,
            "token_accounting": "reserve request bound; replace with consistent reported total; expire after completion",
            "console_progress": self.show_progress,
            "scope": "conservative local pacing; not exact tokenization or coordination with other account clients",
            "wait_in_generation_and_episode_time": True}}

    def _wait_seconds(self, model, budget, now):
        waiting = max(0., self._last_started.get(model, -math.inf) + self.interval - now)
        if self.tokens_per_minute is None:
            return waiting
        reservations = [item for item in self._reservations.get(model, [])
                        if item[0] + self.window > now]
        self._reservations[model] = reservations
        used = sum(item[1] for item in reservations)
        for timestamp, tokens in sorted(reservations):
            if used + budget <= self.tokens_per_minute:
                break
            used -= tokens
            waiting = max(waiting, timestamp + self.window - now)
        return waiting

    def __call__(self, model, request):
        measure = getattr(self.generate, "request_input_bytes", None)
        if callable(measure):
            input_bytes = measure(request)
        else:
            input_bytes = sum(len(item["content"].encode("utf-8")) for item in render_messages(request))
        budget = input_bytes + self.output + 512
        if budget > self.budget or (self.tokens_per_minute is not None and budget > self.tokens_per_minute):
            raise BackendContextError("La solicitud supera el presupuesto conservador de cuota; no se recorta ni envía.")
        wait_started = self.clock()
        waiting = self._wait_seconds(model, budget, wait_started)
        remaining = waiting
        if self.show_progress and remaining > 0:
            print(f"[cuota] Esperando {remaining:.0f} s antes de enviar la siguiente llamada.", flush=True)
        last_notice = self.clock()
        while remaining > 0:
            self.sleep(min(remaining, 10.))
            remaining = self._wait_seconds(model, budget, self.clock())
            if self.show_progress and remaining > 0 and self.clock() - last_notice >= 20:
                print(f"[cuota] Faltan {remaining:.0f} s.", flush=True)
                last_notice = self.clock()
        self._last_started[model] = self.clock()  # failed attempts consume the interval too
        waiting = self._last_started[model] - wait_started if waiting > 0 else 0.
        reservation = None
        if self.tokens_per_minute is not None:
            reservation = [self._last_started[model], budget]
            self._reservations.setdefault(model, []).append(reservation)
        if self.show_progress:
            print(f"[API] Enviando solicitud a {model}.", flush=True)
        try:
            reply = self.generate(model, request)
            if reservation is not None and isinstance(reply, ModelReply):
                total = _known_total(reply.usage)
                if total is not None:
                    reservation[1] = total
        finally:
            # Keep the full reservation for failures/unknown usage. Expiration
            # starts after completion so slow generations cannot free it early.
            if reservation is not None:
                reservation[0] = self.clock()
        if isinstance(reply, ModelReply):
            usage = deepcopy(reply.usage)
            metadata = usage.setdefault("provider_metadata", {})
            metadata["quota_wait_seconds"] = waiting
            return ModelReply(reply.content, usage, reply.cost_usd)
        return reply
