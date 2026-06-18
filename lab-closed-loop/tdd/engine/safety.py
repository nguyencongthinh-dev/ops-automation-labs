"""Blast-radius guard and circuit breaker for the closed-loop orchestrator."""

import time
from collections import defaultdict, deque

from engine.logger import JsonLogger

log = JsonLogger("safety")


class BlastRadiusGuard:
    """Enforce per-minute global and per-service-per-hour action limits."""

    def __init__(self, max_per_minute: int, max_restarts_per_hour: int):
        self._max_per_minute = max_per_minute
        self._max_restarts_per_hour = max_restarts_per_hour
        self._global_window: deque[float] = deque()
        self._service_window: dict[str, deque[float]] = defaultdict(deque)

    def _prune(self, window: deque, horizon: float):
        while window and window[0] < horizon:
            window.popleft()

    def check(self, service: str) -> tuple[bool, str]:
        now = time.time()
        self._prune(self._global_window, now - 60)
        self._prune(self._service_window[service], now - 3600)
        if len(self._global_window) >= self._max_per_minute:
            return False, f"global actions/min limit ({self._max_per_minute}) reached"
        if len(self._service_window[service]) >= self._max_restarts_per_hour:
            return (
                False,
                f"restarts/hour limit ({self._max_restarts_per_hour}) for {service}",
            )
        return True, "ok"

    def record(self, service: str):
        now = time.time()
        self._global_window.append(now)
        self._service_window[service].append(now)


class CircuitBreaker:
    """Halt automation after N consecutive verify failures."""

    def __init__(self, threshold: int):
        self._threshold = threshold
        self._failures = 0
        self._open = False

    def is_open(self) -> bool:
        return self._open

    def record_failure(self):
        self._failures += 1
        if self._failures >= self._threshold:
            self._open = True
            log.error(
                "CIRCUIT_BREAKER_HALT",
                consecutive_failures=self._failures,
                threshold=self._threshold,
                message="Automation halted. Manual intervention required.",
            )

    def record_success(self):
        self._failures = 0
