"""Client-side rate limiter and MistralModel retry behaviour, with a fake clock and fake client (no API calls)."""

import threading
import time
from types import SimpleNamespace

import pytest

from recon.llm import MistralModel, ModelError, RateLimiter, shared_limiter


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.now += s


def limiter(rate, burst=1.0):
    clock = FakeClock()
    return RateLimiter(rate, burst=burst, clock=clock, sleep=clock.sleep), clock


def test_first_call_is_free_then_calls_are_spaced():
    lim, clock = limiter(rate=4)
    waits = [lim.acquire() for _ in range(4)]
    assert waits == pytest.approx([0, 0.25, 0.25, 0.25])
    assert clock.now == pytest.approx(0.75)


def test_idle_time_refills_up_to_burst_only():
    lim, clock = limiter(rate=2, burst=3)
    for _ in range(3):
        lim.acquire()
    clock.now += 100  # a long idle period must not bank more than `burst` calls
    assert [lim.acquire() for _ in range(4)] == pytest.approx([0, 0, 0, 0.5])


def test_concurrent_reservations_queue_instead_of_colliding():
    """Without sleeping: N reservations made at the same instant get N distinct slots."""
    lim, clock = limiter(rate=5)
    lim.sleep = lambda s: None  # reserve only, so all callers book at t=0
    waits = sorted(lim.acquire() for _ in range(5))
    assert waits == pytest.approx([0, 0.2, 0.4, 0.6, 0.8])


def test_timeout_refunds_the_reservation():
    lim, clock = limiter(rate=1)
    lim.sleep = lambda s: None
    lim.acquire()
    lim.acquire()  # booked for t=1
    with pytest.raises(TimeoutError):
        lim.acquire(timeout_s=1.5)  # would be t=2
    assert lim.acquire(timeout_s=2.5) == pytest.approx(2.0)  # the refunded slot is reused


def test_real_threads_respect_the_rate():
    lim = RateLimiter(rate_per_s=50)
    t0 = time.monotonic()
    threads = [threading.Thread(target=lim.acquire) for _ in range(26)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert time.monotonic() - t0 >= 25 / 50 * 0.9  # 1 free + 25 spaced at 20 ms


def test_shared_limiter_is_one_per_rate():
    assert shared_limiter(3.0) is shared_limiter(3.0)
    assert shared_limiter(3.0) is not shared_limiter(7.0)


# ---------- MistralModel with a fake client ----------

class HTTPError(Exception):
    def __init__(self, status_code):
        super().__init__(f"status {status_code}")
        self.status_code = status_code


def fake_response():
    call = SimpleNamespace(id="abc123def", function=SimpleNamespace(name="search_invoices", arguments='{"query": "x"}'))
    msg = SimpleNamespace(content="", tool_calls=[call])
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)],
                           usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2))


def model_with(outcomes, lim=None, monkeypatch=None):
    m = MistralModel.__new__(MistralModel)  # skip __init__: no API key, no network client
    m.name, m.timeout_s, m.max_retries, m.limiter = "fake", 30.0, 3, lim
    queue = list(outcomes)

    def complete(**_):
        o = queue.pop(0)
        if isinstance(o, Exception):
            raise o
        return o

    m.client = SimpleNamespace(chat=SimpleNamespace(complete=complete))
    if monkeypatch:
        monkeypatch.setattr("recon.llm.time.sleep", lambda s: None)  # skip backoff sleeps
    return m


def test_retries_429_then_succeeds_and_each_attempt_uses_the_limiter(monkeypatch):
    lim, clock = limiter(rate=2)
    m = model_with([HTTPError(429), HTTPError(429), fake_response()], lim, monkeypatch)
    res = m.complete([], [])
    assert res.retries == 2 and res.tool_calls[0].name == "search_invoices"
    assert res.wait_ms == pytest.approx(1000)  # 3 attempts: 0 + 0.5 + 0.5 s queued


def test_gives_up_after_max_retries(monkeypatch):
    m = model_with([HTTPError(429)] * 4, None, monkeypatch)
    with pytest.raises(ModelError, match="429"):
        m.complete([], [])


def test_non_retryable_error_fails_immediately(monkeypatch):
    m = model_with([HTTPError(400), fake_response()], None, monkeypatch)
    with pytest.raises(ModelError, match="400"):
        m.complete([], [])


def test_limiter_queue_past_budget_fails_fast():
    lim, clock = limiter(rate=0.01)  # one call per 100 s
    lim.sleep = lambda s: None
    m = model_with([fake_response(), fake_response()], lim)
    m.complete([], [])
    with pytest.raises(ModelError, match="rate limited locally"):
        m.complete([], [])
