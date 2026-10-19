"""Locust user: asks a random evaluation question.

A request fails if the HTTP call fails or if the agent itself ended in error,
timeout or step_limit: a 200 that wraps a failed run is still a failure.
"""

import json
import random
from pathlib import Path

from locust import HttpUser, between, task

QUESTIONS = [json.loads(line)["question"] for line in
             (Path(__file__).resolve().parent.parent / "eval" / "questions_dev.jsonl").read_text(encoding="utf-8").splitlines()]


class AccountantUser(HttpUser):
    wait_time = between(0, 0.1)  # near-closed loop: measure the service, not think time

    @task
    def ask(self):
        with self.client.post("/ask", json={"question": random.choice(QUESTIONS)}, name="/ask",
                              catch_response=True, timeout=120) as r:
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}")
            elif r.json()["outcome"] in ("error", "timeout", "step_limit"):
                r.failure(f"agent {r.json()['outcome']}")
