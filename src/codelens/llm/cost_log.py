"""Append every LLM call to a CSV file: the evidence for the paper's cost section."""

import csv
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from codelens.core.models import CostLogEntry

COLUMNS = [
    "timestamp", "label", "repo_id", "question", "model", "input_tokens", "output_tokens",
    "cache_write_tokens", "cache_read_tokens", "cost_usd", "latency_seconds",
]  # fmt: skip


class CsvCostLog:
    def __init__(self, path: Path, now: Callable[[], datetime] = lambda: datetime.now(UTC)) -> None:
        self._path = path
        self._now = now
        self._lock = threading.Lock()  # the API may answer several requests at once

    def record(self, entry: CostLogEntry) -> None:
        cost = entry.cost
        row = {
            "timestamp": self._now().isoformat(),
            "label": entry.label,
            "repo_id": entry.repo_id,
            "question": entry.question,
            "model": cost.model,
            "input_tokens": cost.input_tokens,
            "output_tokens": cost.output_tokens,
            "cache_write_tokens": cost.cache_write_tokens,
            "cache_read_tokens": cost.cache_read_tokens,
            "cost_usd": f"{cost.cost_usd:.6f}",
            "latency_seconds": f"{cost.latency_seconds:.3f}",
        }
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            is_new = not self._path.exists()
            with self._path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=COLUMNS)
                if is_new:
                    writer.writeheader()
                writer.writerow(row)

    def total_cost(self) -> float:
        if not self._path.exists():
            return 0.0
        with self._path.open(newline="", encoding="utf-8") as f:
            return sum(float(row["cost_usd"]) for row in csv.DictReader(f))
