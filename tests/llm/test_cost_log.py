import csv
from datetime import UTC, datetime
from pathlib import Path

import pytest

from codelens.core.models import CostLogEntry, CostRecord
from codelens.llm.cost_log import COLUMNS, CsvCostLog

NOW = datetime(2026, 11, 7, 12, 0, tzinfo=UTC)
COST = CostRecord("claude-opus-5-5", 1000, 100, 0, 0, 0.0060004, 1.23456)


def test_writes_header_once_and_rows(tmp_path: Path) -> None:
    log = CsvCostLog(tmp_path / "nested" / "cost.csv", now=lambda: NOW)

    log.record(CostLogEntry("rag", "demo", "q1?", COST))
    log.record(CostLogEntry("rag", "demo", "q2, with comma?", COST))

    with (tmp_path / "nested" / "cost.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == COLUMNS
    assert [r["question"] for r in rows] == ["q1?", "q2, with comma?"]
    assert rows[0]["timestamp"] == NOW.isoformat()
    assert rows[0]["cost_usd"] == "0.006000"
    assert rows[0]["latency_seconds"] == "1.235"


def test_total_cost(tmp_path: Path) -> None:
    log = CsvCostLog(tmp_path / "cost.csv")
    assert log.total_cost() == 0.0

    log.record(CostLogEntry("rag", "demo", "q", COST))
    log.record(CostLogEntry("rag", "demo", "q", COST))

    assert log.total_cost() == pytest.approx(0.012)
