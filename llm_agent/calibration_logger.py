"""CSV logger for calibrated Saez pre-simulation samples.

The calibrated Saez design needs empirical period-income samples from the same
agent population. This logger records one row per completed tax period and
agent, using the Foundation tax component's own tax accounting.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


CALIBRATION_FIELDS = [
    "run_name",
    "period_index",
    "period_start_step",
    "period_end_step",
    "agent_id",
    "income",
    "income_per_step",
    "marginal_rate",
    "effective_rate",
    "tax_paid",
    "lump_sum",
    "coin_after_tax",
    "total_coin_endowment_after_tax",
    "schedule_json",
    "bracket_cutoffs_json",
]


class CalibrationCSVLogger:
    """Append-only CSV logger for random-tax calibration samples."""

    def __init__(self, csv_path: str | Path, run_name: str) -> None:
        self.csv_path = Path(csv_path)
        self.run_name = run_name
        self.period_index = 0
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

    def log_completed_tax_period(self, step: int, tax_component: Any, env: Any) -> bool:
        """Log the latest completed tax period if Foundation produced tax info.

        Returns True when rows were written. The tax component appends an empty list
        on non-tax-collection steps and a dict on collection steps.
        """
        if not getattr(tax_component, "taxes", None):
            return False

        tax_info = tax_component.taxes[-1]
        if not isinstance(tax_info, dict) or "schedule" not in tax_info:
            return False

        self.period_index += 1
        period = int(getattr(tax_component, "period", 0) or 0)
        period_start_step = max(0, step - period + 1) if period else ""
        schedule = _to_float_list(tax_info.get("schedule", []))
        cutoffs = _to_float_list(tax_info.get("cutoffs", []))

        agent_by_id = {str(agent.idx): agent for agent in env.world.agents}
        rows: list[dict[str, Any]] = []
        for agent_id, agent in sorted(agent_by_id.items(), key=lambda x: int(x[0])):
            record = tax_info.get(agent_id, {})
            income = float(record.get("income", 0.0))
            rows.append({
                "run_name": self.run_name,
                "period_index": self.period_index,
                "period_start_step": period_start_step,
                "period_end_step": step,
                "agent_id": agent_id,
                "income": income,
                "income_per_step": income / period if period else "",
                "marginal_rate": float(record.get("marginal_rate", 0.0)),
                "effective_rate": float(record.get("effective_rate", 0.0)),
                "tax_paid": float(record.get("tax_paid", 0.0)),
                "lump_sum": float(record.get("lump_sum", 0.0)),
                "coin_after_tax": float(agent.inventory.get("Coin", 0.0)),
                "total_coin_endowment_after_tax": float(agent.total_endowment("Coin")),
                "schedule_json": json.dumps(schedule),
                "bracket_cutoffs_json": json.dumps(cutoffs),
            })

        self._append_rows(rows)
        return True

    def _append_rows(self, rows: list[dict[str, Any]]) -> None:
        file_exists = self.csv_path.exists() and self.csv_path.stat().st_size > 0
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CALIBRATION_FIELDS)
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows)


def _to_float_list(value: Any) -> list[float]:
    if isinstance(value, np.ndarray):
        return [float(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return []
