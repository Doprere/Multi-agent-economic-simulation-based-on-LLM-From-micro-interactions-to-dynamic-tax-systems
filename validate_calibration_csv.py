"""Validate random-tax calibration samples before calibrated Saez design."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

from llm_agent.calibration_logger import CALIBRATION_FIELDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate random-tax calibration CSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--min-samples", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.csv_path.exists():
        print(f"[FAIL] CSV not found: {args.csv_path}")
        sys.exit(1)

    with args.csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing_fields = [field for field in CALIBRATION_FIELDS if field not in (reader.fieldnames or [])]
        rows = list(reader)

    errors: list[str] = []
    if missing_fields:
        errors.append(f"missing fields: {missing_fields}")
    if len(rows) < args.min_samples:
        errors.append(f"sample rows below target: {len(rows)} < {args.min_samples}")
    if len(rows) % args.agents != 0:
        errors.append(f"row count is not divisible by agents={args.agents}: {len(rows)}")

    run_period_counts: Counter[tuple[str, str]] = Counter()
    positive_income = 0
    nonzero_rate = 0
    schedules: set[str] = set()

    for i, row in enumerate(rows, start=2):
        run_name = row.get("run_name", "")
        period_index = row.get("period_index", "")
        agent_id = row.get("agent_id", "")
        run_period_counts[(run_name, period_index)] += 1
        try:
            income = float(row.get("income", ""))
            marginal_rate = float(row.get("marginal_rate", ""))
            effective_rate = float(row.get("effective_rate", ""))
            tax_paid = float(row.get("tax_paid", ""))
            json.loads(row.get("schedule_json", "[]"))
            json.loads(row.get("bracket_cutoffs_json", "[]"))
        except Exception as exc:
            errors.append(f"row {i} parse failure: {exc}")
            continue

        if not agent_id:
            errors.append(f"row {i} empty agent_id")
        if marginal_rate < 0 or marginal_rate > 1:
            errors.append(f"row {i} marginal_rate out of [0,1]: {marginal_rate}")
        if effective_rate < 0:
            errors.append(f"row {i} negative effective_rate: {effective_rate}")
        if tax_paid < 0:
            errors.append(f"row {i} negative tax_paid: {tax_paid}")
        if income > 0:
            positive_income += 1
        if marginal_rate > 0:
            nonzero_rate += 1
        schedules.add(row.get("schedule_json", ""))

    bad_periods = [
        (run_name, period_index, count)
        for (run_name, period_index), count in run_period_counts.items()
        if count != args.agents
    ]
    if bad_periods:
        errors.append(f"periods with row count != {args.agents}: {bad_periods[:10]}")

    print("=" * 60)
    print("Calibration CSV Validation")
    print(f"CSV: {args.csv_path}")
    print(f"Rows: {len(rows)}")
    print(f"Run-periods: {len(run_period_counts)}")
    print(f"Positive-income rows: {positive_income}")
    print(f"Nonzero marginal-rate rows: {nonzero_rate}")
    print(f"Distinct schedules: {len(schedules)}")
    print("=" * 60)

    if errors:
        print("[FAIL]")
        for error in errors[:30]:
            print(f"- {error}")
        if len(errors) > 30:
            print(f"- ... {len(errors) - 30} more errors")
        sys.exit(1)

    print("[OK] Calibration CSV is structurally valid.")


if __name__ == "__main__":
    main()
