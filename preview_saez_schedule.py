"""Preview the new calibration-based Saez schedule from calibration samples."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from llm_agent.saez_policy import (
    DEFAULT_ELASTICITY,
    build_saez_schedule_from_calibration,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview calibration-based Saez tax schedule",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("calibration_csv", type=Path)
    parser.add_argument("--elasticity", type=float, default=DEFAULT_ELASTICITY)
    parser.add_argument(
        "--income-filter",
        choices=["full", "nonnegative", "positive"],
        default="full",
        help=(
            "How to transform calibration incomes before estimating the Saez "
            "income distribution."
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preview = build_saez_schedule_from_calibration(
        calibration_csv=args.calibration_csv,
        elasticity=args.elasticity,
        income_filter=args.income_filter,
    )

    if args.out_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = args.calibration_csv.parent / f"saez_preview_{stamp}"
    else:
        out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    audit_path = out_dir / "saez_schedule_audit.json"
    with audit_path.open("w", encoding="utf-8") as f:
        json.dump(preview.to_dict(), f, ensure_ascii=False, indent=2)

    schedule_path = out_dir / "saez_bracket_schedule.csv"
    with schedule_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "bracket_index",
                "bracket_cutoff",
                "raw_rate",
                "clipped_rate",
            ],
        )
        writer.writeheader()
        for i, (cutoff, raw_rate, clipped_rate) in enumerate(
            zip(
                preview.bracket_cutoffs,
                preview.bracket_rates,
                preview.clipped_bracket_rates,
            )
        ):
            writer.writerow({
                "bracket_index": i,
                "bracket_cutoff": cutoff,
                "raw_rate": raw_rate,
                "clipped_rate": clipped_rate,
            })

    binned_path = out_dir / "saez_binned_parameters.csv"
    with binned_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "bin_index",
                "bin_left_edge",
                "g_z",
                "alpha_z",
                "marginal_rate",
            ],
        )
        writer.writeheader()
        for i, (edge, gz, az, rate) in enumerate(
            zip(
                preview.bin_edges,
                preview.binned_gz,
                preview.binned_az,
                preview.binned_marginal_rates,
            )
        ):
            writer.writerow({
                "bin_index": i,
                "bin_left_edge": edge,
                "g_z": gz,
                "alpha_z": az,
                "marginal_rate": rate,
            })

    print("=" * 60)
    print("Saez Schedule Preview")
    print(f"Calibration CSV: {args.calibration_csv}")
    print(f"Rows: {preview.total_rows}")
    print(f"Income filter: {preview.income_filter}")
    print(f"Income rows used: {preview.calibration_income_rows_used}")
    print(f"Positive-income rows: {preview.positive_income_rows}")
    print(f"Elasticity: {preview.elasticity}")
    print(f"Welfare weights: {preview.welfare_weight_rule}")
    print(f"Bracket cutoffs: {preview.bracket_cutoffs}")
    print(f"Clipped bracket rates: {preview.clipped_bracket_rates}")
    print(f"Any rate clipped: {preview.any_rate_clipped}")
    print(f"Audit: {audit_path}")
    print(f"Schedule CSV: {schedule_path}")
    print(f"Binned CSV: {binned_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
