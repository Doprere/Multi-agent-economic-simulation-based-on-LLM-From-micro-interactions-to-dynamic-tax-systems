"""Calibration-based Saez utilities.

The formal `saez` baseline is calibration-seeded and dynamic: calibration rows
are loaded as Foundation's initial Saez buffer, then the Foundation component
updates the schedule every tax period during the formal run.

This module also provides an offline schedule preview so the initial schedule
implied by a calibration CSV can be audited before running simulations.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_ELASTICITY = 0.4
DEFAULT_BRACKET_CUTOFFS = [0.0, 9.7, 39.475, 84.2, 160.725, 204.1, 510.3]
DEFAULT_N_ESTIMATION_BINS = 100


@dataclass
class SaezSchedulePreview:
    calibration_csv: str
    total_rows: int
    finite_income_rows: int
    income_filter: str
    calibration_income_rows_used: int
    positive_income_rows: int
    finite_marginal_rate_rows: int
    elasticity: float
    welfare_weight_rule: str
    bracket_cutoffs: list[float]
    bracket_rates: list[float]
    clipped_bracket_rates: list[float]
    any_rate_clipped: bool
    bin_edges: list[float]
    binned_gz: list[float | None]
    binned_az: list[float | None]
    binned_marginal_rates: list[float | None]
    calibration_random_schedule_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_saez_schedule_from_calibration(
    calibration_csv: str | Path,
    elasticity: float = DEFAULT_ELASTICITY,
    welfare_weight_rule: str = "inverse_income",
    income_filter: str = "full",
    bracket_cutoffs: list[float] | None = None,
    n_estimation_bins: int = DEFAULT_N_ESTIMATION_BINS,
) -> SaezSchedulePreview:
    """Preview the initial Saez marginal tax schedule from calibration rows."""
    calibration_csv = Path(calibration_csv)
    rows, csv_cutoffs, random_schedules = _load_calibration_rows(calibration_csv)
    if bracket_cutoffs is None:
        bracket_cutoffs = csv_cutoffs or DEFAULT_BRACKET_CUTOFFS
    bracket_cutoffs = [float(v) for v in bracket_cutoffs]

    income_values = np.array([row["income"] for row in rows], dtype=float)
    marginal_rates = np.array([row["marginal_rate"] for row in rows], dtype=float)
    finite_income_mask = np.isfinite(income_values)
    finite_rate_mask = np.isfinite(marginal_rates)
    finite_incomes = income_values[finite_income_mask]
    filtered_incomes = _filter_incomes(finite_incomes, income_filter)

    if len(filtered_incomes) == 0:
        raise ValueError(
            f"Calibration CSV contains no income values after filter={income_filter!r}."
        )

    top_rate_cutoff = float(bracket_cutoffs[-1])
    bin_edges = np.linspace(0, top_rate_cutoff, n_estimation_bins + 1)
    bin_sizes = np.concatenate([bin_edges[1:] - bin_edges[:-1], [np.inf]])

    binned_gz, binned_az = _get_binned_saez_params(
        filtered_incomes,
        bin_edges,
        welfare_weight_rule=welfare_weight_rule,
    )
    binned_marginal_rates = _get_saez_marginal_rates(
        binned_gz,
        binned_az,
        elasticity,
    )
    bracket_rates = _bracketize_schedule(
        binned_marginal_rates,
        bin_edges,
        bin_sizes,
        bracket_cutoffs,
    )
    clipped_rates = np.clip(bracket_rates, 0.0, 1.0)

    return SaezSchedulePreview(
        calibration_csv=str(calibration_csv),
        total_rows=len(rows),
        finite_income_rows=int(np.sum(finite_income_mask)),
        income_filter=income_filter,
        calibration_income_rows_used=int(len(filtered_incomes)),
        positive_income_rows=int(np.sum(finite_incomes > 0)),
        finite_marginal_rate_rows=int(np.sum(finite_rate_mask)),
        elasticity=float(elasticity),
        welfare_weight_rule=welfare_weight_rule,
        bracket_cutoffs=[float(v) for v in bracket_cutoffs],
        bracket_rates=_finite_or_none_list(bracket_rates),
        clipped_bracket_rates=_finite_or_none_list(clipped_rates),
        any_rate_clipped=bool(np.any(np.abs(bracket_rates - clipped_rates) > 1e-12)),
        bin_edges=[float(v) for v in bin_edges],
        binned_gz=_finite_or_none_list(binned_gz),
        binned_az=_finite_or_none_list(binned_az),
        binned_marginal_rates=_finite_or_none_list(binned_marginal_rates),
        calibration_random_schedule_count=len(random_schedules),
    )


def load_saez_buffer_from_calibration(
    calibration_csv: str | Path,
    income_filter: str = "full",
) -> list[list[float]]:
    """Load calibration samples as Foundation-compatible Saez buffer rows.

    Foundation expects rows shaped as [income, marginal_rate]. The income filter
    is applied consistently with schedule preview sensitivity checks.
    """
    calibration_csv = Path(calibration_csv)
    rows, _csv_cutoffs, _random_schedules = _load_calibration_rows(calibration_csv)
    incomes = np.array([row["income"] for row in rows], dtype=float)
    marginal_rates = np.array([row["marginal_rate"] for row in rows], dtype=float)
    finite_mask = np.isfinite(incomes) & np.isfinite(marginal_rates)
    incomes = incomes[finite_mask]
    marginal_rates = marginal_rates[finite_mask]

    if income_filter == "full":
        filtered_incomes = incomes
        filtered_rates = marginal_rates
    elif income_filter == "nonnegative":
        filtered_incomes = np.maximum(incomes, 0.0)
        filtered_rates = marginal_rates
    elif income_filter == "positive":
        positive_mask = incomes > 0
        filtered_incomes = incomes[positive_mask]
        filtered_rates = marginal_rates[positive_mask]
    else:
        raise ValueError(
            "income_filter must be one of: full, nonnegative, positive"
        )

    return [
        [float(income), float(rate)]
        for income, rate in zip(filtered_incomes, filtered_rates)
    ]


def _filter_incomes(incomes: np.ndarray, income_filter: str) -> np.ndarray:
    if income_filter == "full":
        return incomes
    if income_filter == "nonnegative":
        return np.maximum(incomes, 0.0)
    if income_filter == "positive":
        return incomes[incomes > 0]
    raise ValueError(
        "income_filter must be one of: full, nonnegative, positive"
    )


def _load_calibration_rows(
    calibration_csv: Path,
) -> tuple[list[dict[str, float]], list[float] | None, set[str]]:
    rows: list[dict[str, float]] = []
    bracket_cutoffs: list[float] | None = None
    random_schedules: set[str] = set()

    with calibration_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_number, row in enumerate(reader, start=2):
            try:
                income = float(row["income"])
                marginal_rate = float(row["marginal_rate"])
            except Exception as exc:
                raise ValueError(f"Invalid calibration row {row_number}: {exc}") from exc

            rows.append({"income": income, "marginal_rate": marginal_rate})

            schedule_json = row.get("schedule_json", "")
            if schedule_json:
                random_schedules.add(schedule_json)

            if bracket_cutoffs is None and row.get("bracket_cutoffs_json"):
                try:
                    bracket_cutoffs = [
                        float(v) for v in json.loads(row["bracket_cutoffs_json"])
                    ]
                except Exception as exc:
                    raise ValueError(
                        f"Invalid bracket_cutoffs_json at row {row_number}: {exc}"
                    ) from exc

    if not rows:
        raise ValueError(f"No calibration rows found in {calibration_csv}.")
    return rows, bracket_cutoffs, random_schedules


def _get_binned_saez_params(
    population_incomes: np.ndarray,
    bin_edges: np.ndarray,
    welfare_weight_rule: str,
) -> tuple[np.ndarray, np.ndarray]:
    def clip(x: float, lo: float | None = None, hi: float | None = None) -> float:
        if lo is not None:
            x = max(lo, x)
        if hi is not None:
            x = min(x, hi)
        return x

    def bin_z(left: np.ndarray | float, right: np.ndarray | float) -> np.ndarray | float:
        return 0.5 * (left + right)

    def pareto(z: np.ndarray) -> np.ndarray:
        if welfare_weight_rule == "uniform":
            return np.ones_like(z)
        if welfare_weight_rule == "inverse_income":
            return 1.0 / np.maximum(1, z)
        raise ValueError(f"Unsupported welfare_weight_rule: {welfare_weight_rule}")

    counts, lefts = np.histogram(population_incomes, bins=bin_edges)
    incomes_below = population_incomes[population_incomes < lefts[0]]
    incomes_above = population_incomes[population_incomes > lefts[-1]]

    n_below = len(incomes_below)
    n_above = len(incomes_above)
    n_total = np.sum(counts) + n_below + n_above
    if n_total <= 0:
        raise ValueError("No incomes available to compute Saez distribution.")

    p_below = n_below / n_total
    pz = np.array([count / n_total for count in counts] + [n_above / n_total])

    cum_pz = [pz[0] + p_below]
    for p in pz[1:]:
        cum_pz.append(clip(cum_pz[-1] + p, 0, 1.0))
    cum_pz_arr = np.array(cum_pz)

    if len(incomes_below) > 0:
        pareto_weight_below = np.sum(pareto(np.maximum(incomes_below, 0)))
    else:
        pareto_weight_below = 0.0
    if len(incomes_above) > 0:
        pareto_weight_above = np.sum(pareto(incomes_above))
    else:
        pareto_weight_above = 0.0

    pareto_weight_per_bin = counts * pareto(bin_z(lefts[:-1], lefts[1:]))
    pareto_norm = (
        pareto_weight_per_bin.sum()
        + pareto_weight_below
        + pareto_weight_above
        + 1e-9
    )
    normalized_pareto_density = np.concatenate(
        [pareto_weight_per_bin, [pareto_weight_above]]
    ) / pareto_norm
    cumulative_pareto_density_geq_z = np.cumsum(
        normalized_pareto_density[::-1]
    )[::-1]
    cumulative_prob_geq_z = np.cumsum(pz[::-1])[::-1]
    avg_pareto_weight_geq_z = cumulative_pareto_density_geq_z / (
        cumulative_prob_geq_z + 1e-9
    )

    gz_at_left_edge = avg_pareto_weight_geq_z[:-1]
    gz_at_right_edge = avg_pareto_weight_geq_z[1:]
    binned_gz = np.concatenate([
        0.5 * (gz_at_left_edge + gz_at_right_edge),
        [avg_pareto_weight_geq_z[-1]],
    ])

    p_geq_z = 1 - cum_pz_arr + (0.5 * pz)
    az_values: list[float] = []
    for i in range(len(lefts[:-1])):
        if pz[i] == 0:
            az_values.append(np.nan)
        else:
            z = float(bin_z(lefts[i], lefts[i + 1]))
            az = z * pz[i] / (clip(float(p_geq_z[i]), 0, 1) + 1e-9)
            az = az / (lefts[i + 1] - lefts[i])
            az_values.append(float(az))

    if len(incomes_above) > 0:
        cutoff = lefts[-1]
        avg_income_above_cutoff = float(np.mean(incomes_above))
        az_above = avg_income_above_cutoff / (
            avg_income_above_cutoff - cutoff + 1e-9
        )
    else:
        az_above = 0.0

    binned_az = np.concatenate([np.array(az_values), [az_above]])
    return binned_gz, binned_az


def _get_saez_marginal_rates(
    binned_gz: np.ndarray,
    binned_az: np.ndarray,
    elasticity: float,
) -> np.ndarray:
    rates = (1.0 - binned_gz) / (
        1.0 - binned_gz + binned_az * elasticity + 1e-9
    )
    rates = np.array(rates, dtype=float)

    real_indices = np.where(np.isfinite(rates))[0]
    if len(real_indices) == 0:
        return np.zeros_like(rates)

    first_real = int(real_indices[0])
    if first_real > 0:
        rates[:first_real] = np.linspace(0.0, rates[first_real], first_real + 1)[:-1]

    last_real = first_real
    for idx in real_indices[1:]:
        idx = int(idx)
        if idx - last_real > 1:
            gap_indices = np.arange(last_real + 1, idx)
            rates[gap_indices] = np.linspace(
                rates[last_real],
                rates[idx],
                len(gap_indices) + 2,
            )[1:-1]
        last_real = idx

    if last_real < len(rates) - 1:
        rates[last_real + 1 :] = rates[last_real]

    return rates


def _bracketize_schedule(
    bin_marginal_rates: np.ndarray,
    bin_edges: np.ndarray,
    bin_sizes: np.ndarray,
    bracket_cutoffs: list[float],
) -> np.ndarray:
    bracket_edges = np.concatenate([np.array(bracket_cutoffs), [np.inf]])
    bracket_sizes = bracket_edges[1:] - bracket_edges[:-1]

    last_bracket_total = 0.0
    bracket_avg_marginal_rates: list[float] = []
    for b_idx, income in enumerate(bracket_cutoffs[1:]):
        past_cutoff = np.maximum(0, income - bin_edges)
        bin_income = np.minimum(bin_sizes, past_cutoff)
        bin_taxes = bin_marginal_rates * bin_income
        taxes_due = np.maximum(0, np.sum(bin_taxes))
        bracket_tax_burden = taxes_due - last_bracket_total
        bracket_avg_marginal_rates.append(
            float(bracket_tax_burden / bracket_sizes[b_idx])
        )
        last_bracket_total = taxes_due

    bracket_avg_marginal_rates.append(float(bin_marginal_rates[-1]))
    bracket_rates = np.array(bracket_avg_marginal_rates)
    if len(bracket_rates) != len(bracket_cutoffs):
        raise AssertionError("Bracketized Saez schedule length mismatch.")
    return bracket_rates


def _finite_or_none_list(values: np.ndarray) -> list[float | None]:
    out: list[float | None] = []
    for value in values:
        if np.isfinite(value):
            out.append(float(value))
        else:
            out.append(None)
    return out
