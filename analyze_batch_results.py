"""Batch analysis utilities for 100-run thesis experiments.

Stage 1 creates a transparent run-level summary table from raw simulation
outputs. It does not modify the source result files.

Example:
    python analyze_batch_results.py --batch simulation_results/20260429_120220
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_BATCH = Path("simulation_results/20260429_120220")
RUN_NUM_RE = re.compile(r"_run(\d+)$")
EXPECTED_STEPS = 1000
AGENT_COUNT = 4
GPT4O_MINI_INPUT_PER_M = 0.15
GPT4O_MINI_OUTPUT_PER_M = 0.60
REQUIRED_STEP_COLUMNS = [
    "gini",
    "mean_coin",
    "total_coin",
    "swf_absolute",
    "planner_reward",
]
PRIMARY_METRICS = [
    "final_gini",
    "mean_gini",
    "gini_auc_per_step",
    "final_mean_coin",
    "mean_coin_over_time",
    "completed_build_count",
    "final_swf_absolute",
    "mean_swf_absolute",
    "cumulative_planner_reward",
    "coin_per_labor",
    "builds_per_labor",
    "fallback_count",
    "illegal_action_warning_count",
    "noop_action_share",
    "build_action_share",
    "order_action_share",
    "move_action_share",
    "mean_tax_action_value",
    "tax_action_volatility",
]
ROBUSTNESS_METRICS = [
    "mean_gini",
    "gini_auc_per_step",
    "final_gini",
    "mean_coin_over_time",
    "final_mean_coin",
    "completed_build_count",
    "mean_swf_absolute",
    "final_swf_absolute",
    "cumulative_planner_reward",
    "noop_action_share",
    "build_action_share",
    "order_action_share",
    "move_action_share",
]
COMPARISON_GROUPS = ("gpt4omini", "random_tax")
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 20260512
METRIC_DISPLAY_LABELS = {
    "final_swf_absolute": "Final Social Welfare",
    "mean_swf_absolute": "Mean Social Welfare",
    "swf_absolute": "Social Welfare",
    "cumulative_planner_reward": "Cumulative SWF Gain",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create run-level summaries for thesis batch experiments."
    )
    parser.add_argument(
        "--batch",
        type=Path,
        default=DEFAULT_BATCH,
        help=f"Batch result directory (default: {DEFAULT_BATCH})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: <batch>/_analysis)",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=BOOTSTRAP_ITERATIONS,
        help=f"Bootstrap iterations for CIs (default: {BOOTSTRAP_ITERATIONS})",
    )
    return parser.parse_args()


def infer_group(run_name: str) -> str:
    if "main_gpt4omini" in run_name:
        return "gpt4omini"
    if "main_random_tax" in run_name or "main_saez" in run_name:
        return "random_tax"
    return "unknown"


def metric_display_label(metric: str) -> str:
    """Human-readable label for figures while preserving raw CSV column names."""
    return METRIC_DISPLAY_LABELS.get(metric, metric.replace("_", " ").title())


def expected_runs_from_batch(batch: Path) -> list[str]:
    """Return one row target per observed run directory or batch log.

    This avoids survivorship bias from discovering only completed runs with
    summary.json. If a run crashed before writing summary.json but left a log or
    directory, it still appears in the run-level table with read errors.
    """
    names: set[str] = set()
    for child in batch.iterdir() if batch.exists() else []:
        if child.is_dir() and not child.name.startswith("_"):
            names.add(child.name)
    log_dir = batch / "_batch_logs"
    if log_dir.exists():
        for log_path in log_dir.glob("*.log"):
            names.add(log_path.stem)
    return sorted(names, key=lambda n: (infer_group(n), infer_run_num(n) or 10**9, n))


def infer_run_num(run_name: str) -> int | None:
    match = RUN_NUM_RE.search(run_name)
    return int(match.group(1)) if match else None


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_text_if_exists(path: Path) -> tuple[str, str | None]:
    if not path.exists():
        return "", "missing"
    try:
        return path.read_text(encoding="utf-8", errors="ignore"), None
    except OSError as exc:
        return "", f"{type(exc).__name__}: {exc}"


def read_json_if_exists(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, "missing"
    try:
        return load_json(path), None
    except (json.JSONDecodeError, OSError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def read_csv_if_exists(path: Path) -> tuple[pd.DataFrame | None, str | None]:
    if not path.exists():
        return None, "missing"
    try:
        return pd.read_csv(path), None
    except Exception as exc:  # pandas can raise parser, unicode, and IO errors.
        return None, f"{type(exc).__name__}: {exc}"


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def trapezoid_auc(series: pd.Series) -> float:
    """Area under a per-step series using step index spacing."""
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if len(values) == 0:
        return float("nan")
    return float(np.trapezoid(values))


def column_error(df: pd.DataFrame | None, required: list[str]) -> str | None:
    if df is None:
        return None
    missing = [col for col in required if col not in df.columns]
    return f"missing columns: {', '.join(missing)}" if missing else None


def series_mean(df: pd.DataFrame | None, col: str) -> float | None:
    return float(df[col].mean()) if df is not None and col in df.columns else None


def series_max(df: pd.DataFrame | None, col: str) -> float | None:
    return float(df[col].max()) if df is not None and col in df.columns else None


def series_min(df: pd.DataFrame | None, col: str) -> float | None:
    return float(df[col].min()) if df is not None and col in df.columns else None


def series_auc(df: pd.DataFrame | None, col: str) -> float | None:
    return trapezoid_auc(df[col]) if df is not None and col in df.columns else None


def action_category_counts(action_df: pd.DataFrame) -> dict[str, int]:
    counts = {
        "noop_action_count": 0,
        "build_action_count": 0,
        "order_action_count": 0,
        "move_action_count": 0,
        "action_unknown_count": 0,
    }
    action_cols = [c for c in action_df.columns if c.startswith("action_agent_")]
    for col in action_cols:
        actions = pd.to_numeric(action_df[col], errors="coerce")
        counts["noop_action_count"] += int((actions == 0).sum())
        counts["build_action_count"] += int((actions == 1).sum())
        counts["order_action_count"] += int(((actions >= 2) & (actions <= 45)).sum())
        counts["move_action_count"] += int(((actions >= 46) & (actions <= 49)).sum())
        counts["action_unknown_count"] += int(actions.isna().sum())
    return counts


def log_quality_counts(log_text: str) -> dict[str, int | bool]:
    return {
        "has_end_marker": "[End]" in log_text,
        "traceback_count": log_text.count("Traceback"),
        "timeout_count": len(re.findall(r"timeout|timed out", log_text, flags=re.I)),
        "http_non_200_count": len(
            re.findall(r'HTTP/1\.1"\s+(?!200\b)\d+', log_text)
        ),
        "fallback_count": len(
            re.findall(
                r"all retries failed, fallback to random action",
                log_text,
                flags=re.I,
            )
        ),
        "fallback_thought_count": len(
            re.findall(r"LLM retries exhausted", log_text, flags=re.I)
        ),
        "illegal_action_warning_count": len(
            re.findall(r"illegal action_id=", log_text)
        ),
        "warning_count": len(re.findall(r"\[WARNING\]", log_text)),
        "error_count": len(re.findall(r"\[ERROR\]", log_text)),
    }


def token_usage_fields(summary: dict[str, Any]) -> dict[str, Any]:
    token_usage = summary.get("token_usage") or {}
    agents = token_usage.get("agents") or {}
    planner = token_usage.get("planner") or {}
    has_token_usage = bool(token_usage)
    return {
        "token_usage_present": has_token_usage,
        "agent_backend": agents.get("backend"),
        "agent_model": agents.get("model"),
        "agent_api_calls": agents.get("api_calls"),
        "agent_prompt_tokens": agents.get("prompt_tokens"),
        "agent_completion_tokens": agents.get("completion_tokens"),
        "agent_total_tokens": agents.get("total_tokens"),
        "planner_backend": planner.get("backend"),
        "planner_model": planner.get("model"),
        "planner_api_calls": planner.get("api_calls"),
        "planner_prompt_tokens": planner.get("prompt_tokens"),
        "planner_completion_tokens": planner.get("completion_tokens"),
        "planner_total_tokens": planner.get("total_tokens"),
    }


def metadata_from_name(run_name: str) -> dict[str, Any]:
    group = infer_group(run_name)
    if group == "gpt4omini":
        return {
            "condition": "gemma4:e2b_agents__gpt4o-mini_planner",
            "agent_backend": "ollama",
            "agent_model": "gemma4:e2b",
            "planner_backend": "openai",
            "planner_model": "gpt-4o-mini",
            "planner_is_rule_based": False,
        }
    if group == "random_tax":
        return {
            "condition": "gemma4:e2b_agents__random-tax_baseline",
            "agent_backend": "ollama",
            "agent_model": "gemma4:e2b",
            "planner_backend": "random-tax",
            "planner_model": "random-tax",
            "planner_is_rule_based": True,
        }
    return {
        "condition": "unknown",
        "planner_is_rule_based": None,
    }


def gpt4o_mini_cost(prompt_tokens: Any, completion_tokens: Any) -> float | None:
    """Estimate USD cost with user-provided gpt-4o-mini pricing.

    Pricing:
      input: 0.15 USD / 1M tokens
      output: 0.60 USD / 1M tokens
    Cached input is not estimated because cached-token counts are not logged.
    """
    prompt = safe_float(prompt_tokens)
    completion = safe_float(completion_tokens)
    if prompt is None or completion is None:
        return None
    return (
        prompt / 1_000_000 * GPT4O_MINI_INPUT_PER_M
        + completion / 1_000_000 * GPT4O_MINI_OUTPUT_PER_M
    )


def parse_elapsed_seconds(log_text: str) -> float | None:
    match = re.search(r"real\s+(\d+)m([\d.]+)s", log_text)
    if not match:
        return None
    return int(match.group(1)) * 60 + float(match.group(2))


def tax_summary_fields(tax_df: pd.DataFrame | None) -> dict[str, Any]:
    if tax_df is None or tax_df.empty or "tax_brackets" not in tax_df.columns:
        return {
            "tax_schedule_changes": None,
            "mean_tax_action_value": None,
            "mean_top_tax_action_value": None,
            "final_top_tax_action_value": None,
            "tax_action_volatility": None,
        }
    schedules: list[list[float]] = []
    for value in tax_df["tax_brackets"]:
        try:
            parsed = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            try:
                import ast

                parsed = ast.literal_eval(str(value))
            except (ValueError, SyntaxError):
                continue
        if isinstance(parsed, list):
            rates = []
            for item in parsed:
                if isinstance(item, dict):
                    rate = item.get("rate")
                elif isinstance(item, (list, tuple)) and item:
                    rate = item[-1]
                else:
                    rate = item
                num = safe_float(rate)
                if num is not None:
                    rates.append(num)
            if rates:
                schedules.append(rates)
    if not schedules:
        return {
            "tax_schedule_changes": None,
            "mean_tax_action_value": None,
            "mean_top_tax_action_value": None,
            "final_top_tax_action_value": None,
            "tax_action_volatility": None,
        }
    max_len = max(len(s) for s in schedules)
    arr = np.full((len(schedules), max_len), np.nan)
    for i, rates in enumerate(schedules):
        arr[i, : len(rates)] = rates
    changes = sum(schedules[i] != schedules[i - 1] for i in range(1, len(schedules)))
    return {
        "tax_schedule_changes": int(changes),
        "mean_tax_action_value": float(np.nanmean(arr)),
        "mean_top_tax_action_value": float(np.nanmean(arr[:, -1])),
        "final_top_tax_action_value": float(schedules[-1][-1]),
        "tax_action_volatility": float(np.nanmean(np.nanstd(arr, axis=0))),
    }


def build_run_row(run_name: str, batch_dir: Path) -> dict[str, Any]:
    run_dir = batch_dir / run_name
    summary_path = run_dir / "summary.json"
    step_path = run_dir / "step_metrics.csv"
    action_path = run_dir / "action_log.csv"
    tax_path = run_dir / "tax_log.csv"
    log_path = batch_dir / "_batch_logs" / f"{run_name}.log"

    summary, summary_error = read_json_if_exists(summary_path)
    step_df, step_error = read_csv_if_exists(step_path)
    action_df, action_error = read_csv_if_exists(action_path)
    tax_df, tax_error = read_csv_if_exists(tax_path)
    log_text, log_error = read_text_if_exists(log_path)
    step_column_error = column_error(step_df, REQUIRED_STEP_COLUMNS)

    summary = summary or {}
    final = summary.get("final_metrics") or {}
    cumulative = summary.get("cumulative_metrics") or {}
    numeric = (
        step_df.select_dtypes(include=[np.number])
        if step_df is not None
        else pd.DataFrame()
    )

    total_steps = summary.get("total_steps")
    step_rows = len(step_df) if step_df is not None else 0
    action_rows = len(action_df) if action_df is not None else 0
    tax_rows = len(tax_df) if tax_df is not None else 0
    final_step = final.get("step")

    row: dict[str, Any] = {
        "run_name": run_name,
        "group": infer_group(run_name),
        "condition": metadata_from_name(run_name)["condition"],
        "run_num": infer_run_num(run_name),
        "expected_steps": EXPECTED_STEPS,
        "total_steps": total_steps,
        "step_rows": step_rows,
        "action_rows": action_rows,
        "tax_rows": tax_rows,
        "tax_count": summary.get("tax_count"),
        "has_run_dir": run_dir.exists(),
        "has_summary_json": summary_path.exists(),
        "has_step_metrics_csv": step_path.exists(),
        "has_action_log_csv": action_path.exists(),
        "has_tax_log_csv": tax_path.exists(),
        "has_agent_thoughts_xlsx": (run_dir / "agent_thoughts.xlsx").exists(),
        "has_log": log_path.exists(),
        "summary_read_error": summary_error,
        "step_metrics_read_error": step_error,
        "step_metrics_column_error": step_column_error,
        "action_log_read_error": action_error,
        "tax_log_read_error": tax_error,
        "log_read_error": log_error,
        "nan_count": int(numeric.isna().sum().sum()),
        "inf_count": int(np.isinf(numeric.to_numpy()).sum()) if not numeric.empty else 0,
        "final_step": final_step,
        "final_step_matches_expected": final_step == EXPECTED_STEPS - 1,
        "step_rows_match_expected": step_rows == EXPECTED_STEPS,
        "action_rows_match_expected": action_rows == EXPECTED_STEPS,
        "tax_rows_match_expected": tax_rows == 10,
        "is_complete": (
            summary_error is None
            and step_error is None
            and step_column_error is None
            and action_error is None
            and tax_error is None
            and total_steps == EXPECTED_STEPS
            and step_rows == EXPECTED_STEPS
            and action_rows == EXPECTED_STEPS
            and tax_rows == 10
            and final_step == EXPECTED_STEPS - 1
        ),
        "has_interpretability_artifacts": (
            log_path.exists()
            and log_error is None
            and "[End]" in log_text
            and (run_dir / "agent_thoughts.xlsx").exists()
        ),
        "final_gini": final.get("gini"),
        "mean_gini": series_mean(step_df, "gini"),
        "max_gini": series_max(step_df, "gini"),
        "min_gini": series_min(step_df, "gini"),
        "gini_auc": series_auc(step_df, "gini"),
        "gini_auc_per_step": (
            series_auc(step_df, "gini") / max(step_rows - 1, 1)
            if series_auc(step_df, "gini") is not None
            else None
        ),
        "final_mean_coin": final.get("mean_coin"),
        "mean_coin_over_time": series_mean(step_df, "mean_coin"),
        "max_mean_coin": series_max(step_df, "mean_coin"),
        "final_total_coin": final.get("total_coin"),
        "mean_total_coin_over_time": series_mean(step_df, "total_coin"),
        "build_total": cumulative.get("build_total"),
        "completed_build_count": cumulative.get("build_total"),
        "final_swf_absolute": final.get("swf_absolute"),
        "mean_swf_absolute": series_mean(step_df, "swf_absolute"),
        "max_swf_absolute": series_max(step_df, "swf_absolute"),
        "final_planner_reward": final.get("planner_reward"),
        "mean_planner_reward": series_mean(step_df, "planner_reward"),
        "cumulative_planner_reward": final.get("cumulative_planner_reward"),
        "duration_seconds": parse_elapsed_seconds(log_text),
    }
    row["analysis_ready"] = row["is_complete"] and row["has_interpretability_artifacts"]

    for agent_id in range(4):
        row[f"final_coin_agent_{agent_id}"] = final.get(f"coin_agent_{agent_id}")
        row[f"final_labor_agent_{agent_id}"] = final.get(f"labor_agent_{agent_id}")
        row[f"final_wood_agent_{agent_id}"] = final.get(f"wood_agent_{agent_id}")
        row[f"final_stone_agent_{agent_id}"] = final.get(f"stone_agent_{agent_id}")
        row[f"build_agent_{agent_id}"] = cumulative.get(f"build_agent_{agent_id}")
        reward_col = f"reward_agent_{agent_id}"
        labor_col = f"labor_agent_{agent_id}"
        if step_df is not None and reward_col in step_df.columns:
            row[f"mean_reward_agent_{agent_id}"] = float(step_df[reward_col].mean())
            row[f"cumulative_reward_agent_{agent_id}"] = float(step_df[reward_col].sum())
        if step_df is not None and labor_col in step_df.columns:
            row[f"mean_labor_agent_{agent_id}"] = float(step_df[labor_col].mean())

    if action_df is not None:
        row.update(action_category_counts(action_df))
    else:
        row.update(
            {
                "noop_action_count": None,
                "build_action_count": None,
                "order_action_count": None,
                "move_action_count": None,
                "action_unknown_count": None,
            }
        )

    row.update(log_quality_counts(log_text))

    token_fields = token_usage_fields(summary)
    inferred = metadata_from_name(run_name)
    for key, value in inferred.items():
        row[key] = value if row.get(key) is None else row.get(key)
    for key, value in token_fields.items():
        row[key] = value if value is not None else row.get(key)

    row["estimated_planner_cost_usd"] = gpt4o_mini_cost(
        row.get("planner_prompt_tokens"), row.get("planner_completion_tokens")
    )

    agent_steps = action_rows * AGENT_COUNT if action_rows else None
    for count_col, share_col in [
        ("noop_action_count", "noop_action_share"),
        ("build_action_count", "build_action_share"),
        ("order_action_count", "order_action_share"),
        ("move_action_count", "move_action_share"),
    ]:
        row[share_col] = (
            row[count_col] / agent_steps
            if agent_steps and row.get(count_col) is not None
            else None
        )

    labor_values = [
        safe_float(row.get(f"final_labor_agent_{agent_id}"))
        for agent_id in range(AGENT_COUNT)
    ]
    total_labor = sum(labor_values) if all(v is not None for v in labor_values) else None
    row["final_total_labor"] = total_labor
    row["coin_per_labor"] = (
        row["final_total_coin"] / total_labor
        if total_labor and row.get("final_total_coin") is not None
        else None
    )
    row["builds_per_labor"] = (
        row["completed_build_count"] / total_labor
        if total_labor and row.get("completed_build_count") is not None
        else None
    )
    row.update(tax_summary_fields(tax_df))
    return row


def write_methods(out_dir: Path, batch: Path, run_count: int) -> None:
    methods = f"""# Stage 1 Run-Level Summary Methods

Generated: {datetime.now(timezone.utc).isoformat()}

## Input

- Batch directory: `{batch}`
- Runs included: `{run_count}`
- Each run is expected to contain `summary.json`, `step_metrics.csv`, `action_log.csv`, `tax_log.csv`, and a matching `_batch_logs/*.log`.

## Run-Level Metrics

- The unit of analysis is every observed run directory or `_batch_logs/*.log`, not only successful runs with `summary.json`.
- Final metrics are read from `summary.json.final_metrics` when available.
- Completed build metrics are read from `summary.json.cumulative_metrics`.
- Mean trajectory metrics are arithmetic means across the 1000 rows of `step_metrics.csv`.
- `gini_auc` is the raw trapezoidal area under the per-step Gini trajectory.
- `gini_auc_per_step` normalizes that area by observed step intervals.
- Action categories are selected action IDs from `action_log.csv`: NOOP=0, Build=1, order placement=2-45, Move=46-49. They should not be interpreted as completed trades.
- Action shares use denominator `action_rows * 4 agents`.

## Data Quality Metrics

- `fallback_count` counts log lines matching `all retries failed, fallback to random action`.
- `illegal_action_warning_count` counts log lines containing `illegal action_id=`.
- `timeout_count` counts `timeout` or `timed out` in logs.
- `http_non_200_count` counts HTTP response log lines with status other than 200.
- `traceback_count` counts Python traceback occurrences.
- `is_complete` requires readable required files, 1000 step/action rows, 10 tax rows, and final step 999.
- `has_interpretability_artifacts` requires a readable log with `[End]` and `agent_thoughts.xlsx`.
- `analysis_ready` requires both `is_complete` and `has_interpretability_artifacts`.
- No run is excluded in Stage 1.

## Cost Estimate

- `estimated_planner_cost_usd` is computed only when planner token usage is available.
- GPT-4o-mini pricing used: input USD 0.15 / 1M tokens, output USD 0.60 / 1M tokens.
- Cached input is not estimated because cached-token counts are not logged.
- Random-tax baseline runs have no OpenAI planner cost.

## Tax Policy Variables

- `tax_*_action_value` fields summarize the numeric values stored in `tax_log.csv`.
- These values are treated as planner tax action/schedule values, not necessarily literal percentage tax rates.
"""
    (out_dir / "run_level_methods.md").write_text(methods, encoding="utf-8")


def bootstrap_ci(
    values: pd.Series,
    iterations: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> tuple[float | None, float | None]:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return None, None
    sample_indices = rng.integers(0, len(arr), size=(iterations, len(arr)))
    means = arr[sample_indices].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def bootstrap_diff_ci(
    a: pd.Series,
    b: pd.Series,
    iterations: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> tuple[float | None, float | None]:
    arr_a = pd.to_numeric(a, errors="coerce").dropna().to_numpy(dtype=float)
    arr_b = pd.to_numeric(b, errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr_a) == 0 or len(arr_b) == 0:
        return None, None
    idx_a = rng.integers(0, len(arr_a), size=(iterations, len(arr_a)))
    idx_b = rng.integers(0, len(arr_b), size=(iterations, len(arr_b)))
    diffs = arr_a[idx_a].mean(axis=1) - arr_b[idx_b].mean(axis=1)
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def bootstrap_median_diff_ci(
    a: pd.Series,
    b: pd.Series,
    iterations: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> tuple[float | None, float | None]:
    arr_a = pd.to_numeric(a, errors="coerce").dropna().to_numpy(dtype=float)
    arr_b = pd.to_numeric(b, errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr_a) == 0 or len(arr_b) == 0:
        return None, None
    idx_a = rng.integers(0, len(arr_a), size=(iterations, len(arr_a)))
    idx_b = rng.integers(0, len(arr_b), size=(iterations, len(arr_b)))
    diffs = np.median(arr_a[idx_a], axis=1) - np.median(arr_b[idx_b], axis=1)
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def welch_t_test(a: pd.Series, b: pd.Series) -> dict[str, float | None]:
    arr_a = pd.to_numeric(a, errors="coerce").dropna().to_numpy(dtype=float)
    arr_b = pd.to_numeric(b, errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr_a) < 2 or len(arr_b) < 2:
        return {"welch_t": None, "welch_df": None, "welch_p": None}
    mean_a, mean_b = arr_a.mean(), arr_b.mean()
    var_a, var_b = arr_a.var(ddof=1), arr_b.var(ddof=1)
    se2 = var_a / len(arr_a) + var_b / len(arr_b)
    if se2 <= 0:
        return {"welch_t": None, "welch_df": None, "welch_p": None}
    t_stat = (mean_a - mean_b) / np.sqrt(se2)
    df_num = se2**2
    df_den = (var_a / len(arr_a)) ** 2 / (len(arr_a) - 1) + (
        var_b / len(arr_b)
    ) ** 2 / (len(arr_b) - 1)
    df = df_num / df_den if df_den else None
    # Normal approximation is adequate for n=100 per group and avoids adding
    # scipy as a hard dependency for the analysis script.
    p_value = 2 * (1 - normal_cdf(abs(float(t_stat))))
    return {"welch_t": float(t_stat), "welch_df": float(df), "welch_p": float(p_value)}


def rankdata_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def mann_whitney_u_test(a: pd.Series, b: pd.Series) -> dict[str, float | None]:
    arr_a = pd.to_numeric(a, errors="coerce").dropna().to_numpy(dtype=float)
    arr_b = pd.to_numeric(b, errors="coerce").dropna().to_numpy(dtype=float)
    n_a, n_b = len(arr_a), len(arr_b)
    if n_a == 0 or n_b == 0:
        return {"mann_whitney_u": None, "mann_whitney_p": None}
    combined = np.concatenate([arr_a, arr_b])
    ranks = rankdata_average(combined)
    rank_sum_a = ranks[:n_a].sum()
    u_a = rank_sum_a - n_a * (n_a + 1) / 2.0
    mean_u = n_a * n_b / 2.0
    _, tie_counts = np.unique(combined, return_counts=True)
    tie_term = np.sum(tie_counts**3 - tie_counts)
    total = n_a + n_b
    var_u = n_a * n_b / 12.0 * (
        total + 1 - tie_term / (total * (total - 1)) if total > 1 else 0
    )
    if var_u <= 0:
        return {"mann_whitney_u": float(u_a), "mann_whitney_p": None}
    z = (u_a - mean_u) / np.sqrt(var_u)
    p_value = 2 * (1 - normal_cdf(abs(float(z))))
    return {"mann_whitney_u": float(u_a), "mann_whitney_p": float(p_value)}


def cohens_d(a: pd.Series, b: pd.Series) -> float | None:
    arr_a = pd.to_numeric(a, errors="coerce").dropna().to_numpy(dtype=float)
    arr_b = pd.to_numeric(b, errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr_a) < 2 or len(arr_b) < 2:
        return None
    pooled_var = (
        (len(arr_a) - 1) * arr_a.var(ddof=1)
        + (len(arr_b) - 1) * arr_b.var(ddof=1)
    ) / (len(arr_a) + len(arr_b) - 2)
    if pooled_var <= 0:
        return None
    return float((arr_a.mean() - arr_b.mean()) / np.sqrt(pooled_var))


def cliffs_delta(a: pd.Series, b: pd.Series) -> float | None:
    arr_a = pd.to_numeric(a, errors="coerce").dropna().to_numpy(dtype=float)
    arr_b = pd.to_numeric(b, errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr_a) == 0 or len(arr_b) == 0:
        return None
    diff = arr_a[:, None] - arr_b[None, :]
    greater = np.sum(diff > 0)
    less = np.sum(diff < 0)
    return float((greater - less) / (len(arr_a) * len(arr_b)))


def create_group_level_outputs(
    run_df: pd.DataFrame,
    out_dir: Path,
    bootstrap_iterations: int,
    batch_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    analysis_df = run_df[run_df["analysis_ready"]].copy()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summary_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []

    for metric in PRIMARY_METRICS:
        if metric not in analysis_df.columns:
            continue
        for group, group_df in analysis_df.groupby("group"):
            values = pd.to_numeric(group_df[metric], errors="coerce").dropna()
            ci_lo, ci_hi = bootstrap_ci(values, bootstrap_iterations, rng)
            summary_rows.append(
                {
                    "metric": metric,
                    "group": group,
                    "n": int(values.count()),
                    "mean": float(values.mean()) if len(values) else None,
                    "std": float(values.std(ddof=1)) if len(values) > 1 else None,
                    "median": float(values.median()) if len(values) else None,
                    "iqr": float(values.quantile(0.75) - values.quantile(0.25))
                    if len(values)
                    else None,
                    "min": float(values.min()) if len(values) else None,
                    "max": float(values.max()) if len(values) else None,
                    "bootstrap_mean_ci_low": ci_lo,
                    "bootstrap_mean_ci_high": ci_hi,
                }
            )

        if all(g in set(analysis_df["group"]) for g in COMPARISON_GROUPS):
            a = analysis_df.loc[analysis_df["group"] == COMPARISON_GROUPS[0], metric]
            b = analysis_df.loc[analysis_df["group"] == COMPARISON_GROUPS[1], metric]
            arr_a = pd.to_numeric(a, errors="coerce").dropna()
            arr_b = pd.to_numeric(b, errors="coerce").dropna()
            diff_ci_lo, diff_ci_hi = bootstrap_diff_ci(
                arr_a, arr_b, bootstrap_iterations, rng
            )
            row = {
                "metric": metric,
                "group_a": COMPARISON_GROUPS[0],
                "group_b": COMPARISON_GROUPS[1],
                "n_a": int(arr_a.count()),
                "n_b": int(arr_b.count()),
                "full_group_n_a": int(
                    (run_df["group"] == COMPARISON_GROUPS[0]).sum()
                ),
                "full_group_n_b": int(
                    (run_df["group"] == COMPARISON_GROUPS[1]).sum()
                ),
                "analysis_ready_n_a": int(
                    (
                        (run_df["group"] == COMPARISON_GROUPS[0])
                        & (run_df["analysis_ready"])
                    ).sum()
                ),
                "analysis_ready_n_b": int(
                    (
                        (run_df["group"] == COMPARISON_GROUPS[1])
                        & (run_df["analysis_ready"])
                    ).sum()
                ),
                "valid_numeric_n_warning": int(arr_a.count()) != int(
                    (run_df["group"] == COMPARISON_GROUPS[0]).sum()
                )
                or int(arr_b.count()) != int(
                    (run_df["group"] == COMPARISON_GROUPS[1]).sum()
                ),
                "mean_a": float(arr_a.mean()) if len(arr_a) else None,
                "mean_b": float(arr_b.mean()) if len(arr_b) else None,
                "mean_diff_a_minus_b": float(arr_a.mean() - arr_b.mean())
                if len(arr_a) and len(arr_b)
                else None,
                "bootstrap_diff_ci_low": diff_ci_lo,
                "bootstrap_diff_ci_high": diff_ci_hi,
                "cohens_d": cohens_d(arr_a, arr_b),
                "cliffs_delta": cliffs_delta(arr_a, arr_b),
            }
            row.update(welch_t_test(arr_a, arr_b))
            row.update(mann_whitney_u_test(arr_a, arr_b))
            test_rows.append(row)

    group_summary = pd.DataFrame(summary_rows)
    tests = pd.DataFrame(test_rows)
    group_summary.to_csv(out_dir / "group_level_summary.csv", index=False, encoding="utf-8-sig")
    tests.to_csv(out_dir / "statistical_tests.csv", index=False, encoding="utf-8-sig")
    write_group_methods(out_dir, bootstrap_iterations)
    create_stage2_figures(analysis_df, out_dir)
    create_time_series_figures(analysis_df, batch_dir, out_dir)
    create_robustness_outputs(run_df, out_dir, bootstrap_iterations)
    return group_summary, tests


def write_group_methods(out_dir: Path, bootstrap_iterations: int) -> None:
    methods = f"""# Stage 2 Group-Level Comparison Methods

Generated: {datetime.now(timezone.utc).isoformat()}

## Included Runs

- Stage 2 includes rows where `analysis_ready == True`.
- `statistical_tests.csv` records full group n, analysis-ready n, and valid numeric n for each metric.
- No outlier or fallback-based exclusion is applied.
- Comparison groups are `gpt4omini` and `random_tax`.

## Descriptive Statistics

- `group_level_summary.csv` reports n, mean, standard deviation, median, IQR, min, max, and bootstrap 95% confidence intervals for group means.
- Bootstrap CIs use {bootstrap_iterations} resamples with replacement and fixed random seed `{BOOTSTRAP_SEED}`.

## Statistical Tests

- `statistical_tests.csv` reports GPT-4o-mini minus random-tax mean differences.
- Welch t-statistics are used for mean differences. P-values use a normal approximation and should be treated as approximate screening results, not final inferential proof.
- Mann-Whitney U test is used as a non-parametric distributional comparison. P-values use a normal approximation with tie correction.
- Cohen's d reports standardized mean difference.
- Cliff's delta reports ordinal effect size.
- Bootstrap difference CIs report the 95% CI of GPT-4o-mini mean minus random-tax mean.
- No multiple-comparison correction is applied in this exploratory Stage 2 table. Interpret significance across many correlated metrics cautiously.

## Interpretation Rule

- Test results describe empirical differences in this simulation design.
- They should not be treated as causal proof without the design assumptions, robustness checks, time-series analysis, and thought-trace interpretation.
"""
    (out_dir / "group_level_methods.md").write_text(methods, encoding="utf-8")


def create_stage2_figures(run_df: pd.DataFrame, out_dir: Path) -> None:
    figure_dir = out_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    plot_metrics = [
        ("final_gini", "Final Gini"),
        ("mean_gini", "Mean Gini"),
        ("gini_auc_per_step", "Gini AUC Per Step"),
        ("final_mean_coin", "Final Mean Coin"),
        ("mean_coin_over_time", "Mean Coin Over Time"),
        ("completed_build_count", "Completed Builds"),
        ("final_swf_absolute", "Final Social Welfare"),
        ("mean_swf_absolute", "Mean Social Welfare"),
        ("cumulative_planner_reward", "Cumulative SWF Gain"),
        ("coin_per_labor", "Coin Per Labor"),
        ("builds_per_labor", "Builds Per Labor"),
        ("fallback_count", "Fallback Count"),
        ("illegal_action_warning_count", "Illegal Action Warnings"),
        ("noop_action_share", "NOOP Action Share"),
        ("build_action_share", "Build Action Share"),
        ("order_action_share", "Order Action Share"),
        ("move_action_share", "Move Action Share"),
        ("mean_tax_action_value", "Mean Tax Action Value"),
        ("tax_action_volatility", "Tax Action Volatility"),
    ]
    colors = {"gpt4omini": "#0072B2", "random_tax": "#D55E00"}
    labels = {"gpt4omini": "GPT-4o-mini Planner", "random_tax": "Random-Tax Baseline"}

    for metric, title in plot_metrics:
        if metric not in run_df.columns:
            continue
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        data = [
            pd.to_numeric(run_df.loc[run_df["group"] == group, metric], errors="coerce").dropna()
            for group in COMPARISON_GROUPS
        ]
        box = ax.boxplot(
            data,
            tick_labels=[labels[group] for group in COMPARISON_GROUPS],
            patch_artist=True,
            showfliers=False,
        )
        for patch, group in zip(box["boxes"], COMPARISON_GROUPS):
            patch.set_facecolor(colors[group])
            patch.set_alpha(0.35)
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        for i, (group, values) in enumerate(zip(COMPARISON_GROUPS, data), start=1):
            jitter = rng.normal(loc=i, scale=0.035, size=len(values))
            ax.scatter(jitter, values, s=14, alpha=0.55, color=colors[group], edgecolors="none")
        ax.set_title(title)
        ax.set_ylabel(metric_display_label(metric))
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(figure_dir / f"{metric}_box_jitter.png", dpi=180)
        fig.savefig(figure_dir / f"{metric}_box_jitter.svg")
        plt.close(fig)

    create_mean_difference_figure(out_dir, figure_dir)
    create_metric_family_figures(run_df, figure_dir, colors, labels)


def create_mean_difference_figure(out_dir: Path, figure_dir: Path) -> None:
    tests_path = out_dir / "statistical_tests.csv"
    if not tests_path.exists():
        return
    tests = pd.read_csv(tests_path)
    preferred_order = [
        "mean_gini",
        "gini_auc_per_step",
        "final_gini",
        "mean_coin_over_time",
        "final_mean_coin",
        "completed_build_count",
        "mean_swf_absolute",
        "final_swf_absolute",
        "cumulative_planner_reward",
        "coin_per_labor",
        "builds_per_labor",
        "noop_action_share",
        "build_action_share",
        "order_action_share",
        "move_action_share",
        "fallback_count",
        "illegal_action_warning_count",
        "mean_tax_action_value",
        "tax_action_volatility",
    ]
    tests = tests[tests["metric"].isin(preferred_order)].copy()
    tests["metric"] = pd.Categorical(tests["metric"], categories=preferred_order, ordered=True)
    tests = tests.sort_values("metric")
    if tests.empty:
        return

    _plot_mean_difference(
        tests,
        figure_dir / "mean_difference_forest_raw_all",
        "Raw Mean Differences with Bootstrap 95% CI",
        "Mean difference: GPT-4o-mini planner minus random-tax baseline",
    )
    # Backward-compatible filename for existing references.
    _plot_mean_difference(
        tests,
        figure_dir / "mean_difference_forest",
        "Raw Mean Differences with Bootstrap 95% CI",
        "Mean difference: GPT-4o-mini planner minus random-tax baseline",
    )
    create_standardized_effect_figure(tests, figure_dir)
    create_mean_difference_family_figures(tests, figure_dir)


def _plot_mean_difference(
    tests: pd.DataFrame,
    output_stem: Path,
    title: str,
    xlabel: str,
) -> None:
    y = np.arange(len(tests))
    diff = tests["mean_diff_a_minus_b"].astype(float)
    lo = tests["bootstrap_diff_ci_low"].astype(float)
    hi = tests["bootstrap_diff_ci_high"].astype(float)
    xerr = np.vstack([diff - lo, hi - diff])

    fig, ax = plt.subplots(figsize=(9, max(6, len(tests) * 0.38)))
    significant = (lo > 0) | (hi < 0)
    ax.errorbar(
        diff,
        y,
        xerr=xerr,
        fmt="o",
        color="#333333",
        ecolor="#777777",
        capsize=3,
        markersize=4,
        linewidth=1,
    )
    ax.scatter(diff[significant], y[significant], color="#0072B2", s=28, zorder=3)
    ax.axvline(0, color="#B00020", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([metric_display_label(str(m)) for m in tests["metric"]])
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=200)
    fig.savefig(output_stem.with_suffix(".svg"))
    plt.close(fig)


def create_standardized_effect_figure(tests: pd.DataFrame, figure_dir: Path) -> None:
    values = tests["cohens_d"].astype(float)
    y = np.arange(len(tests))
    fig, ax = plt.subplots(figsize=(8.5, max(6, len(tests) * 0.38)))
    colors = np.where(values > 0, "#0072B2", "#D55E00")
    ax.barh(y, values, color=colors, alpha=0.8)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.axvline(0.2, color="#999999", linestyle=":", linewidth=0.8)
    ax.axvline(-0.2, color="#999999", linestyle=":", linewidth=0.8)
    ax.axvline(0.5, color="#999999", linestyle=":", linewidth=0.8)
    ax.axvline(-0.5, color="#999999", linestyle=":", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([metric_display_label(str(m)) for m in tests["metric"]])
    ax.invert_yaxis()
    ax.set_xlabel("Cohen's d: GPT-4o-mini planner minus random-tax baseline")
    ax.set_title("Standardized Effect Sizes Across Metrics")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "standardized_effect_size_forest.png", dpi=200)
    fig.savefig(figure_dir / "standardized_effect_size_forest.svg")
    plt.close(fig)


def create_mean_difference_family_figures(tests: pd.DataFrame, figure_dir: Path) -> None:
    families = {
        "inequality": ["mean_gini", "gini_auc_per_step", "final_gini"],
        "productivity": [
            "mean_coin_over_time",
            "final_mean_coin",
            "completed_build_count",
        ],
        "welfare": [
            "mean_swf_absolute",
            "final_swf_absolute",
            "cumulative_planner_reward",
        ],
        "behavior": [
            "noop_action_share",
            "order_action_share",
            "move_action_share",
            "build_action_share",
        ],
        "quality": ["fallback_count", "illegal_action_warning_count"],
        "tax_policy": ["mean_tax_action_value", "tax_action_volatility"],
    }
    for family_name, metric_order in families.items():
        family = tests[tests["metric"].isin(metric_order)].copy()
        if family.empty:
            continue
        family["metric"] = family["metric"].astype(str)
        family["metric"] = pd.Categorical(
            family["metric"], categories=metric_order, ordered=True
        )
        family = family.sort_values("metric")
        _plot_mean_difference(
            family,
            figure_dir / f"mean_difference_forest_{family_name}",
            f"{family_name.replace('_', ' ').title()} Mean Differences",
            "Mean difference: GPT-4o-mini planner minus random-tax baseline",
        )


def create_metric_family_figures(
    run_df: pd.DataFrame,
    figure_dir: Path,
    colors: dict[str, str],
    labels: dict[str, str],
) -> None:
    families = {
        "inequality_metrics": [
            ("final_gini", "Final Gini"),
            ("mean_gini", "Mean Gini"),
            ("gini_auc_per_step", "Gini AUC Per Step"),
        ],
        "productivity_metrics": [
            ("final_mean_coin", "Final Mean Coin"),
            ("mean_coin_over_time", "Mean Coin Over Time"),
            ("completed_build_count", "Completed Builds"),
        ],
        "welfare_metrics": [
            ("final_swf_absolute", "Final Social Welfare"),
            ("mean_swf_absolute", "Mean Social Welfare"),
            ("cumulative_planner_reward", "Cumulative SWF Gain"),
        ],
        "behavior_metrics": [
            ("noop_action_share", "NOOP Share"),
            ("order_action_share", "Order Share"),
            ("move_action_share", "Move Share"),
            ("build_action_share", "Build Share"),
        ],
        "quality_metrics": [
            ("fallback_count", "Fallback Count"),
            ("illegal_action_warning_count", "Illegal Action Warnings"),
        ],
    }

    for family_name, metrics in families.items():
        available = [(m, title) for m, title in metrics if m in run_df.columns]
        if not available:
            continue
        fig, axes = plt.subplots(
            1,
            len(available),
            figsize=(4.2 * len(available), 4.2),
            squeeze=False,
        )
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        for ax, (metric, title) in zip(axes.ravel(), available):
            data = [
                pd.to_numeric(
                    run_df.loc[run_df["group"] == group, metric], errors="coerce"
                ).dropna()
                for group in COMPARISON_GROUPS
            ]
            box = ax.boxplot(
                data,
                tick_labels=[labels[group] for group in COMPARISON_GROUPS],
                patch_artist=True,
                showfliers=False,
            )
            for patch, group in zip(box["boxes"], COMPARISON_GROUPS):
                patch.set_facecolor(colors[group])
                patch.set_alpha(0.35)
            for i, (group, values) in enumerate(zip(COMPARISON_GROUPS, data), start=1):
                jitter = rng.normal(loc=i, scale=0.035, size=len(values))
                ax.scatter(
                    jitter,
                    values,
                    s=10,
                    alpha=0.45,
                    color=colors[group],
                    edgecolors="none",
                )
            ax.set_title(title)
            ax.tick_params(axis="x", rotation=20)
            ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(figure_dir / f"{family_name}.png", dpi=200)
        fig.savefig(figure_dir / f"{family_name}.svg")
        plt.close(fig)


def bootstrap_ci_for_matrix(
    values: np.ndarray,
    rng: np.random.Generator,
    iterations: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap CI for the mean trajectory across runs."""
    if values.size == 0:
        return np.array([]), np.array([])
    n_runs = values.shape[0]
    idx = rng.integers(0, n_runs, size=(iterations, n_runs))
    sampled_means = values[idx].mean(axis=1)
    lo = np.quantile(sampled_means, 0.025, axis=0)
    hi = np.quantile(sampled_means, 0.975, axis=0)
    return lo, hi


def load_time_series_for_group(
    run_df: pd.DataFrame,
    batch_dir: Path,
    group: str,
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    steps: np.ndarray | None = None
    for _, run in run_df[run_df["group"] == group].sort_values("run_num").iterrows():
        step_path = batch_dir / str(run["run_name"]) / "step_metrics.csv"
        if not step_path.exists():
            continue
        df = pd.read_csv(step_path)
        if metric not in df.columns or "step" not in df.columns:
            continue
        metric_values = pd.to_numeric(df[metric], errors="coerce")
        if metric_values.isna().any():
            continue
        current_steps = pd.to_numeric(df["step"], errors="coerce").to_numpy(dtype=int)
        if steps is None:
            steps = current_steps
        if len(current_steps) != len(steps) or np.any(current_steps != steps):
            continue
        rows.append(metric_values.to_numpy(dtype=float))
    if steps is None or not rows:
        return np.array([]), np.array([])
    return steps, np.vstack(rows)


def load_cumulative_build_series_for_group(
    run_df: pd.DataFrame,
    batch_dir: Path,
    group: str,
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    steps: np.ndarray | None = None
    for _, run in run_df[run_df["group"] == group].sort_values("run_num").iterrows():
        step_path = batch_dir / str(run["run_name"]) / "step_metrics.csv"
        if not step_path.exists():
            continue
        df = pd.read_csv(step_path)
        if "build_total" not in df.columns or "step" not in df.columns:
            continue
        current_steps = pd.to_numeric(df["step"], errors="coerce").to_numpy(dtype=int)
        if steps is None:
            steps = current_steps
        if len(current_steps) != len(steps) or np.any(current_steps != steps):
            continue
        rows.append(pd.to_numeric(df["build_total"], errors="coerce").fillna(0).cumsum().to_numpy(dtype=float))
    if steps is None or not rows:
        return np.array([]), np.array([])
    return steps, np.vstack(rows)


def load_action_share_series_for_group(
    run_df: pd.DataFrame,
    batch_dir: Path,
    group: str,
    category: str,
    window: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    category_ranges = {
        "noop": lambda s: s == 0,
        "build": lambda s: s == 1,
        "order": lambda s: (s >= 2) & (s <= 45),
        "move": lambda s: (s >= 46) & (s <= 49),
    }
    rows: list[np.ndarray] = []
    steps: np.ndarray | None = None
    predicate = category_ranges[category]
    for _, run in run_df[run_df["group"] == group].sort_values("run_num").iterrows():
        action_path = batch_dir / str(run["run_name"]) / "action_log.csv"
        if not action_path.exists():
            continue
        df = pd.read_csv(action_path)
        action_cols = [c for c in df.columns if c.startswith("action_agent_")]
        if not action_cols or "step" not in df.columns:
            continue
        current_steps = pd.to_numeric(df["step"], errors="coerce").to_numpy(dtype=int)
        if steps is None:
            steps = current_steps
        if len(current_steps) != len(steps) or np.any(current_steps != steps):
            continue
        actions = df[action_cols].apply(pd.to_numeric, errors="coerce")
        share = predicate(actions).mean(axis=1).rolling(window=window, min_periods=1).mean()
        rows.append(share.to_numpy(dtype=float))
    if steps is None or not rows:
        return np.array([]), np.array([])
    return steps, np.vstack(rows)


def plot_group_trajectory(
    ax: plt.Axes,
    steps: np.ndarray,
    values_by_group: dict[str, np.ndarray],
    title: str,
    ylabel: str,
    colors: dict[str, str],
    labels: dict[str, str],
) -> None:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for group in COMPARISON_GROUPS:
        values = values_by_group.get(group)
        if values is None or values.size == 0:
            continue
        mean = values.mean(axis=0)
        lo, hi = bootstrap_ci_for_matrix(values, rng)
        ax.plot(steps, mean, color=colors[group], linewidth=1.8, label=labels[group])
        if len(lo):
            ax.fill_between(steps, lo, hi, color=colors[group], alpha=0.18, linewidth=0)
        ax.scatter([steps[-1]], [mean[-1]], color=colors[group], s=28, zorder=3)
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)


def create_time_series_figures(
    run_df: pd.DataFrame,
    batch_dir: Path,
    out_dir: Path,
) -> None:
    figure_dir = out_dir / "figures" / "time_series"
    figure_dir.mkdir(parents=True, exist_ok=True)
    colors = {"gpt4omini": "#0072B2", "random_tax": "#D55E00"}
    labels = {"gpt4omini": "GPT-4o-mini Planner", "random_tax": "Random-Tax Baseline"}

    metric_families = {
        "time_series_inequality": [
            ("gini", "Gini Over Time", "Gini"),
        ],
        "time_series_productivity": [
            ("mean_coin", "Mean Coin Over Time", "Mean Coin"),
            ("cumulative_builds", "Cumulative Builds Over Time", "Completed Builds"),
        ],
        "time_series_welfare": [
            ("swf_absolute", "Social Welfare Over Time", "Social Welfare"),
            ("cumulative_planner_reward", "Cumulative SWF Gain Over Time", "Cumulative SWF Gain"),
        ],
    }

    for family_name, metrics in metric_families.items():
        fig, axes = plt.subplots(1, len(metrics), figsize=(6.2 * len(metrics), 4.4), squeeze=False)
        for ax, (metric, title, ylabel) in zip(axes.ravel(), metrics):
            values_by_group: dict[str, np.ndarray] = {}
            steps = np.array([])
            for group in COMPARISON_GROUPS:
                if metric == "cumulative_builds":
                    group_steps, group_values = load_cumulative_build_series_for_group(run_df, batch_dir, group)
                else:
                    group_steps, group_values = load_time_series_for_group(run_df, batch_dir, group, metric)
                if len(group_steps):
                    steps = group_steps
                values_by_group[group] = group_values
            if len(steps):
                plot_group_trajectory(ax, steps, values_by_group, title, ylabel, colors, labels)
            else:
                ax.set_title(f"{title} (missing)")
        fig.tight_layout()
        fig.savefig(figure_dir / f"{family_name}.png", dpi=200)
        fig.savefig(figure_dir / f"{family_name}.svg")
        plt.close(fig)

    behavior_categories = [
        ("noop", "NOOP Share"),
        ("build", "Build Share"),
        ("order", "Order Share"),
        ("move", "Move Share"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), squeeze=False)
    for ax, (category, title) in zip(axes.ravel(), behavior_categories):
        values_by_group = {}
        steps = np.array([])
        for group in COMPARISON_GROUPS:
            group_steps, group_values = load_action_share_series_for_group(
                run_df, batch_dir, group, category
            )
            if len(group_steps):
                steps = group_steps
            values_by_group[group] = group_values
        if len(steps):
            plot_group_trajectory(
                ax,
                steps,
                values_by_group,
                f"{title} Over Time (50-step rolling mean)",
                title,
                colors,
                labels,
            )
        else:
            ax.set_title(f"{title} (missing)")
    fig.tight_layout()
    fig.savefig(figure_dir / "time_series_behavior.png", dpi=200)
    fig.savefig(figure_dir / "time_series_behavior.svg")
    plt.close(fig)


def add_quality_event_count(run_df: pd.DataFrame) -> pd.DataFrame:
    df = run_df.copy()
    quality_cols = ["fallback_count", "illegal_action_warning_count"]
    for col in quality_cols:
        if col not in df.columns:
            df[col] = 0
    df["quality_event_count"] = (
        pd.to_numeric(df["fallback_count"], errors="coerce").fillna(0)
        + pd.to_numeric(df["illegal_action_warning_count"], errors="coerce").fillna(0)
    )
    return df


def filter_top_quality_tail(
    run_df: pd.DataFrame,
    quality_col: str,
    quantile: float = 0.95,
) -> pd.DataFrame:
    keep = pd.Series(False, index=run_df.index)
    for group in COMPARISON_GROUPS:
        group_mask = run_df["group"] == group
        values = pd.to_numeric(run_df.loc[group_mask, quality_col], errors="coerce")
        threshold = values.quantile(quantile)
        keep.loc[group_mask] = values <= threshold
    return run_df.loc[keep].copy()


def winsorize_metric_by_group(
    run_df: pd.DataFrame,
    metric: str,
    lower: float = 0.05,
    upper: float = 0.95,
) -> pd.DataFrame:
    df = run_df.copy()
    df[metric] = pd.to_numeric(df[metric], errors="coerce").astype(float)
    for group in COMPARISON_GROUPS:
        mask = df["group"] == group
        values = pd.to_numeric(df.loc[mask, metric], errors="coerce")
        lo = values.quantile(lower)
        hi = values.quantile(upper)
        df.loc[mask, metric] = values.clip(lo, hi)
    return df


def summarize_robust_metric(
    run_df: pd.DataFrame,
    metric: str,
    scenario: str,
    estimator: str,
    bootstrap_iterations: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    a = pd.to_numeric(
        run_df.loc[run_df["group"] == COMPARISON_GROUPS[0], metric],
        errors="coerce",
    ).dropna()
    b = pd.to_numeric(
        run_df.loc[run_df["group"] == COMPARISON_GROUPS[1], metric],
        errors="coerce",
    ).dropna()
    if estimator == "median":
        estimate_a = float(a.median()) if len(a) else None
        estimate_b = float(b.median()) if len(b) else None
        diff = (
            float(a.median() - b.median())
            if len(a) and len(b)
            else None
        )
        ci_lo, ci_hi = bootstrap_median_diff_ci(a, b, bootstrap_iterations, rng)
    else:
        estimate_a = float(a.mean()) if len(a) else None
        estimate_b = float(b.mean()) if len(b) else None
        diff = (
            float(a.mean() - b.mean())
            if len(a) and len(b)
            else None
        )
        ci_lo, ci_hi = bootstrap_diff_ci(a, b, bootstrap_iterations, rng)
    return {
        "scenario": scenario,
        "metric": metric,
        "metric_label": metric_display_label(metric),
        "estimator": estimator,
        "n_gpt4omini": int(a.count()),
        "n_random_tax": int(b.count()),
        "estimate_gpt4omini": estimate_a,
        "estimate_random_tax": estimate_b,
        "diff_gpt4omini_minus_random_tax": diff,
        "bootstrap_diff_ci_low": ci_lo,
        "bootstrap_diff_ci_high": ci_hi,
        "ci_excludes_zero": bool(ci_lo is not None and ci_hi is not None and (ci_lo > 0 or ci_hi < 0)),
        "cohens_d": cohens_d(a, b),
        "cliffs_delta": cliffs_delta(a, b),
    }


def create_robustness_outputs(
    run_df: pd.DataFrame,
    out_dir: Path,
    bootstrap_iterations: int,
) -> pd.DataFrame:
    analysis_df = add_quality_event_count(run_df[run_df["analysis_ready"]].copy())
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    scenario_builders = [
        (
            "main_analysis_all_runs",
            "mean",
            lambda metric: analysis_df,
        ),
        (
            "exclude_top_5pct_fallback_within_group",
            "mean",
            lambda metric: filter_top_quality_tail(analysis_df, "fallback_count"),
        ),
        (
            "exclude_top_5pct_illegal_actions_within_group",
            "mean",
            lambda metric: filter_top_quality_tail(
                analysis_df, "illegal_action_warning_count"
            ),
        ),
        (
            "exclude_top_5pct_quality_events_within_group",
            "mean",
            lambda metric: filter_top_quality_tail(analysis_df, "quality_event_count"),
        ),
        (
            "winsorize_metric_5_95_within_group",
            "mean",
            lambda metric: winsorize_metric_by_group(analysis_df, metric),
        ),
        (
            "median_difference_all_runs",
            "median",
            lambda metric: analysis_df,
        ),
    ]
    rows: list[dict[str, Any]] = []
    available_metrics = [m for m in ROBUSTNESS_METRICS if m in analysis_df.columns]
    for metric in available_metrics:
        for scenario, estimator, builder in scenario_builders:
            scenario_df = builder(metric)
            rows.append(
                summarize_robust_metric(
                    scenario_df,
                    metric,
                    scenario,
                    estimator,
                    bootstrap_iterations,
                    rng,
                )
            )
    robustness = pd.DataFrame(rows)
    if not robustness.empty:
        baseline = robustness[
            robustness["scenario"] == "main_analysis_all_runs"
        ][["metric", "diff_gpt4omini_minus_random_tax"]].rename(
            columns={"diff_gpt4omini_minus_random_tax": "baseline_diff"}
        )
        robustness = robustness.merge(baseline, on="metric", how="left")
        robustness["same_direction_as_baseline"] = np.sign(
            robustness["diff_gpt4omini_minus_random_tax"]
        ) == np.sign(robustness["baseline_diff"])
        robustness["absolute_change_from_baseline"] = (
            robustness["diff_gpt4omini_minus_random_tax"] - robustness["baseline_diff"]
        ).abs()
    robustness.to_csv(
        out_dir / "robustness_checks.csv", index=False, encoding="utf-8-sig"
    )
    write_robustness_methods(out_dir, bootstrap_iterations)
    create_robustness_figures(analysis_df, robustness, out_dir)
    return robustness


def write_robustness_methods(out_dir: Path, bootstrap_iterations: int) -> None:
    methods = f"""# Robustness Check Methods

Generated: {datetime.now(timezone.utc).isoformat()}

## Purpose

Robustness checks test whether the main GPT-4o-mini planner versus random-tax baseline differences are driven by a small number of problematic or extreme runs. They are sensitivity checks, not replacements for the main analysis.

## Included Runs

- The starting pool is `analysis_ready == True`.
- No source result files are modified.
- The main analysis remains the full 100 versus 100 run comparison.

## Scenarios

- `main_analysis_all_runs`: original analysis-ready sample.
- `exclude_top_5pct_fallback_within_group`: excludes runs above each group's 95th percentile fallback count.
- `exclude_top_5pct_illegal_actions_within_group`: excludes runs above each group's 95th percentile illegal-action warning count.
- `exclude_top_5pct_quality_events_within_group`: excludes runs above each group's 95th percentile of fallback plus illegal-action warning count.
- `winsorize_metric_5_95_within_group`: clips each metric within each group to its 5th and 95th percentile before comparing means.
- `median_difference_all_runs`: compares group medians rather than group means.

## Reported Statistics

- `diff_gpt4omini_minus_random_tax` is GPT-4o-mini planner minus random-tax baseline.
- Bootstrap 95% CIs use {bootstrap_iterations} resamples with replacement and fixed random seed `{BOOTSTRAP_SEED}`.
- `same_direction_as_baseline` checks whether each robustness scenario has the same sign as the main analysis estimate.
- These checks assess stability of observed differences. They do not establish causal mechanisms by themselves.
"""
    (out_dir / "robustness_methods.md").write_text(methods, encoding="utf-8")


def create_robustness_figures(
    run_df: pd.DataFrame,
    robustness: pd.DataFrame,
    out_dir: Path,
) -> None:
    figure_dir = out_dir / "figures" / "robustness"
    figure_dir.mkdir(parents=True, exist_ok=True)
    if robustness.empty:
        return

    stability = robustness[
        robustness["scenario"].isin(
            [
                "main_analysis_all_runs",
                "exclude_top_5pct_fallback_within_group",
                "exclude_top_5pct_illegal_actions_within_group",
                "exclude_top_5pct_quality_events_within_group",
                "winsorize_metric_5_95_within_group",
                "median_difference_all_runs",
            ]
        )
    ].copy()
    scenario_labels = {
        "main_analysis_all_runs": "All runs",
        "exclude_top_5pct_fallback_within_group": "Drop high fallback",
        "exclude_top_5pct_illegal_actions_within_group": "Drop high illegal",
        "exclude_top_5pct_quality_events_within_group": "Drop high quality events",
        "winsorize_metric_5_95_within_group": "Winsorized 5-95%",
        "median_difference_all_runs": "Median diff",
    }
    stability["scenario_label"] = stability["scenario"].map(scenario_labels)

    key_metrics = [
        "mean_gini",
        "mean_coin_over_time",
        "mean_swf_absolute",
        "cumulative_planner_reward",
        "noop_action_share",
        "order_action_share",
    ]
    plot_df = stability[stability["metric"].isin(key_metrics)].copy()
    plot_df["metric"] = pd.Categorical(plot_df["metric"], categories=key_metrics, ordered=True)
    plot_df = plot_df.sort_values(["metric", "scenario"])
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), squeeze=False)
    for ax, metric in zip(axes.ravel(), key_metrics):
        metric_df = plot_df[plot_df["metric"] == metric]
        y = np.arange(len(metric_df))
        ax.errorbar(
            metric_df["diff_gpt4omini_minus_random_tax"],
            y,
            xerr=np.vstack(
                [
                    metric_df["diff_gpt4omini_minus_random_tax"]
                    - metric_df["bootstrap_diff_ci_low"],
                    metric_df["bootstrap_diff_ci_high"]
                    - metric_df["diff_gpt4omini_minus_random_tax"],
                ]
            ),
            fmt="o",
            color="#333333",
            ecolor="#777777",
            capsize=3,
            markersize=4,
        )
        ax.axvline(0, color="#B00020", linestyle="--", linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(metric_df["scenario_label"])
        ax.set_title(metric_display_label(metric))
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Robustness of Mean Differences Across Sensitivity Checks", y=1.02)
    fig.tight_layout()
    fig.savefig(figure_dir / "robustness_effect_stability.png", dpi=200)
    fig.savefig(figure_dir / "robustness_effect_stability.svg")
    plt.close(fig)

    heat = stability.pivot(
        index="metric_label",
        columns="scenario_label",
        values="same_direction_as_baseline",
    )
    heat = heat.reindex([metric_display_label(m) for m in ROBUSTNESS_METRICS if metric_display_label(m) in heat.index])
    fig, ax = plt.subplots(figsize=(10, max(5, len(heat) * 0.32)))
    matrix = heat.astype(float).to_numpy()
    ax.imshow(matrix, aspect="auto", cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_xticklabels(heat.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels(heat.index)
    ax.set_title("Robustness Direction Check (Green = Same Sign as Main Estimate)")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, "same" if matrix[i, j] == 1 else "flip", ha="center", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(figure_dir / "robustness_direction_heatmap.png", dpi=200)
    fig.savefig(figure_dir / "robustness_direction_heatmap.svg")
    plt.close(fig)

    colors = {"gpt4omini": "#0072B2", "random_tax": "#D55E00"}
    labels = {"gpt4omini": "GPT-4o-mini Planner", "random_tax": "Random-Tax Baseline"}
    scatter_metrics = [
        ("mean_swf_absolute", "Mean Social Welfare"),
        ("mean_gini", "Mean Gini"),
        ("mean_coin_over_time", "Mean Coin Over Time"),
    ]
    fig, axes = plt.subplots(1, len(scatter_metrics), figsize=(5.3 * len(scatter_metrics), 4.2), squeeze=False)
    for ax, (metric, title) in zip(axes.ravel(), scatter_metrics):
        for group in COMPARISON_GROUPS:
            group_df = run_df[run_df["group"] == group]
            ax.scatter(
                group_df["quality_event_count"],
                group_df[metric],
                s=22,
                alpha=0.65,
                color=colors[group],
                label=labels[group],
                edgecolors="none",
            )
        ax.set_title(title)
        ax.set_xlabel("Fallback + illegal-action warnings")
        ax.set_ylabel(metric_display_label(metric))
        ax.grid(alpha=0.25)
    axes.ravel()[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "quality_events_vs_core_outcomes.png", dpi=200)
    fig.savefig(figure_dir / "quality_events_vs_core_outcomes.svg")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    batch = args.batch
    out_dir = args.out or (batch / "_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    run_names = expected_runs_from_batch(batch)
    if not run_names:
        raise SystemExit(f"No run directories or batch logs found under {batch}")

    rows = [build_run_row(run_name, batch) for run_name in run_names]
    df = pd.DataFrame(rows).sort_values(["group", "run_num", "run_name"])
    out_csv = out_dir / "run_level_summary.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    group_summary, tests = create_group_level_outputs(
        df, out_dir, args.bootstrap_iterations, batch
    )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "run_and_group_level_analysis",
        "batch": str(batch),
        "output_dir": str(out_dir),
        "run_count": int(len(df)),
        "complete_run_count": int(df["is_complete"].sum()),
        "analysis_ready_run_count": int(df["analysis_ready"].sum()),
        "bootstrap_iterations": args.bootstrap_iterations,
        "groups": df["group"].value_counts().sort_index().to_dict(),
        "outputs": [
            "run_level_summary.csv",
            "run_level_methods.md",
            "group_level_summary.csv",
            "group_level_methods.md",
            "statistical_tests.csv",
            "robustness_checks.csv",
            "robustness_methods.md",
            "figures/*.png",
            "figures/*.svg",
            "figures/robustness/*.png",
            "figures/robustness/*.svg",
            "analysis_manifest.json",
        ],
    }
    (out_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_methods(out_dir, batch, len(df))

    print(f"Wrote {out_csv}")
    print(f"Runs: {len(df)}")
    print(f"Complete runs: {int(df['is_complete'].sum())}")
    print(f"Analysis-ready runs: {int(df['analysis_ready'].sum())}")
    print(df["group"].value_counts().sort_index().to_string())
    print("\nSelected quality totals:")
    print(
        df[
            [
                "group",
                "fallback_count",
                "illegal_action_warning_count",
                "timeout_count",
                "http_non_200_count",
                "traceback_count",
            ]
        ]
        .groupby("group")
        .sum()
        .to_string()
    )
    print("\nStage 2 outputs:")
    print(f"Group summary rows: {len(group_summary)}")
    print(f"Statistical test rows: {len(tests)}")


if __name__ == "__main__":
    main()
