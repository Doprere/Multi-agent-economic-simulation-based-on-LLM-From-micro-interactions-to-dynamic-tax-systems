#!/usr/bin/env python3
"""Parallel Linux batch runner for AI Economist LLM simulations.

This wrapper intentionally calls run_simulation.py through subprocesses instead
of importing run_episode(). That keeps the research code untouched and avoids
cross-run global config cache contamination.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import shutil
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_SIMULATION = PROJECT_ROOT / "run_simulation.py"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sanitize_tag(value: str) -> str:
    safe = []
    for ch in value:
        if ch.isalnum():
            safe.append(ch)
        elif ch in ("-", "_"):
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe).strip("_")


def csv_row_count(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = sum(1 for _ in reader)
    return max(0, rows - 1)


def expected_tax_count(steps: int, tax_period: int) -> int:
    if steps <= 0:
        return 0
    return ((steps - 1) // tax_period) + 1


@dataclass
class ValidationResult:
    ok: bool
    details: dict[str, Any]
    errors: list[str]


def validate_completed_run(
    run_dir: Path,
    expected_steps: int,
    tax_period: int,
    require_completed_marker: bool,
) -> ValidationResult:
    errors: list[str] = []
    summary_path = run_dir / "summary.json"
    completed_path = run_dir / "COMPLETED.json"

    summary: dict[str, Any] = {}
    if not summary_path.exists():
        errors.append("summary.json missing")
    else:
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"summary.json unreadable: {exc}")

    completed: dict[str, Any] = {}
    if require_completed_marker:
        if not completed_path.exists():
            errors.append("COMPLETED.json missing")
        else:
            try:
                completed = json.loads(completed_path.read_text(encoding="utf-8"))
                if completed.get("status") != "completed":
                    errors.append("COMPLETED.json status is not completed")
            except Exception as exc:
                errors.append(f"COMPLETED.json unreadable: {exc}")

    total_steps = summary.get("total_steps")
    final_step = (summary.get("final_metrics") or {}).get("step")
    tax_count = summary.get("tax_count")
    token_usage = summary.get("token_usage") or {}
    has_agents_usage = bool(token_usage.get("agents"))
    has_planner_usage = bool(token_usage.get("planner"))

    if total_steps != expected_steps:
        errors.append(f"total_steps {total_steps!r} != expected {expected_steps}")
    if final_step != expected_steps - 1:
        errors.append(f"final_metrics.step {final_step!r} != expected {expected_steps - 1}")

    expected_taxes = expected_tax_count(expected_steps, tax_period)
    if tax_count != expected_taxes:
        errors.append(f"tax_count {tax_count!r} != expected {expected_taxes}")

    step_rows = csv_row_count(run_dir / "step_metrics.csv")
    action_rows = csv_row_count(run_dir / "action_log.csv")
    tax_rows = csv_row_count(run_dir / "tax_log.csv")

    if step_rows != expected_steps:
        errors.append(f"step_metrics rows {step_rows!r} != expected {expected_steps}")
    if action_rows != expected_steps:
        errors.append(f"action_log rows {action_rows!r} != expected {expected_steps}")
    if tax_rows != expected_taxes:
        errors.append(f"tax_log rows {tax_rows!r} != expected {expected_taxes}")
    if not has_agents_usage:
        errors.append("summary.token_usage.agents missing")
    if not has_planner_usage:
        errors.append("summary.token_usage.planner missing")

    details = {
        "summary_path": str(summary_path),
        "total_steps": total_steps,
        "final_step": final_step,
        "tax_count": tax_count,
        "expected_tax_count": expected_taxes,
        "has_token_usage": has_agents_usage and has_planner_usage,
        "step_metrics_rows": step_rows,
        "action_log_rows": action_rows,
        "tax_log_rows": tax_rows,
        "completed_marker": completed,
    }
    return ValidationResult(ok=not errors, details=details, errors=errors)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_log(path: Path, message: str) -> None:
    line = f"{utc_now()} {message}\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
    print(line, end="", flush=True)


def archive_existing_run(run_dir: Path, reason: str, batch_log: Path) -> None:
    if not run_dir.exists():
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archived = run_dir.with_name(f"{run_dir.name}.{reason}_{stamp}")
    suffix = 1
    while archived.exists():
        archived = run_dir.with_name(f"{run_dir.name}.{reason}_{stamp}_{suffix}")
        suffix += 1
    shutil.move(str(run_dir), str(archived))
    append_log(batch_log, f"[ARCHIVE] {run_dir.name} -> {archived.name}")


def is_archived_run_dir(run_dir: Path) -> bool:
    name = run_dir.name
    archive_tags = (".force_rerun_", ".incomplete_", ".interrupted_", ".failed_")
    return any(tag in name for tag in archive_tags)


def refresh_completed_list(output_dir: Path, expected_steps: int, tax_period: int) -> list[Path]:
    completed: list[Path] = []
    for marker in sorted(output_dir.glob("*/COMPLETED.json")):
        run_dir = marker.parent
        if is_archived_run_dir(run_dir):
            continue
        validation = validate_completed_run(
            run_dir=run_dir,
            expected_steps=expected_steps,
            tax_period=tax_period,
            require_completed_marker=True,
        )
        if validation.ok:
            completed.append(run_dir)

    list_path = output_dir / "completed_experiments.txt"
    with list_path.open("w", encoding="utf-8") as f:
        for run_dir in completed:
            f.write(str(run_dir.resolve()) + "\n")
    return completed


def build_command(args: argparse.Namespace, run_name: str, output_dir: Path) -> list[str]:
    cmd = [
        args.python,
        str(RUN_SIMULATION),
        "--steps",
        str(args.steps),
        "--agent-backend",
        args.agent_backend,
        "--agent-model",
        args.agent_model,
        "--planner-backend",
        args.planner_backend,
        "--planner-model",
        args.planner_model,
        "--ollama-url",
        args.ollama_url,
        "--run-name",
        run_name,
        "--output-dir",
        str(output_dir),
    ]
    if args.config:
        cmd += ["--config", args.config]
    if args.debug:
        cmd += ["--debug"]
    return cmd


async def stream_reader(reader: asyncio.StreamReader, log_file) -> None:
    while True:
        line = await reader.readline()
        if not line:
            break
        text = line.decode("utf-8", errors="replace")
        log_file.write(text)
        log_file.flush()


async def run_episode(
    args: argparse.Namespace,
    episode_index: int,
    output_dir: Path,
    logs_dir: Path,
    batch_log: Path,
    semaphore: asyncio.Semaphore,
) -> tuple[int, bool]:
    async with semaphore:
        agent_tag = sanitize_tag(args.agent_model)
        planner_tag = sanitize_tag(args.planner_model)
        run_name = f"{args.prefix}_{episode_index:04d}_agent_{agent_tag}_planner_{planner_tag}"
        run_dir = output_dir / run_name
        log_path = logs_dir / f"{run_name}.log"

        existing = validate_completed_run(
            run_dir=run_dir,
            expected_steps=args.steps,
            tax_period=args.tax_period,
            require_completed_marker=True,
        )
        if existing.ok and not args.force_rerun:
            append_log(batch_log, f"[SKIP] {run_name} already completed")
            return episode_index, True

        if run_dir.exists():
            if args.force_rerun:
                archive_existing_run(run_dir, "force_rerun", batch_log)
            elif existing.errors:
                write_json(run_dir / "INTERRUPTED.json", {
                    "status": "interrupted",
                    "run_name": run_name,
                    "episode_index": episode_index,
                    "interrupted_at": utc_now(),
                    "log_path": str(log_path),
                    "validation_errors": existing.errors,
                })
                archive_existing_run(run_dir, "incomplete", batch_log)

        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_command(args, run_name, output_dir)
        started_at = utc_now()
        t0 = time.time()
        write_json(run_dir / "RUNNING.json", {
            "status": "running",
            "run_name": run_name,
            "episode_index": episode_index,
            "started_at": started_at,
            "steps": args.steps,
            "agent_model": args.agent_model,
            "planner_model": args.planner_model,
            "command": cmd,
            "pid": None,
            "log_path": str(log_path),
        })

        append_log(batch_log, f"[START] {run_name}")
        env = os.environ.copy()
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(PROJECT_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        running_path = run_dir / "RUNNING.json"
        running = json.loads(running_path.read_text(encoding="utf-8"))
        running["pid"] = proc.pid
        write_json(running_path, running)

        interrupted = False
        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write(f"$ {' '.join(cmd)}\n\n")
            stdout_task = asyncio.create_task(stream_reader(proc.stdout, log_file))  # type: ignore[arg-type]
            stderr_task = asyncio.create_task(stream_reader(proc.stderr, log_file))  # type: ignore[arg-type]
            try:
                return_code = await proc.wait()
            except asyncio.CancelledError:
                interrupted = True
                if proc.returncode is None:
                    proc.send_signal(signal.SIGINT)
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=30)
                    except asyncio.TimeoutError:
                        proc.kill()
                        await proc.wait()
                return_code = proc.returncode if proc.returncode is not None else -130
            finally:
                await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)

        ended_at = utc_now()
        duration = round(time.time() - t0, 3)
        if running_path.exists():
            running_path.unlink()

        if interrupted:
            write_json(run_dir / "INTERRUPTED.json", {
                "status": "interrupted",
                "run_name": run_name,
                "episode_index": episode_index,
                "interrupted_at": ended_at,
                "log_path": str(log_path),
            })
            append_log(batch_log, f"[INTERRUPTED] {run_name}")
            return episode_index, False

        if return_code != 0:
            error_summary = ""
            if log_path.exists():
                lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                error_summary = "\n".join(lines[-20:])
            write_json(run_dir / "FAILED.json", {
                "status": "failed",
                "run_name": run_name,
                "episode_index": episode_index,
                "started_at": started_at,
                "ended_at": ended_at,
                "duration_seconds": duration,
                "return_code": return_code,
                "log_path": str(log_path),
                "error_summary": error_summary,
            })
            append_log(batch_log, f"[FAILED] {run_name} return_code={return_code}")
            return episode_index, False

        validation = validate_completed_run(
            run_dir=run_dir,
            expected_steps=args.steps,
            tax_period=args.tax_period,
            require_completed_marker=False,
        )
        if not validation.ok:
            write_json(run_dir / "FAILED.json", {
                "status": "failed",
                "run_name": run_name,
                "episode_index": episode_index,
                "started_at": started_at,
                "ended_at": ended_at,
                "duration_seconds": duration,
                "return_code": return_code,
                "log_path": str(log_path),
                "error_summary": "Completion validation failed",
                "validation_errors": validation.errors,
                "validation_details": validation.details,
            })
            append_log(batch_log, f"[FAILED] {run_name} validation failed: {validation.errors}")
            return episode_index, False

        completed = {
            "status": "completed",
            "run_name": run_name,
            "episode_index": episode_index,
            "started_at": started_at,
            "ended_at": ended_at,
            "duration_seconds": duration,
            **validation.details,
        }
        write_json(run_dir / "COMPLETED.json", completed)
        append_log(batch_log, f"[DONE] {run_name} duration={duration}s")
        return episode_index, True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parallel, resumable Linux runner for AI Economist LLM simulations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--episodes", type=int, required=True, help="Total number of episodes to ensure completed.")
    parser.add_argument("--parallel", type=int, default=1, help="Maximum concurrent simulations.")
    parser.add_argument("--steps", type=int, default=1000, help="Steps per episode.")
    parser.add_argument("--tax-period", type=int, default=100, help="Expected planner tax period for validation.")
    parser.add_argument("--prefix", default="episode", help="Run-name prefix.")
    parser.add_argument("--start-index", type=int, default=1, help="First episode index.")
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("SIM_OUTPUT_DIR", "linux_simulation_results"),
        help="Output root for simulation runs.",
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable used for child processes.")
    parser.add_argument("--agent-backend", choices=["openai", "ollama"], default="ollama")
    parser.add_argument("--agent-model", default="gemma4:e2b")
    parser.add_argument("--planner-backend", choices=["openai", "ollama"], default="openai")
    parser.add_argument("--planner-model", default="gpt-5.4-mini")
    parser.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    parser.add_argument("--config", default="", help="Optional config path passed through to run_simulation.py.")
    parser.add_argument("--force-rerun", action="store_true", help="Archive existing run directories and rerun them.")
    parser.add_argument("--debug", action="store_true", help="Pass --debug to run_simulation.py.")
    return parser.parse_args()


async def amain() -> int:
    args = parse_args()
    if args.episodes <= 0:
        print("--episodes must be positive", file=sys.stderr)
        return 2
    if args.parallel <= 0:
        print("--parallel must be positive", file=sys.stderr)
        return 2
    if not RUN_SIMULATION.exists():
        print(f"run_simulation.py not found: {RUN_SIMULATION}", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    batch_log = logs_dir / "batch.log"

    append_log(batch_log, (
        f"[BATCH START] episodes={args.episodes} parallel={args.parallel} "
        f"steps={args.steps} output_dir={output_dir}"
    ))

    semaphore = asyncio.Semaphore(args.parallel)
    indices = range(args.start_index, args.start_index + args.episodes)
    tasks = [
        asyncio.create_task(run_episode(args, i, output_dir, logs_dir, batch_log, semaphore))
        for i in indices
    ]

    interrupted = False
    results: list[tuple[int, bool]] = []
    try:
        for task in asyncio.as_completed(tasks):
            results.append(await task)
            completed = refresh_completed_list(output_dir, args.steps, args.tax_period)
            append_log(batch_log, f"[PROGRESS] completed_valid={len(completed)}")
    except (KeyboardInterrupt, asyncio.CancelledError):
        interrupted = True
        append_log(batch_log, "[BATCH INTERRUPT] Ctrl+C received; stopping children")
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        completed = refresh_completed_list(output_dir, args.steps, args.tax_period)
        append_log(batch_log, f"[BATCH END] completed_valid={len(completed)}")

    failures = [idx for idx, ok in results if not ok]
    if interrupted:
        return 130
    if failures:
        append_log(batch_log, f"[BATCH FAILED] failed_or_incomplete={failures}")
        return 1
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(amain()))


if __name__ == "__main__":
    main()
