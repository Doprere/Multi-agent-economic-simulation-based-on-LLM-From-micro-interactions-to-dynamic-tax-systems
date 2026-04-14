"""
visualize_experiments.py — AI Economist 模擬結果視覺化。

產生時間序列圖表：
  A 組（跨實驗比較）：SWF, Gini, Coin, Planner Reward
  B 組（個體 Agent）：Coin 持有, 累積勞動, 資源庫存
  C 組（行為分析）：動作分類比例, Build 事件
  D 組（稅制政策）：稅率結構演化, Gini vs 稅率

  執行方式:python visualize_experiments.py --experiments simulation_results/20260405_015101/expB_qwen2_5_7b_run1 simulation_results/20260405_015101/expC_llama3_1_8b_run1 simulation_results/20260405_015101/expD_qwen2_5_3b_run1 simulation_results/20260406_031441/expC_llama3_1_8b_run1 simulation_results/saez_llama3_1_8b_run1

"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 中文字型設定
# ---------------------------------------------------------------------------
for font_name in ["Microsoft JhengHei", "Noto Sans CJK TC", "SimHei", "Arial Unicode MS"]:
    try:
        matplotlib.font_manager.findfont(font_name, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------------------------
# 常數
# ---------------------------------------------------------------------------
DPI = 300
AGENT_NAMES = {0: "Youth", 1: "Young Adult", 2: "Middle-aged", 3: "Senior"}
AGENT_COLORS = {0: "#0072B2", 1: "#E69F00", 2: "#009E73", 3: "#CC79A7"}
AGENT_MARKERS = {0: "o", 1: "s", 2: "^", 3: "D"}

# 顏色以模型為單位（同模型同色，run 用符號區分）
MODEL_COLORS: dict[str, str] = {
    "llama3.1:8b":  "#1f77b4",   # 藍色
    "qwen2.5:7b":   "#d62728",   # 紅色
    "qwen2.5:3b":   "#9467bd",   # 紫色
    "gemma4:e2b":   "#e377c2",   # 粉紅色
    "gemma4:e4b":   "#8c564b",   # 棕色
    "gpt-4o-mini":  "#2ca02c",   # 綠色
}
# 符號以 run 為單位（run1=圓、run2=方、Saez=星）
RUN_MARKERS: dict[str, str] = {
    "run1": "o",
    "run2": "s",
    "saez": "P",
}

ACTION_GROUPS = {
    "NOOP": {0},
    "Build": {1},
    "Trade": set(range(2, 46)),
    "Move": {46, 47, 48, 49},
}


# ---------------------------------------------------------------------------
# 工具函式
# ---------------------------------------------------------------------------
def detect_model(exp_name: str) -> str:
    """從實驗資料夾名稱推斷模型。"""
    if "llama3_1_8b" in exp_name:
        return "llama3.1:8b"
    if "qwen2_5_7b" in exp_name:
        return "qwen2.5:7b"
    if "qwen2_5_3b" in exp_name:
        return "qwen2.5:3b"
    if "gemma4_e2b" in exp_name:
        return "gemma4:e2b"
    if "gemma4_e4b" in exp_name:
        return "gemma4:e4b"
    if "gpt4o_mini" in exp_name:
        return "gpt-4o-mini"
    return "unknown"


def short_label(exp_dir: Path) -> str:
    """產生簡短的圖例標籤，例如 'llama3.1:8b LLM run1 (500)'。"""
    model = detect_model(exp_dir.name)
    planner_type = "Saez" if "saez" in exp_dir.name.lower() else "LLM"
    run = "run2" if "run2" in exp_dir.name else "run1"
    steps_csv = exp_dir / "step_metrics.csv"
    if steps_csv.exists():
        with open(steps_csv, "r") as f:
            n = sum(1 for _ in f) - 1
    else:
        n = "?"
    return f"{model} {planner_type} {run} ({n}步)"


def model_color(exp_dir: Path) -> str:
    return MODEL_COLORS.get(detect_model(exp_dir.name), "#888888")


def model_linestyle(exp_dir: Path) -> str:
    return "-"


def model_marker(exp_dir: Path) -> str:
    if "saez" in exp_dir.name.lower():
        return RUN_MARKERS["saez"]
    return RUN_MARKERS["run2"] if "run2" in exp_dir.name else RUN_MARKERS["run1"]


def discover_experiments(base: Path, min_steps: int = 400) -> list[Path]:
    """遞迴搜尋所有含 summary.json 且步數 >= min_steps 的實驗。"""
    results: list[Path] = []
    for s in sorted(base.rglob("summary.json")):
        exp_dir = s.parent
        sm = exp_dir / "step_metrics.csv"
        if sm.exists():
            with open(sm, "r") as f:
                n = sum(1 for _ in f) - 1
            if n >= min_steps:
                results.append(exp_dir)
    return results


def load_metrics(exp_dir: Path) -> pd.DataFrame:
    return pd.read_csv(exp_dir / "step_metrics.csv")


def load_actions(exp_dir: Path) -> pd.DataFrame:
    return pd.read_csv(exp_dir / "action_log.csv")


def load_tax(exp_dir: Path) -> list[dict[str, Any]]:
    tax_file = exp_dir / "tax_log.csv"
    if not tax_file.exists():
        return []
    df = pd.read_csv(tax_file)
    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        brackets = ast.literal_eval(str(r["tax_brackets"]))
        rows.append({"step": int(r["step"]), "brackets": brackets})
    return rows


def classify_action(a: int) -> str:
    for name, ids in ACTION_GROUPS.items():
        if a in ids:
            return name
    return "Trade"


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# 圖組 A：跨實驗比較（總體經濟指標）
# ---------------------------------------------------------------------------
def plot_a1_swf(experiments: list[Path], out_dir: Path) -> None:
    """A1. SWF 時間序列比較"""
    fig, ax = plt.subplots(figsize=(12, 5))
    for exp in experiments:
        df = load_metrics(exp)
        swf = (1 - df["gini"]) * df["mean_coin"]
        ax.plot(df["step"], swf, label=short_label(exp),
                color=model_color(exp), linestyle=model_linestyle(exp), linewidth=1.5,
                marker=model_marker(exp), markevery=max(1, len(df) // 12), markersize=5)
    # 稅率調整虛線
    max_step = max(load_metrics(e)["step"].max() for e in experiments)
    for s in range(100, int(max_step) + 1, 100):
        ax.axvline(s, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("SWF = (1 - Gini) × Mean Coin")
    ax.set_title("A1. 社會福利函數 (SWF) 時間序列")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "A1_swf_comparison.png", dpi=DPI)
    plt.close(fig)


def plot_a2_gini(experiments: list[Path], out_dir: Path) -> None:
    """A2. Gini 係數時間序列比較"""
    fig, ax = plt.subplots(figsize=(12, 5))
    for exp in experiments:
        df = load_metrics(exp)
        ax.plot(df["step"], df["gini"], label=short_label(exp),
                color=model_color(exp), linestyle=model_linestyle(exp), linewidth=1.5,
                marker=model_marker(exp), markevery=max(1, len(df) // 12), markersize=5)
    max_step = max(load_metrics(e)["step"].max() for e in experiments)
    for s in range(100, int(max_step) + 1, 100):
        ax.axvline(s, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Gini Coefficient")
    ax.set_title("A2. Gini 係數時間序列")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "A2_gini_comparison.png", dpi=DPI)
    plt.close(fig)


def plot_a3_coin(experiments: list[Path], out_dir: Path) -> None:
    """A3. Total Coin & Mean Coin 時間序列比較"""
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()
    for exp in experiments:
        df = load_metrics(exp)
        lbl = short_label(exp)
        c = model_color(exp)
        ls = model_linestyle(exp)
        mk = model_marker(exp)
        me = max(1, len(df) // 12)
        ax1.plot(df["step"], df["total_coin"], color=c, linestyle=ls,
                 linewidth=1.5, label=f"Total: {lbl}",
                 marker=mk, markevery=me, markersize=5)
        ax2.plot(df["step"], df["mean_coin"], color=c, linestyle=ls,
                 linewidth=0.8, alpha=0.5,
                 marker=mk, markevery=me, markersize=3)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Total Coin")
    ax2.set_ylabel("Mean Coin (半透明)")
    ax1.set_title("A3. 總 Coin / 平均 Coin 時間序列")
    ax1.legend(fontsize=7, loc="upper left")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "A3_coin_comparison.png", dpi=DPI)
    plt.close(fig)


def plot_a4_planner_reward(experiments: list[Path], out_dir: Path) -> None:
    """A4. Planner Reward 時間序列比較（含 rolling avg）"""
    fig, ax = plt.subplots(figsize=(12, 5))
    for exp in experiments:
        df = load_metrics(exp)
        c = model_color(exp)
        ls = model_linestyle(exp)
        lbl = short_label(exp)
        ax.plot(df["step"], df["planner_reward"], color=c, alpha=0.15, linewidth=0.5)
        rolling = df["planner_reward"].rolling(window=20, min_periods=1).mean()
        ax.plot(df["step"], rolling, color=c, linestyle=ls, linewidth=1.5, label=lbl,
                marker=model_marker(exp), markevery=max(1, len(df) // 12), markersize=5)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Planner Reward (20-step rolling avg)")
    ax.set_title("A4. Planner Reward 時間序列")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "A4_planner_reward_comparison.png", dpi=DPI)
    plt.close(fig)


def plot_a5_build_cumulative(experiments: list[Path], out_dir: Path) -> None:
    """A5. 累計 Build 事件跨實驗比較"""
    fig, ax = plt.subplots(figsize=(12, 5))
    for exp in experiments:
        act_df = load_actions(exp)
        build_per_step = sum(
            (act_df[f"action_agent_{i}"] == 1).astype(int)
            for i in range(4)
        )
        cumulative = build_per_step.cumsum()
        ax.plot(act_df["step"], cumulative, label=short_label(exp),
                color=model_color(exp), linestyle=model_linestyle(exp), linewidth=1.5,
                marker=model_marker(exp), markevery=max(1, len(act_df) // 12), markersize=5)
    max_step = max(load_actions(e)["step"].max() for e in experiments)
    for s in range(100, int(max_step) + 1, 100):
        ax.axvline(s, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative Build Count")
    ax.set_title("A5. 累計 Build 事件比較")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "A5_build_cumulative.png", dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 圖組 B：個體經濟指標
# ---------------------------------------------------------------------------
def plot_b1_agent_coin(exp_dir: Path, out_dir: Path) -> None:
    """B1. 各 Agent Coin 持有量"""
    df = load_metrics(exp_dir)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(4):
        ax.plot(df["step"], df[f"coin_agent_{i}"],
                color=AGENT_COLORS[i], linewidth=1.5,
                marker=AGENT_MARKERS[i], markevery=max(1, len(df) // 12), markersize=5,
                label=f"Agent {i} ({AGENT_NAMES[i]})")
    for s in range(100, int(df["step"].max()) + 1, 100):
        ax.axvline(s, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Coin")
    ax.set_title(f"B1. 各 Agent Coin 持有量 — {exp_dir.name}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "B1_agent_coin.png", dpi=DPI)
    plt.close(fig)


def plot_b2_agent_labor(exp_dir: Path, out_dir: Path) -> None:
    """B2. 各 Agent 累積勞動"""
    df = load_metrics(exp_dir)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(4):
        ax.plot(df["step"], df[f"labor_agent_{i}"],
                color=AGENT_COLORS[i], linewidth=1.5,
                marker=AGENT_MARKERS[i], markevery=max(1, len(df) // 12), markersize=5,
                label=f"Agent {i} ({AGENT_NAMES[i]})")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative Labor")
    ax.set_title(f"B2. 各 Agent 累積勞動 — {exp_dir.name}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "B2_agent_labor.png", dpi=DPI)
    plt.close(fig)


def _compute_source_split(
    df: pd.DataFrame, act_df: pd.DataFrame, agent_idx: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """將 agent 的 Wood/Stone 持有量分別分解為「採集來源」與「交易來源」。

    規則:
      - wood/stone delta > 0 且 action 屬 Move(46-49) → 計入 gathered
      - wood/stone delta > 0 且 action 屬 Buy 範圍  → 計入 bought
      - wood/stone delta < 0 (Build/Sell/其他 outflow) → 按當前 gathered/bought
        比例同步扣除 (proportional depletion)
    回傳與 df 同長度的四條 numpy array:
        (g_wood_arr, b_wood_arr, g_stone_arr, b_stone_arr)
    """
    buy_wood_ids = set(range(24, 35))
    buy_stone_ids = set(range(2, 13))
    move_ids = {46, 47, 48, 49}

    wood = df[f"wood_agent_{agent_idx}"].to_numpy(dtype=float)
    stone = df[f"stone_agent_{agent_idx}"].to_numpy(dtype=float)
    # 以 step 對齊 action_log;若長度不一致以 metrics df 為準
    act_series = act_df.set_index("step")[f"action_agent_{agent_idx}"]
    steps = df["step"].to_numpy()

    n = len(steps)
    g_wood_arr = np.zeros(n)
    b_wood_arr = np.zeros(n)
    g_stone_arr = np.zeros(n)
    b_stone_arr = np.zeros(n)

    g_wood = b_wood = 0.0
    g_stone = b_stone = 0.0
    prev_wood = 0.0
    prev_stone = 0.0

    for t in range(n):
        w_now = float(wood[t])
        s_now = float(stone[t])
        w_delta = w_now - prev_wood
        s_delta = s_now - prev_stone

        step_val = int(steps[t])
        action = int(act_series.get(step_val, -1)) if step_val in act_series.index else -1

        # Wood 變化分類
        if w_delta > 0:
            if action in move_ids:
                g_wood += w_delta
            elif action in buy_wood_ids:
                b_wood += w_delta
            else:
                # 未知來源(例:初始稟賦或資料異常)歸為 gathered 作為預設
                g_wood += w_delta
        elif w_delta < 0 and prev_wood > 0:
            frac = min(1.0, (-w_delta) / prev_wood)
            g_wood *= (1.0 - frac)
            b_wood *= (1.0 - frac)

        # Stone 變化分類
        if s_delta > 0:
            if action in move_ids:
                g_stone += s_delta
            elif action in buy_stone_ids:
                b_stone += s_delta
            else:
                g_stone += s_delta
        elif s_delta < 0 and prev_stone > 0:
            frac = min(1.0, (-s_delta) / prev_stone)
            g_stone *= (1.0 - frac)
            b_stone *= (1.0 - frac)

        # 數值穩定:防止負值
        g_wood = max(0.0, g_wood)
        b_wood = max(0.0, b_wood)
        g_stone = max(0.0, g_stone)
        b_stone = max(0.0, b_stone)

        g_wood_arr[t] = g_wood
        b_wood_arr[t] = b_wood
        g_stone_arr[t] = g_stone
        b_stone_arr[t] = b_stone

        prev_wood = w_now
        prev_stone = s_now

    return g_wood_arr, b_wood_arr, g_stone_arr, b_stone_arr


def plot_b3_resource_stock(exp_dir: Path, out_dir: Path) -> None:
    """B3. 各 Agent Wood/Stone 庫存(採集 vs 交易來源)+ Coin 折線 + Build 事件"""
    df = load_metrics(exp_dir)
    act_df = load_actions(exp_dir)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True)
    axes_flat = axes.flatten()

    coin_color = "#d4a017"
    # 4 層顏色:Wood = 棕色系,Stone = 灰色系;gathered 深、bought 淺
    gw_color = "#8c564b"  # Gathered Wood — dark brown
    bw_color = "#c49c94"  # Bought Wood — light brown
    gs_color = "#525252"  # Gathered Stone — dark gray
    bs_color = "#bdbdbd"  # Bought Stone — light gray

    legend_handles: list[Any] = []
    legend_labels: list[str] = []

    for i in range(4):
        ax = axes_flat[i]
        steps = df["step"].to_numpy()
        g_wood, b_wood, g_stone, b_stone = _compute_source_split(df, act_df, i)

        # 左軸:4 層堆疊面積(底 → 頂:gw → bw → gs → bs)
        y0 = np.zeros_like(g_wood)
        y1 = g_wood
        y2 = y1 + b_wood
        y3 = y2 + g_stone
        y4 = y3 + b_stone

        h_gw = ax.fill_between(steps, y0, y1, alpha=0.75, color=gw_color,
                               label="Gathered Wood")
        h_bw = ax.fill_between(steps, y1, y2, alpha=0.75, color=bw_color,
                               label="Bought Wood")
        h_gs = ax.fill_between(steps, y2, y3, alpha=0.75, color=gs_color,
                               label="Gathered Stone")
        h_bs = ax.fill_between(steps, y3, y4, alpha=0.75, color=bs_color,
                               label="Bought Stone")

        ax.set_ylabel("Units", fontsize=8)
        ax.tick_params(axis="y", labelsize=7)

        # 右軸:Coin 折線
        ax_twin = ax.twinx()
        coin_line, = ax_twin.plot(
            df["step"], df[f"coin_agent_{i}"],
            color=coin_color, linewidth=1.8,
            marker=AGENT_MARKERS[i], markevery=max(1, len(df) // 12),
            markersize=5, markerfacecolor=coin_color,
            markeredgecolor="black", markeredgewidth=0.4,
            label="Coin",
        )
        ax_twin.set_ylabel("Coin", fontsize=8, color=coin_color)
        ax_twin.tick_params(axis="y", labelsize=7, colors=coin_color)

        # Build 事件紅線
        build_steps = act_df.loc[act_df[f"action_agent_{i}"] == 1, "step"]
        build_handle = None
        for bs in build_steps:
            build_handle = ax.axvline(bs, color="red", alpha=0.5, linewidth=0.8)

        ax.set_title(f"Agent {i} ({AGENT_NAMES[i]})", fontsize=9)
        ax.grid(True, axis="x", alpha=0.2)

        if i == 0:
            legend_handles = [h_gw, h_bw, h_gs, h_bs, coin_line]
            legend_labels = [
                "Gathered Wood", "Bought Wood",
                "Gathered Stone", "Bought Stone",
                "Coin",
            ]
            if build_handle is not None:
                legend_handles.append(build_handle)
                legend_labels.append("Build 事件")

    axes_flat[2].set_xlabel("Step")
    axes_flat[3].set_xlabel("Step")

    if legend_handles:
        fig.legend(
            legend_handles, legend_labels,
            loc="upper center", ncol=len(legend_handles),
            fontsize=8, bbox_to_anchor=(0.5, 0.97),
            frameon=True,
        )
    fig.suptitle(
        f"B3. Wood/Stone 庫存(採集 vs 交易)+ Coin 折線 + Build 事件 — {exp_dir.name}",
        fontsize=11, y=1.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_dir / "B3_resource_stock.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 圖組 C：行為分析
# ---------------------------------------------------------------------------
def plot_c1_action_distribution(exp_dir: Path, out_dir: Path) -> None:
    """C1. 動作分類比例時間序列（20 步滑動窗口，堆疊面積）"""
    act_df = load_actions(exp_dir)
    window = 20
    categories = ["NOOP", "Build", "Trade", "Move"]
    cat_colors = {"NOOP": "#bdbdbd", "Build": "#e41a1c", "Trade": "#377eb8", "Move": "#4daf4a"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes_flat = axes.flatten()
    for i in range(4):
        ax = axes_flat[i]
        actions = act_df[f"action_agent_{i}"]
        classified = actions.apply(classify_action)
        # one-hot encode
        cat_cols = {c: (classified == c).astype(float) for c in categories}
        # rolling ratio
        ratios = {c: cat_cols[c].rolling(window=window, min_periods=1).mean() for c in categories}
        steps = act_df["step"]
        bottom = np.zeros(len(steps))
        for c in categories:
            vals = ratios[c].values
            ax.fill_between(steps, bottom, bottom + vals, alpha=0.7,
                            color=cat_colors[c], label=c if i == 0 else None)
            bottom = bottom + vals
        ax.set_ylim(0, 1)
        ax.set_title(f"Agent {i} ({AGENT_NAMES[i]})", fontsize=9)
        ax.set_ylabel("比例")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    axes_flat[2].set_xlabel("Step")
    axes_flat[3].set_xlabel("Step")
    fig.legend(categories, loc="upper center", ncol=4, fontsize=9,
              bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"C1. 動作分類比例 (20步滑動窗口) — {exp_dir.name}", fontsize=11, y=1.05)
    fig.tight_layout()
    fig.savefig(out_dir / "C1_action_distribution.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_c2_build_events(exp_dir: Path, out_dir: Path) -> None:
    """C2. Build 事件散點圖"""
    act_df = load_actions(exp_dir)
    df = load_metrics(exp_dir)
    fig, ax = plt.subplots(figsize=(10, 3))
    for i in range(4):
        build_mask = act_df[f"action_agent_{i}"] == 1
        build_steps = act_df.loc[build_mask, "step"]
        if len(build_steps) > 0:
            # Coin 增量色彩
            coin_vals = df.loc[df["step"].isin(build_steps), f"coin_agent_{i}"].values
            ax.scatter(build_steps, [i] * len(build_steps), c=coin_vals,
                      cmap="YlOrRd", s=40, edgecolors="black", linewidths=0.5,
                      zorder=3, label=f"Agent {i}" if len(build_steps) > 0 else None)
    ax.set_yticks(range(4))
    ax.set_yticklabels([f"Agent {i}\n({AGENT_NAMES[i]})" for i in range(4)], fontsize=8)
    ax.set_xlabel("Step")
    ax.set_title(f"C2. Build 事件分布 — {exp_dir.name}")
    ax.grid(True, axis="x", alpha=0.3)
    if any((act_df[f"action_agent_{i}"] == 1).any() for i in range(4)):
        cbar = fig.colorbar(ax.collections[0], ax=ax, label="Coin at Build", shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_dir / "C2_build_events.png", dpi=DPI)
    plt.close(fig)


def plot_c3_trade_success(exp_dir: Path, out_dir: Path) -> None:
    """C3. 各 Agent 交易成功率（買入成功 vs 賣出成功，滑動窗口）"""
    df = load_metrics(exp_dir)
    act_df = load_actions(exp_dir)
    window = 20

    # Buy Stone: 2-12, Sell Stone: 13-23, Buy Wood: 24-34, Sell Wood: 35-45
    buy_range = list(range(2, 13)) + list(range(24, 35))    # all buy actions
    sell_range = list(range(13, 24)) + list(range(35, 46))  # all sell actions

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes_flat = axes.flatten()

    for i in range(4):
        ax = axes_flat[i]
        actions = act_df[f"action_agent_{i}"]
        stone_delta = df[f"stone_agent_{i}"].diff().fillna(0)
        wood_delta = df[f"wood_agent_{i}"].diff().fillna(0)

        # --- Buy attempts & success ---
        buy_stone_mask = actions.between(2, 12)
        buy_wood_mask = actions.between(24, 34)
        buy_attempt = (buy_stone_mask | buy_wood_mask).astype(int)
        buy_success = (
            (buy_stone_mask & (stone_delta > 0)) |
            (buy_wood_mask & (wood_delta > 0))
        ).astype(int)

        # --- Sell attempts & success ---
        sell_stone_mask = actions.between(13, 23)
        sell_wood_mask = actions.between(35, 45)
        sell_attempt = (sell_stone_mask | sell_wood_mask).astype(int)
        sell_success = (
            (sell_stone_mask & (stone_delta < 0)) |
            (sell_wood_mask & (wood_delta < 0))
        ).astype(int)

        # Cumulative
        steps = act_df["step"]
        ax.plot(steps, buy_success.cumsum(), color="#2166ac", linewidth=1.5,
                label=f"Buy Success ({buy_success.sum()})")
        ax.plot(steps, buy_attempt.cumsum(), color="#2166ac", linewidth=0.8,
                linestyle="--", alpha=0.5, label=f"Buy Attempt ({buy_attempt.sum()})")
        ax.plot(steps, sell_success.cumsum(), color="#b2182b", linewidth=1.5,
                label=f"Sell Success ({sell_success.sum()})")
        ax.plot(steps, sell_attempt.cumsum(), color="#b2182b", linewidth=0.8,
                linestyle="--", alpha=0.5, label=f"Sell Attempt ({sell_attempt.sum()})")

        # Success rate annotation
        buy_rate = buy_success.sum() / max(buy_attempt.sum(), 1) * 100
        sell_rate = sell_success.sum() / max(sell_attempt.sum(), 1) * 100
        ax.text(0.98, 0.02, f"Buy rate: {buy_rate:.1f}%\nSell rate: {sell_rate:.1f}%",
                transform=ax.transAxes, fontsize=8, ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

        ax.set_title(f"Agent {i} ({AGENT_NAMES[i]})", fontsize=10)
        ax.set_ylabel("Cumulative Count")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)

    axes_flat[2].set_xlabel("Step")
    axes_flat[3].set_xlabel("Step")
    fig.suptitle(f"C3. 交易成功分析 (累計買入/賣出) — {exp_dir.name}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "C3_trade_success.png", dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 圖組 D：稅制政策
# ---------------------------------------------------------------------------
def plot_d1_tax_progression(exp_dir: Path, out_dir: Path) -> None:
    """D1. 稅率累進結構演化（小倍數圖）"""
    tax_data = load_tax(exp_dir)
    if not tax_data:
        return
    n = len(tax_data)
    cols = min(n, 5)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False)
    # 自動偵測稅率範圍（Saez float 0-1 vs LLM int 0-21）
    max_tax_val = max(max(td["brackets"]) for td in tax_data)
    is_float_tax = max_tax_val <= 1.0
    for idx, td in enumerate(tax_data):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        brackets = td["brackets"]
        tiers = list(range(1, len(brackets) + 1))
        ax.bar(tiers, brackets, color="#4c72b0", alpha=0.8, edgecolor="black", linewidth=0.5)
        ax.set_title(f"Step {td['step']}", fontsize=9)
        ax.set_ylim(0, 1.05 if is_float_tax else 22)
        if is_float_tax:
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.set_xlabel("Tier")
        ax.set_ylabel("Tax Rate")
    # 隱藏多餘子圖
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)
    fig.suptitle(f"D1. 稅率累進結構演化 — {exp_dir.name}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "D1_tax_progression.png", dpi=DPI)
    plt.close(fig)


def plot_d2_gini_vs_tax(exp_dir: Path, out_dir: Path) -> None:
    """D2. Gini vs 稅率對照圖（上: Gini 曲線, 下: 稅率 heatmap）"""
    df = load_metrics(exp_dir)
    tax_data = load_tax(exp_dir)
    if not tax_data:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})
    # 上圖：Gini
    ax1.plot(df["step"], df["gini"], color="#d62728", linewidth=1.2)
    for td in tax_data:
        ax1.axvline(td["step"], color="gray", linestyle=":", alpha=0.5)
    ax1.set_ylabel("Gini Coefficient")
    ax1.set_title(f"D2. Gini vs 稅率變化 — {exp_dir.name}")
    ax1.grid(True, alpha=0.3)

    # 下圖：稅率 heatmap
    max_step = int(df["step"].max())
    n_tiers = len(tax_data[0]["brackets"])
    tax_matrix = np.full((n_tiers, max_step + 1), np.nan)
    sorted_tax = sorted(tax_data, key=lambda x: x["step"])
    for idx, td in enumerate(sorted_tax):
        start = td["step"]
        end = sorted_tax[idx + 1]["step"] if idx + 1 < len(sorted_tax) else max_step + 1
        for tier_idx, val in enumerate(td["brackets"]):
            tax_matrix[tier_idx, start:end] = val

    # 動態偵測稅率範圍
    max_tax_val = float(np.nanmax(tax_matrix))
    is_float_tax = max_tax_val <= 1.0
    vmax = 1.0 if is_float_tax else 21
    cbar_label = "Tax Rate (%)" if is_float_tax else "Tax Rate (0-21)"
    im = ax2.imshow(tax_matrix, aspect="auto", cmap="YlOrRd",
                    extent=[0, max_step, n_tiers + 0.5, 0.5],
                    interpolation="nearest", vmin=0, vmax=vmax)
    ax2.set_ylabel("Tax Tier")
    ax2.set_xlabel("Step")
    ax2.set_yticks(range(1, n_tiers + 1))
    fig.colorbar(im, ax=ax2, label=cbar_label, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_dir / "D2_gini_vs_tax.png", dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="AI Economist 模擬結果視覺化")
    parser.add_argument("--all", action="store_true", help="掃描全部 500+ 步實驗")
    parser.add_argument("--experiments", nargs="*", help="指定實驗資料夾路徑")
    parser.add_argument("--base", default="simulation_results", help="結果根目錄")
    args = parser.parse_args()

    base = Path(args.base)
    if not base.exists():
        print(f"Error: {base} not found")
        sys.exit(1)

    if args.all:
        experiments = discover_experiments(base, min_steps=400)
    elif args.experiments:
        experiments = [Path(e) for e in args.experiments]
    else:
        experiments = discover_experiments(base, min_steps=400)

    if not experiments:
        print("No experiments found.")
        sys.exit(1)

    print(f"Found {len(experiments)} experiments:")
    for e in experiments:
        print(f"  - {e.relative_to(base)}")

    # A 組：跨實驗比較
    comp_dir = ensure_dir(base / "comparison_charts")
    print("\n=== 圖組 A：跨實驗比較 ===")
    print("  A1. SWF 時間序列...")
    plot_a1_swf(experiments, comp_dir)
    print("  A2. Gini 係數...")
    plot_a2_gini(experiments, comp_dir)
    print("  A3. Total/Mean Coin...")
    plot_a3_coin(experiments, comp_dir)
    print("  A4. Planner Reward...")
    plot_a4_planner_reward(experiments, comp_dir)
    print("  A5. Build Cumulative...")
    plot_a5_build_cumulative(experiments, comp_dir)

    # B, C, D 組：各實驗個體圖
    for exp in experiments:
        chart_dir = ensure_dir(exp / "charts")
        exp_label = exp.relative_to(base)
        print(f"\n=== {exp_label} ===")

        print("  B1. Agent Coin...")
        plot_b1_agent_coin(exp, chart_dir)
        print("  B2. Agent Labor...")
        plot_b2_agent_labor(exp, chart_dir)
        print("  B3. Resource Stock...")
        plot_b3_resource_stock(exp, chart_dir)
        print("  C1. Action Distribution...")
        plot_c1_action_distribution(exp, chart_dir)
        print("  C2. Build Events...")
        plot_c2_build_events(exp, chart_dir)
        print("  C3. Trade Success...")
        plot_c3_trade_success(exp, chart_dir)
        print("  D1. Tax Progression...")
        plot_d1_tax_progression(exp, chart_dir)
        print("  D2. Gini vs Tax...")
        plot_d2_gini_vs_tax(exp, chart_dir)

    print(f"\nDone! Comparison charts: {comp_dir}")
    print(f"Per-experiment charts: <experiment>/charts/")


if __name__ == "__main__":
    main()
