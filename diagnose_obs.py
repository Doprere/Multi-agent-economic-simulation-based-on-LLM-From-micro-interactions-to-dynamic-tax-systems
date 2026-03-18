"""
diagnose_obs.py — 診斷 Gym 環境觀察資料結構，並視覺化地圖 + Agent 位置
執行：
    python diagnose_obs.py

輸出：
    1. obs 結構（每個 agent 的 key, shape/type/value）
    2. world.maps.state 的 channel 順序（解決 translator channel bug）
    3. 地圖視覺化圖（saved to simulation_results/diagnose_map.png）
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

# UTF-8 fix
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np

_PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(_PROJECT_ROOT))
_AI_ECON_PKG = _PROJECT_ROOT / "ai_economist"
if _AI_ECON_PKG.is_dir():
    sys.path.insert(0, str(_AI_ECON_PKG))

from ai_economist import foundation
from llm_agent.config import load_config, make_env_config

# ── 初始化環境 ─────────────────────────────────────────────────
cfg = load_config()
env_config = make_env_config(cfg)
env = foundation.make_env_instance(**env_config)
obs = env.reset()

# ── 1. Obs 結構診斷 ────────────────────────────────────────────
print("=" * 70)
print("  OBS STRUCTURE DIAGNOSIS")
print("=" * 70)
print(f"\n[obs top-level keys]: {list(obs.keys())}\n")

for agent_key in sorted(obs.keys()):
    agent_obs = obs[agent_key]
    print(f"--- obs['{agent_key}'] ---")
    for k, v in agent_obs.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}, "
                  f"min={v.min():.3f}, max={v.max():.3f}, nonzero={np.count_nonzero(v)}")
        else:
            print(f"  {k}: type={type(v).__name__}, value={v}")
    print()

# ── 2. world.maps 的 channel 順序 ─────────────────────────────
print("=" * 70)
print("  world.maps.state channel order")
print("=" * 70)
maps_state = env.world.maps.state  # shape: (n_channels, H, W)
if hasattr(env.world.maps, "_maps"):
    print(f"map layers (in order): {list(env.world.maps._maps.keys())}")
elif hasattr(env.world.maps, "maps"):
    print(f"map layers (in order): {list(env.world.maps.maps.keys())}")
else:
    # 嘗試 resources 屬性
    try:
        print(f"world resources: {env.world.resources}")
    except:
        pass

print(f"\nmaps_state.shape: {maps_state.shape}")
for ch in range(maps_state.shape[0]):
    layer = maps_state[ch]
    print(f"  Channel {ch}: min={layer.min():.3f}, max={layer.max():.3f}, "
          f"nonzero={np.count_nonzero(layer)}")

# ── 3. Agent obs['map'] channel 診斷 ──────────────────────────
print("\n" + "=" * 70)
print("  Agent 0 obs['map'] channel analysis")
print("=" * 70)
agent0_obs = obs.get("0", {})
_tmp = agent0_obs.get("world-map")
agent0_map = _tmp if _tmp is not None else agent0_obs.get("map")
if agent0_map is not None:
    am = np.array(agent0_map)
    print(f"  shape: {am.shape}")
    for ch in range(am.shape[0]):
        layer = am[ch]
        print(f"  Channel {ch}: min={layer.min():.3f}, max={layer.max():.3f}, "
              f"nonzero={np.count_nonzero(layer)}, sum={layer.sum():.3f}")
else:
    print("  ⚠️  obs['0'] has NO 'map' key!")
    print(f"  Available keys: {list(agent0_obs.keys())}")

# ── 4. Agent 位置確認 ──────────────────────────────────────────
print("\n" + "=" * 70)
print("  Agent locations")
print("=" * 70)
for agent in env.world.agents:
    print(f"  Agent {agent.idx}: loc={agent.loc}")

# ── 5. 視覺化地圖 ─────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # 試著找 channel 對應名稱
    channel_names = []
    for attr in ["_maps", "maps"]:
        if hasattr(env.world.maps, attr):
            channel_names = list(getattr(env.world.maps, attr).keys())
            break
    if not channel_names:
        channel_names = [f"ch{i}" for i in range(maps_state.shape[0])]

    W_idx = next((i for i, n in enumerate(channel_names) if "Wood" in n and "Source" not in n), None)
    S_idx = next((i for i, n in enumerate(channel_names) if "Stone" in n and "Source" not in n), None)
    H_idx = next((i for i, n in enumerate(channel_names) if "House" in n or "house" in n), None)

    print(f"\n  channel_names: {channel_names}")
    print(f"  Wood channel: {W_idx}, Stone channel: {S_idx}, House channel: {H_idx}")

    # ── Full map 視覺化 ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor="#1a1a2e")

    # Left: 全地圖（資源分布 + agent 位置）
    ax = axes[0]
    ax.set_facecolor("#0f0f23")
    H, W_sz = maps_state.shape[1], maps_state.shape[2]

    composite = np.zeros((H, W_sz, 3))
    if W_idx is not None:
        wood_layer = maps_state[W_idx]
        composite[:, :, 1] += np.clip(wood_layer / 2.0, 0, 1) * 0.8  # green
    if S_idx is not None:
        stone_layer = maps_state[S_idx]
        composite[:, :, 0] += np.clip(stone_layer / 2.0, 0, 1) * 0.6  # red
        composite[:, :, 2] += np.clip(stone_layer / 2.0, 0, 1) * 0.6  # blue
    if H_idx is not None:
        house_layer = maps_state[H_idx]
        composite[:, :, 0] += np.clip(house_layer / 2.0, 0, 1) * 1.0

    composite = np.clip(composite, 0, 1)
    ax.imshow(composite, interpolation="nearest", aspect="equal")

    # Agent 位置
    colors = ["#FF6B6B", "#4ECDC4", "#FFE66D", "#A29BFE"]
    agent_locs = []
    for agent in env.world.agents:
        r, c = agent.loc
        agent_locs.append((r, c))
        ax.scatter(c, r, s=300, color=colors[agent.idx % len(colors)],
                   edgecolors="white", linewidths=1.5, zorder=5)
        ax.text(c + 0.5, r - 0.5, str(agent.idx), color="white",
                fontsize=8, fontweight="bold", zorder=6)

    ax.set_title("Full World Map + Agent Positions", color="white", fontsize=13, pad=10)
    ax.set_xlabel("Column", color="white")
    ax.set_ylabel("Row", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("white")

    patches = [mpatches.Patch(color="#00CC44", label="Wood")]
    if S_idx is not None:
        patches.append(mpatches.Patch(color="#9966CC", label="Stone"))
    if H_idx is not None:
        patches.append(mpatches.Patch(color="#FF4444", label="House"))
    for i, agent in enumerate(env.world.agents):
        patches.append(mpatches.Patch(color=colors[i % len(colors)],
                                      label=f"Agent {agent.idx}"))
    ax.legend(handles=patches, loc="upper right", framealpha=0.6,
              facecolor="#1a1a2e", labelcolor="white", fontsize=8)

    # Right: Agent 0 egocentric view
    ax2 = axes[1]
    ax2.set_facecolor("#0f0f23")

    if agent0_map is not None:
        am = np.array(agent0_map, dtype=float)
        # 重建 composite
        comp2 = np.zeros((am.shape[1], am.shape[2], 3))

        # channel_names 對應的是 world.maps，但 agent obs 的 map 前幾個 ch 是 world resource channels
        # 後面是 landmark channel，最後是 agent 存在 channel
        # 嘗試相同 channel 映射
        if W_idx is not None and W_idx < am.shape[0]:
            comp2[:, :, 1] += np.clip(am[W_idx] / 2.0, 0, 1) * 0.8
        if S_idx is not None and S_idx < am.shape[0]:
            comp2[:, :, 0] += np.clip(am[S_idx] / 2.0, 0, 1) * 0.6
            comp2[:, :, 2] += np.clip(am[S_idx] / 2.0, 0, 1) * 0.6

        # 最後一個 channel 通常是 water/boundary
        # 標記中心（agent 位置）
        comp2 = np.clip(comp2, 0, 1)
        ax2.imshow(comp2, interpolation="nearest", aspect="equal")

        # 中心點 = agent 自己
        center_r, center_c = am.shape[1] // 2, am.shape[2] // 2
        ax2.scatter(center_c, center_r, s=400, color=colors[0],
                    edgecolors="white", linewidths=2.0, zorder=5, marker="*")

        title_map = f"Agent 0 Egocentric View (shape={am.shape})\n" \
                    f"loc={env.world.agents[0].loc}"
        ax2.set_title(title_map, color="white", fontsize=11, pad=10)

        # 標出每個 channel 的 nonzero counts
        info_lines = []
        for ch in range(am.shape[0]):
            nz = np.count_nonzero(am[ch])
            ch_name = channel_names[ch] if ch < len(channel_names) else f"ch{ch}"
            info_lines.append(f"ch{ch} [{ch_name}]: {nz} nonzero")
        ax2.set_xlabel("\n".join(info_lines), color="#AAAAAA", fontsize=7)
    else:
        ax2.text(0.5, 0.5, "obs['0'] has no 'map' key!\nCheck config 'flatten_observations'",
                 transform=ax2.transAxes, ha="center", va="center",
                 color="#FF6B6B", fontsize=12, fontweight="bold")
    ax2.tick_params(colors="white")
    for spine in ax2.spines.values():
        spine.set_edgecolor("white")

    plt.tight_layout(pad=2.0)
    out_dir = _PROJECT_ROOT / "simulation_results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "diagnose_map.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="#1a1a2e", edgecolor="none")
    print(f"\n[OK] 地圖視覺化已儲存: {out_path}")
    plt.close()

except ImportError:
    print("\n[WARN] matplotlib 未安裝，跳過視覺化。請執行: pip install matplotlib")
except Exception as e:
    import traceback
    print(f"\n[ERROR] 視覺化失敗: {e}")
    traceback.print_exc()

print("\n" + "=" * 70)
print("  DIAGNOSIS COMPLETE")
print("=" * 70)
