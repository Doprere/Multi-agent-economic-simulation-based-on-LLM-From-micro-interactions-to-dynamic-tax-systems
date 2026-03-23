"""
translator.py — Convert Gym obs dict into natural-language state descriptions for LLM prompts.
All output is in English to avoid encoding issues when passed as LLM context.
"""
from __future__ import annotations

import numpy as np

from .action_map import get_masked_description


# ── Helpers ────────────────────────────────────────────────────

def _extract_resource_positions(visible_map: np.ndarray, channel_names: list[str] | None = None) -> dict[str, list[tuple]]:
    """Extract resource positions from visible map (relative to center).

    channel_names: ordered list from world.maps._maps.keys().
    If None, fallback to Foundation default order: Stone=0, Wood=1.
    """
    resources_found: dict[str, list[tuple]] = {"Wood": [], "Stone": []}
    h, w = visible_map.shape[1], visible_map.shape[2]
    center_r, center_c = h // 2, w // 2

    # Foundation 的 map channel 順序由 world.maps._maps 決定，預設為 alphabetical:
    # Stone=0, Wood=1, House=2, Water=3, StoneSourceBlock=4, WoodSourceBlock=5
    if channel_names is not None:
        CHANNEL_MAP = {}
        for i, name in enumerate(channel_names):
            if name == "Wood":
                CHANNEL_MAP[i] = "Wood"
            elif name == "Stone":
                CHANNEL_MAP[i] = "Stone"
    else:
        # 正確的預設 (alphabetical order)：Stone=0, Wood=1
        CHANNEL_MAP = {0: "Stone", 1: "Wood"}

    for ch, name in CHANNEL_MAP.items():
        if ch >= visible_map.shape[0]:
            continue
        positions = np.argwhere(visible_map[ch] > 0)
        for r, c in positions:
            rel_r = int(r) - center_r
            rel_c = int(c) - center_c
            resources_found[name].append((rel_r, rel_c))
    return resources_found


def _direction_desc(r: int, c: int) -> str:
    """Convert relative (row, col) offset to human-readable direction description.

    Positive row = South, positive col = East.
    """
    dist = abs(r) + abs(c)
    if dist == 0:
        return "on your tile (dist=0)"
    if dist == 1:
        # Adjacent tile — reachable next step
        if r == -1:
            direction = "immediately North"
        elif r == 1:
            direction = "immediately South"
        elif c == -1:
            direction = "immediately West"
        else:
            direction = "immediately East"
        return f"{direction} — nearby (reachable next step!)"

    # Multi-step: describe with direction language
    parts = []
    if r < 0:
        parts.append(f"{abs(r)} North")
    elif r > 0:
        parts.append(f"{abs(r)} South")
    if c < 0:
        parts.append(f"{abs(c)} West")
    elif c > 0:
        parts.append(f"{abs(c)} East")
    return f"{dist} steps away ({', '.join(parts)})"


def _format_positions(positions: list[tuple]) -> str:
    """Format resource positions with direction language relative to agent.

    Adjacent tiles (dist=1) are marked as 'nearby (reachable next step!)'.
    Others use direction language like '5 steps away (3 South, 2 West)'.
    """
    if not positions:
        return "none visible"
    parts = []
    for r, c in sorted(positions, key=lambda p: abs(p[0]) + abs(p[1]))[:8]:
        parts.append(_direction_desc(r, c))
    suffix = f" (+{len(positions)-8} more)" if len(positions) > 8 else ""
    return "; ".join(parts) + suffix


def _extract_market_info(obs: dict) -> dict:
    """Parse CDA market observations from obs dict."""
    market: dict[str, dict] = {}
    prefix = "ContinuousDoubleAuction-"
    for resource in ["Wood", "Stone"]:
        asks = obs.get(f"{prefix}available_asks-{resource}")
        bids = obs.get(f"{prefix}available_bids-{resource}")
        rate = obs.get(f"{prefix}market_rate-{resource}", None)
        my_bids = obs.get(f"{prefix}my_bids-{resource}")
        my_asks = obs.get(f"{prefix}my_asks-{resource}")

        lowest_ask = highest_bid = None
        if asks is not None and np.sum(asks) > 0:
            nz = np.nonzero(asks)[0]
            lowest_ask = int(nz[0]) if len(nz) > 0 else None
        if bids is not None and np.sum(bids) > 0:
            nz = np.nonzero(bids)[0]
            highest_bid = int(nz[-1]) if len(nz) > 0 else None

        market[resource] = {
            "lowest_ask":    lowest_ask,
            "highest_bid":   highest_bid,
            "market_rate":   round(float(rate), 2) if rate is not None else None,
            "my_bids_count": int(np.sum(my_bids)) if my_bids is not None else 0,
            "my_asks_count": int(np.sum(my_asks)) if my_asks is not None else 0,
        }
    return market


def _gini(values: np.ndarray) -> float:
    values = np.sort(np.abs(values))
    n = len(values)
    if n == 0 or np.sum(values) == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float(
        (2 * np.dot(idx, values) - (n + 1) * np.sum(values))
        / (n * np.sum(values))
    )


def _get_current_tax_info(env) -> str:
    try:
        tax_comp = env.get_component("PeriodicBracketTax")
        brackets = getattr(tax_comp, "curr_marginal_rates", None)
        if brackets is None:
            brackets = getattr(tax_comp, "tax_rates", [])
        if brackets is not None:
            rates = [f"{r*100:.1f}%" for r in brackets]
            return f"  Current marginal tax rates (brackets): {', '.join(rates)}"
        return "  (Tax rate info unavailable)"
    except (KeyError, AttributeError):
        return "  (Tax component not active or unreadable)"


# ── Main Translator ────────────────────────────────────────────

class ObsTranslator:
    """Translate Foundation obs dicts into English LLM-readable text."""

    def translate_agent_obs(
        self,
        obs: dict,
        agent_id: int,
        agent_name: str,
        agent_role: str,
        env,
        step: int,
    ) -> str:
        agent = env.get_agent(str(agent_id))
        inv   = agent.inventory
        labor = agent.endogenous.get("Labor", 0.0)
        loc   = agent.loc
        build_pay = agent.state.get("build_payment", "N/A")
        build_pay_str = f"{float(build_pay):.1f} Coin/house" if build_pay != "N/A" else "N/A"

        # ── 地圖觀察 ────────────────────────────────────────────
        # obs key 是 'world-map'（Foundation 的 env_wrapper flatten 前綴）
        _vm = obs.get("world-map")
        visible_map = _vm if _vm is not None else obs.get("map")

        # 動態取得 channel 順序（world.maps._maps 的 key 順序）
        channel_names: list[str] | None = None
        try:
            channel_names = list(env.world.maps._maps.keys())
        except AttributeError:
            pass

        if visible_map is not None:
            resource_pos = _extract_resource_positions(
                np.array(visible_map), channel_names=channel_names
            )
        else:
            resource_pos = {"Wood": [], "Stone": []}

        market = _extract_market_info(obs)
        mask   = np.array(obs.get("action_mask", np.zeros(50)), dtype=np.float32)

        w_pos = _format_positions(resource_pos.get("Wood", []))
        s_pos = _format_positions(resource_pos.get("Stone", []))

        wm = market.get("Wood", {})
        sm = market.get("Stone", {})

        def ask_s(m): a = m.get("lowest_ask"); return f"{a} Coin" if a is not None else "no asks"
        def bid_s(m): b = m.get("highest_bid"); return f"{b} Coin" if b is not None else "no bids"

        lines = [
            f"=== Step {step}/1000 | Agent {agent_id} ({agent_name}) ===",
            f"Role: {agent_role.strip()}",
            "",
            "[Game Rules]",
            "  - Movement: Left(46), Right(47), Up(48), Down(49). Each move costs labor.",
            "  - Gathering: Move onto a resource tile (dist=0) to collect automatically.",
            "  - Building: Requires 1 Wood + 1 Stone. Earns Coin based on your build skill. Action 1.",
            "  - Trading: Submit buy/sell orders on the market. A trade executes when a bid >= an ask.",
            "    Buy Wood (actions 2-12): Spend Coin to buy Wood at bid price 0-10.",
            "    Sell Wood (actions 13-23): Sell your Wood at ask price 0-10.",
            "    Buy Stone (actions 24-34): Spend Coin to buy Stone at bid price 0-10.",
            "    Sell Stone (actions 35-45): Sell your Stone at ask price 0-10.",
            "  - Ways to earn Coin: Build houses OR sell resources to other agents on the market.",
            "  - NOOP (action 0): Do nothing this step.",
            "",
            "[Personal Status]",
            f"  Location  : row={loc[0]}, col={loc[1]}",
            f"  Inventory : Coin={inv.get('Coin', 0):.1f}, "
            f"Wood={inv.get('Wood', 0)}, Stone={inv.get('Stone', 0)}",
            f"  Labor used: {labor:.2f}",
            f"  Build income per house: {build_pay_str}",
            "",
            "[Visible Resources]",
            f"  Wood  : {w_pos}",
            f"  Stone : {s_pos}",
            "",
            "[Market Status]",
            f"  Wood  — lowest ask={ask_s(wm)}, highest bid={bid_s(wm)}, "
            f"avg price~{wm.get('market_rate', 'N/A')} Coin",
            f"  Stone — lowest ask={ask_s(sm)}, highest bid={bid_s(sm)}, "
            f"avg price~{sm.get('market_rate', 'N/A')} Coin",
            f"  My open orders — Wood: {wm.get('my_bids_count',0)} buy / "
            f"{wm.get('my_asks_count',0)} sell  |  "
            f"Stone: {sm.get('my_bids_count',0)} buy / {sm.get('my_asks_count',0)} sell",
            "",
            "[Valid Actions] (only choose from the action_ids listed below)",
            get_masked_description(mask),
        ]
        return "\n".join(lines)

    def translate_planner_obs(
        self,
        obs: dict,
        env,
        step: int,
        is_tax_day: bool = False,
    ) -> str:
        agents = env.world.agents
        coin_endowments = np.array([a.total_endowment("Coin") for a in agents])
        mean_coin = float(np.mean(coin_endowments))
        gini = _gini(coin_endowments)

        agent_lines = []
        for a in agents:
            inv   = a.inventory
            labor = a.endogenous.get("Labor", 0.0)
            agent_lines.append(
                f"  Agent {a.idx}: Coin={inv.get('Coin',0):.1f}, "
                f"Wood={inv.get('Wood',0)}, Stone={inv.get('Stone',0)}, "
                f"Labor={labor:.1f}"
            )

        tax_info = _get_current_tax_info(env)
        header = ">>> TAX ADJUSTMENT DAY — output tax_brackets this step! <<<" \
                 if is_tax_day else "(Regular observation step — output NOOP action)"

        lines = [
            f"=== Step {step}/1000 | Social Planner — Global Observation ===",
            header,
            "",
            "[Wealth Distribution]",
            *agent_lines,
            f"  Mean Coin : {mean_coin:.1f}",
            f"  Gini Index: {gini:.4f}  (0 = perfect equality, 1 = max inequality)",
            "",
            "[Tax Information]",
            tax_info,
        ]
        return "\n".join(lines)


# ── CLI demo ───────────────────────────────────────────────────
if __name__ == "__main__":
    import io, sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

    from ai_economist import foundation
    from llm_agent.config import load_config, make_env_config

    cfg = load_config()
    env = foundation.make_env_instance(**make_env_config(cfg))
    obs = env.reset()

    t = ObsTranslator()
    persona = cfg.personas[0]
    print(t.translate_agent_obs(obs.get("0", {}), 0,
                                persona.display_name, persona.role, env, 0))
    print("\n--- Planner ---\n")
    print(t.translate_planner_obs({}, env, 0, is_tax_day=True))
