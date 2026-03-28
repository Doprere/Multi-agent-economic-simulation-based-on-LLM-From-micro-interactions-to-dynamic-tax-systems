"""
action_map.py — Action ID <-> semantic name mapping.

Single action mode layout (Build=1, CDA max_bid_ask=10, Gather=4):
Foundation sorts resources alphabetically → Stone before Wood.
  0       = NOOP
  1       = Build
  2-12    = Buy Stone  (bid price 0..10)
  13-23   = Sell Stone (ask price 0..10)
  24-34   = Buy Wood   (bid price 0..10)
  35-45   = Sell Wood  (ask price 0..10)
  46      = Move Left
  47      = Move Right
  48      = Move Up
  49      = Move Down
Total: 50 actions (IDs 0-49)
"""
from __future__ import annotations

import numpy as np

MAX_BID_ASK = 10

# ── Build action name table ────────────────────────────────────
_CDA_ACTIONS: dict[int, str] = {}
_base = 2  # 0=NOOP, 1=Build
for _resource in ["Stone", "Wood"]:
    for _price in range(MAX_BID_ASK + 1):
        _CDA_ACTIONS[_base] = f"Buy {_resource} (bid {_price} Coin)"
        _base += 1
    for _price in range(MAX_BID_ASK + 1):
        _CDA_ACTIONS[_base] = f"Sell {_resource} (ask {_price} Coin)"
        _base += 1

ACTION_NAMES: dict[int, str] = {
    0:  "NOOP (no action)",
    1:  "Build House (costs 1 Wood + 1 Stone, earns build income)",
    **_CDA_ACTIONS,
    46: "Move Left",
    47: "Move Right",
    48: "Move Up",
    49: "Move Down",
}

TOTAL_ACTIONS = 50

# Semantic groups
GROUP_NOOP       = {0}
GROUP_BUILD      = {1}
GROUP_BUY_STONE  = set(range(2,  13))
GROUP_SELL_STONE = set(range(13, 24))
GROUP_BUY_WOOD   = set(range(24, 35))
GROUP_SELL_WOOD  = set(range(35, 46))
GROUP_MOVE       = {46, 47, 48, 49}

# CDA decode: action_id -> {resource, side, price}
CDA_DECODE: dict[int, dict] = {}
for _aid in range(2, 46):
    _offset  = _aid - 2
    _res_idx, _rem = divmod(_offset, 22)  # 11 buy + 11 sell = 22 per resource
    _side  = "buy" if _rem < 11 else "sell"
    _price = _rem if _rem < 11 else _rem - 11
    CDA_DECODE[_aid] = {
        "resource": ["Stone", "Wood"][_res_idx],
        "side":     _side,
        "price":    _price,
    }


# ── Public API ─────────────────────────────────────────────────

def get_action_name(action_id: int) -> str:
    return ACTION_NAMES.get(action_id, f"unknown_action({action_id})")


def get_valid_action_ids(mask: np.ndarray) -> list[int]:
    return [int(i) for i, m in enumerate(mask) if m == 1]


def get_valid_actions(mask: np.ndarray) -> list[dict]:
    return [
        {"id": i, "name": ACTION_NAMES.get(i, f"action_{i}")}
        for i, m in enumerate(mask) if m == 1
    ]


def get_masked_description(mask: np.ndarray) -> str:
    """Return a human-readable list of valid actions for LLM prompts."""
    valid = get_valid_actions(mask)
    if not valid:
        return "  (no valid actions — only NOOP available)"
    lines = [f"  [{v['id']:>2}] {v['name']}" for v in valid]
    return "\n".join(lines)


def is_valid_action(action_id: int, mask: np.ndarray) -> bool:
    if action_id < 0 or action_id >= len(mask):
        return False
    return bool(mask[action_id] == 1)


def get_random_valid_action(mask: np.ndarray) -> int:
    valid = get_valid_action_ids(mask)
    return int(np.random.choice(valid)) if valid else 0


# ── Self-test ──────────────────────────────────────────────────
if __name__ == "__main__":
    import io, sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    print(f"Total actions: {TOTAL_ACTIONS}")
    print(f"Action name count: {len(ACTION_NAMES)}")
    assert len(ACTION_NAMES) == TOTAL_ACTIONS, "Action name count mismatch!"

    print("\n=== Full Action Map ===")
    for i in range(TOTAL_ACTIONS):
        print(f"  [{i:>2}] {ACTION_NAMES[i]}")

    print("\n=== CDA Decode Test ===")
    for aid in [2, 12, 13, 23, 24, 34, 35, 45]:
        print(f"  [{aid}] {CDA_DECODE[aid]}")

    print("\n=== Mask Filter Test ===")
    test_mask = np.zeros(TOTAL_ACTIONS)
    test_mask[[0, 1, 46, 47, 48, 49]] = 1
    print(get_masked_description(test_mask))
    print("\n[PASS] action_map self-test complete")
