# PROMPTS INVENTORY — 論文/project

> 完整原文為主，<span style="color:red">紅字</span>標示疑似過度引導處，附簡短修改建議。
> **註**：`action_map.py` 中交易動作佔 44/50 (88%) 的結構性問題，已確認**不列入修改**。

---

## 1. `llm_agent/agent.py`

### 1.1 `FEW_SHOT_EXAMPLES`（lines 34–57）

**改動後（現行版本）**：

```
Here are a few response examples (action_id MUST be from the valid action list):

Example 1 (build — materials ready):
{"thought": "I have Wood=1 and Stone=1 — the exact materials needed to build. Build payout scales with my skill; converting these materials into Coin this step.", "action_id": 1}

Example 2 (gather — adjacent resource):
{"thought": "A Wood tile is immediately Right (dist=1) and my Wood=0. Moving Right to collect it.", "action_id": 47}

Example 3 (gather — distant resource):
{"thought": "Nearest Stone is 3 steps away (2 Down, 1 Right). No resources adjacent this step. Moving Down first to approach.", "action_id": 49}

Example 4 (move — blocked direction, reroute):
{"thought": "Wood is 2 tiles to the Right, but Move Right is in the blocked list — the tile to the right is not a valid movable area this step. Moving Down to reroute around.", "action_id": 49}

Example 5 (buy — acquire missing build material):
{"thought": "I have Stone=1 but Wood=0, and no Wood tiles are visible nearby. The Wood market shows a lowest ask of 3 Coin. Placing a bid of 3 on Wood.", "action_id": 27}

Example 6 (sell — surplus for Coin):
{"thought": "My Stone=3 is more than I need for building. The Stone market shows a highest bid of 5 Coin — selling my surplus unit there would earn me 5 Coin. Placing an ask of 5 on Stone.", "action_id": 18}

Example 7 (build — after gathering):
{"thought": "I just collected the last Stone I needed. Wood=2, Stone=1, and I am standing on a buildable tile. Building this step to turn the gathered materials into Coin.", "action_id": 1}
```

> **📌 改動前原文**（6 例含 NOOP）：
>
> ```
> Here are a few response examples (action_id MUST be from the valid action list):
>
> Example 1 (gather — move toward resource):
> {"thought": "Wood is 2 tiles to the Right, my Wood=0. I need it to build. Moving right to collect.", "action_id": 47}
>
> Example 2 (gather — move toward distant resource):
> {"thought": "Stone is 3 steps away (2 Down, 1 Right). No resources nearby. Moving Down first to get closer.", "action_id": 49}
>
> Example 3 (build):
> {"thought": "I have Wood=1, Stone=1. Building now earns 15 Coin — worth the labor.", "action_id": 1}
>
> Example 4 (buy on market):
> {"thought": "I need Stone to build. Lowest ask for Stone is 4 Coin. I'll bid 4 to match and buy immediately.", "action_id": 6}
>
> Example 5 (sell on market):
> {"thought": "I have Stone=3 which exceeds my build needs. Highest bid for Stone is 5 Coin. I'll ask 5 to sell immediately.", "action_id": 18}
>
> Example 6 (NOOP — only when truly idle):
> {"thought": "No resources visible, cannot build, all orders pending. Nothing productive to do this step.", "action_id": 0}
> ```

**修正要點**：
- 移除 NOOP 範例，避免小模型把「無所事事」學成合法策略
- 新增 Build 範例到 Example 1（primacy effect）與 Example 7（recency effect），將建造置於首尾兩個注意力高位
- 新增 Example 4「遇到 blocked direction 重新繞路」，教模型處理障礙而非僵死
- buy/sell 範例移除 `"match...immediately"` / `"sell immediately"` 等鼓勵即時吃單的詞彙，改為描述「掛單價 = 市場當前價位」的中性觀察

**過度引導處**：
- ~~Example 4：<span style="color:red">"I'll bid 4 to match and buy immediately"</span>~~ ✅ **已修正（Step B）**
- ~~Example 5：<span style="color:red">"I'll ask 5 to sell immediately"</span>~~ ✅ **已修正（Step B）**
- ~~Example 6：<span style="color:red">"only when truly idle"</span> + <span style="color:red">"Nothing productive to do"</span>~~ ✅ **已修正（Step B，整個 NOOP 範例已移除）**
- ~~整體結構：buy/sell 範例 thought 比 build 更長更具體（primacy effect）~~ ✅ **已修正（Build 放在 Example 1 和 Example 7）**

---

### 1.2 `_build_system_prompt`（lines 91–122）

**改動後（現行版本）**：

```
You are an Agent in the AI Economist simulation, representing '{display_name}'.
Your role: {role}

Your happiness = Coin earned minus effort spent.
- Coin outcomes depend on the environment, including building activity and market trades.
- Every action (moving, gathering, building, trading) costs some labor.
- Labor has a cost and should be weighed against possible Coin outcomes.
{stamina_description}

Decision rules:
1. Choose exactly ONE action_id per step. It MUST appear in the [Valid Actions] list.
2. IMPORTANT: If an action_id is NOT in the [Valid Actions] list, it is BLOCKED by the environment and your turn will be wasted. Always check the list before choosing.
3. Movement directions may be blocked by walls, water, or other agents. If a direction is not in [Valid Actions], do NOT attempt it — choose a different direction or action.
4. Output MUST be strict JSON: {"thought": "...", "action_id": <integer>}

{FEW_SHOT_EXAMPLES}
```

> **📌 改動前原文**：
>
> ```
> You are an Agent in the AI Economist simulation, representing '{display_name}'.
> Your role: {role}
>
> Your happiness = Coin earned minus effort spent.
> - Gather Wood and Stone, then Build houses or Sell on market to earn Coin.
> - Every action (moving, gathering, building, trading) costs some labor.
> - Spending labor to gather and build is an investment — it pays off in Coin.
> {stamina_description}
> ...
> ```

Stamina 自然語言對照（labor_cost_modifier → 描述）（未改動）：
- `lcm <= 0.7`: "You have good stamina — physical tasks cost you relatively little effort."
- `lcm <= 1.0`: "Physical tasks require a moderate amount of effort."
- `lcm <= 1.3`: "Physical tasks require more effort than they used to."
- `lcm > 1.3`: "Physical tasks cost you more effort than most — pace yourself accordingly."

**修正要點**：
- 移除 `"Gather Wood and Stone, then Build houses or Sell on market to earn Coin"` 這條把 Build 與 Sell 並列為「賺錢管道」的敘述，改為 `"Coin outcomes depend on the environment, including building activity and market trades."` 的中性事實陳述
- 移除 `"Spending labor to gather and build is an investment — it pays off in Coin"` 這條單方面的正向價值判斷，改為 `"Labor has a cost and should be weighed against possible Coin outcomes."`，讓模型自行權衡成本與回報而非預設 gather/build 必然划算

**過度引導處**：
- ~~Line 108：<span style="color:red">"Spending labor to gather and build is an investment — it pays off in Coin."</span>~~ ✅ **已修正（Step B）**
- ~~Line 106：<span style="color:red">"Build houses or Sell on market to earn Coin"</span>~~ ✅ **已修正（Step B）**

---

### 1.3 Retry Hint（line 210）

```
WARNING: Your last response was invalid: {retry_hint}
Please choose a valid action_id.
```

retry_hint 範例（lines 225–228）：
```
action_id={X} is NOT in the valid action list. Valid action_ids: [0, 1, 46, 47, 48, 49]...
```

**評估**：中性，無過度引導。

---

### 1.4 Fallback Thought（line 245）

```
(fallback: LLM retries exhausted)
```

**評估**：內部紀錄，不進 prompt。

---

## 2. `llm_agent/planner.py`

### 2.1 `_observe_system_prompt`（lines 55–61）

```
You are the '{display_name}'. {role}

Each step you observe the socioeconomic state and accumulate understanding through memory.
This is NOT a tax-setting step — focus on observation and analysis only.
Output format: {"thought": "...", "society_comment": "..."}
```

**評估**：中性,無過度引導。

---

### 2.2 `_tax_system_prompt`（lines 63–73) ★ CRITICAL

**改動後（現行版本）**：

```
You are the '{display_name}'. {role}

TAX ADJUSTMENT TIME! Choose tax brackets justified by your observations.

Tax bracket rules:
- tax_brackets is an integer list; each element is the tax rate index (0-21) for that bracket.
- Index 0 = 0% tax rate, index 21 = 100% tax rate (~5% per step).
- Tax brackets may be flat, progressive, or regressive; choose the pattern justified by your observations.
- List length must match the number of tax brackets in the environment (usually 7, US Federal).

Output format: {"thought": "...", "tax_brackets": [<integer list>]}
```

> **📌 改動前原文**：
>
> ```
> You are the '{display_name}'. {role}
>
> TAX ADJUSTMENT TIME! Set the optimal tax brackets based on your observations.
>
> Tax bracket rules:
> - tax_brackets is an integer list; each element is the tax rate index (0-21) for that bracket.
> - Index 0 = 0% tax rate, index 21 = 100% tax rate (~5% per step).
> - Progressive taxation is recommended: lower brackets get lower rates.
> - List length must match the number of tax brackets in the environment (usually 7, US Federal).
>
> Output format: {"thought": "...", "tax_brackets": [<integer list>]}
> ```
>
> **備註**：Step B 執行過程中曾短暫在 `"TAX ADJUSTMENT TIME!"` 與 `"Tax bracket rules:"` 之間加入一行 `"Explain your reasoning clearly enough that a researcher can understand why you chose these brackets."`，但經檢視後判定為冗餘，已從 `_tax_system_prompt` 中一併刪除，未出現在現行版本。

**修正要點**：
- 刪除 `"Progressive taxation is recommended: lower brackets get lower rates."` 這條被標為 CRITICAL 的明文策略引導，這原本是 Planner 幾乎永遠輸出累進稅的根本原因
- 改為中立條列 `"Tax brackets may be flat, progressive, or regressive; choose the pattern justified by your observations."`，三種稅制皆為合法選項，由觀察結論決定
- **Step B++（本輪追加）**：將 Line 66 `"Set the optimal tax brackets based on your observations."` 改為 `"Choose tax brackets justified by your observations."`。動詞從 `"Set the optimal"` 改為 `"Choose ... justified by your observations"`，去掉 "optimal" 暗示存在唯一標準答案的語氣，並與 Line 70 `"choose the pattern justified by your observations"` 的句式一致，整體強調「由證據出發、由觀察結果導出決策」，而不是「尋找單一最佳值」。

**過度引導處**：
- ~~Line 70：<span style="color:red">"Progressive taxation is recommended: lower brackets get lower rates."</span>~~ ✅ **已修正（Step B）**
- ~~Line 66：<span style="color:red">"Set the optimal tax brackets"</span>~~ ✅ **已修正（Step B++）** — 改為 `"Choose tax brackets justified by your observations."`

---

### 2.3 Tax User Prompt Addendum（lines 180–184）

```
{state_desc}

Please set {n_brackets} tax bracket indices (each 0-21). Output a list of exactly {n_brackets} integers.
```

**評估**：格式說明，無過度引導。

---

## 3. `llm_agent/translator.py`

### 3.1 `translate_agent_obs` — Game Rules 區段（lines 205–215）

**改動後（現行版本）**：

```
[Game Rules]
  - Movement: Left(46), Right(47), Up(48), Down(49). Each move costs labor.
  - Gathering: Move onto a resource tile (dist=0) to collect automatically.
  - Building: Requires 1 Wood + 1 Stone. Earns Coin based on your build skill. Action 1.
  - Trading: Buy/sell resources on the market.
    Buy Stone (actions 2-12, bid price 0-10) | Sell Stone (actions 13-23, ask price 0-10)
    Buy Wood (actions 24-34, bid price 0-10) | Sell Wood (actions 35-45, ask price 0-10)
    * BUY locks your Coin in escrow. SELL locks 1 resource. Orders expire after 50 steps.
    * Trade executes when bid price >= ask price. Each order costs 0.25 labor.
  - Coin can change through building activity and market trades.
  - NOOP (action 0): Do nothing this step.
```

> **📌 改動前原文**：
>
> ```
> [Game Rules]
>   - Movement: Left(46), Right(47), Up(48), Down(49). Each move costs labor.
>   - Gathering: Move onto a resource tile (dist=0) to collect automatically.
>   - Building: Requires 1 Wood + 1 Stone. Earns Coin based on your build skill. Action 1.
>   - Trading: Buy/sell resources on the market.
>     Buy Stone (actions 2-12, bid price 0-10) | Sell Stone (actions 13-23, ask price 0-10)
>     Buy Wood (actions 24-34, bid price 0-10) | Sell Wood (actions 35-45, ask price 0-10)
>     * BUY locks your Coin in escrow. SELL locks 1 resource. Orders expire after 50 steps.
>     * Trade executes when bid price >= ask price. Each order costs 0.25 labor.
>     * To buy quickly: bid >= lowest ask. To sell quickly: ask <= highest bid.
>     * Bidding at price 0 almost never works.
>   - Ways to earn Coin: Build houses OR sell resources to other agents on the market.
>   - NOOP (action 0): Do nothing this step.
> ```

**修正要點**：
- 刪除 `"* To buy quickly: bid >= lowest ask. To sell quickly: ask <= highest bid."`（直接教模型快速成交策略）
- 刪除 `"* Bidding at price 0 almost never works."`（勸阻撿便宜策略，把模型推向積極出價）
- `"Ways to earn Coin: Build houses OR sell resources..."` → `"Coin can change through building activity and market trades."`（改為中性事實陳述，不再把 Build 與 Sell 並列為「賺錢手段」）

**過度引導處**：
- ~~Line 214：<span style="color:red">"To buy quickly: bid >= lowest ask. To sell quickly: ask <= highest bid."</span>~~ ✅ **已修正（Step B）**
- ~~Line 215：<span style="color:red">"Bidding at price 0 almost never works."</span>~~ ✅ **已修正（Step B）**
- ~~Line 216：<span style="color:red">"Ways to earn Coin: Build houses OR sell resources to other agents on the market."</span>~~ ✅ **已修正（Step B）**
- <span style="color:red">Build 規則 1 行 vs Trading 規則 ~6 行的資訊密度失衡</span> — ⏸ 部分緩解（刪除兩行 quick-fill 提示後，Trading 區塊從 7 行壓到 5 行，但仍略多於 Build 一行；屬結構性失衡，保留未進一步改動）

---

### 3.2 `translate_agent_obs` — Dynamic Pricing Quick Reference（舊 lines 260–274）

**改動後（現行版本）**：**整塊程式碼已從 `translator.py` 中刪除**。

現行的 `translate_agent_obs` 不再於每步動態生成 `"Quick reference: -> To buy Wood now, bid >= X..."` 區塊。`[Market Status]` 區段仍保留事實性的 `lowest ask / highest bid / avg price` 資料（L233-240），作為 agent 的決策依據，但不再直接教「現在要買就出 X」。

> **📌 改動前原文**（每步動態生成的交易提示）：
>
> ```python
> if w_ask is not None or w_bid is not None or s_ask is not None or s_bid is not None:
>     lines.append("  Quick reference:")
>     if w_ask is not None:
>         lines.append(f"    -> To buy Wood now, bid >= {w_ask} Coin (match the lowest ask)")
>     if w_bid is not None:
>         lines.append(f"    -> To sell Wood now, ask <= {w_bid} Coin (match the highest bid)")
>     if s_ask is not None:
>         lines.append(f"    -> To buy Stone now, bid >= {s_ask} Coin (match the lowest ask)")
>     if s_bid is not None:
>         lines.append(f"    -> To sell Stone now, ask <= {s_bid} Coin (match the highest bid)")
> ```

**修正要點**：
- 整塊程式碼從 `translate_agent_obs` 移除。原本是每步都會依當前市場狀態動態生成的交易提示，強化效應比靜態規則更強，等於每步都在提醒模型「你現在可以出 X 價立刻買/賣」
- `[Market Status]` 區段仍呈現 `lowest ask / highest bid / avg price` 事實數據，作為決策資料源，但不再給模型「可以直接套用的成交策略」

**過度引導處**：
- ~~整個 <span style="color:red">"Quick reference: To buy Wood now, bid >= X Coin..."</span> 動態區塊~~ ✅ **已修正（Step B，整塊程式碼已刪除）**

---

### 3.3 `translate_agent_obs` — Wellbeing Sense（lines 226–229）

```
[Wellbeing Sense]
  Current wealth: {coin:.1f} Coin
  Accumulated fatigue: {labor:.2f}
  (effective value accounting for your stamina modifier={lcm})
```

**評估**：事實描述，無過度引導。

---

### 3.4 Direction Language（lines 57–65）

```python
direction = "immediately Up — reachable via Move Up (action 48)"
direction = "immediately Down — reachable via Move Down (action 49)"
direction = "immediately Left — reachable via Move Left (action 46)"
direction = "immediately Right — reachable via Move Right (action 47)"
```

**評估**：方向＋對應 action ID，資訊性，無過度引導。

---

### 3.5 `translate_planner_obs`（lines 292–330）

```
=== Step {step}/1000 | Social Planner — Global Observation ===
{header}

[Wealth Distribution]
  Agent {idx}: Coin=X, Wood=Y, Stone=Z, Labor=L
  ...
  Mean Coin : {mean:.1f}
  Gini Index: {gini:.4f}  (0 = perfect equality, 1 = max inequality)

[Tax Information]
  Current marginal tax rates (brackets): 10.0%, 12.0%, ...
```

header 動態：
- 稅收日：`">>> TAX ADJUSTMENT DAY — output tax_brackets this step! <<<"`
- 非稅收日：`"(Regular observation step — output NOOP action)"`

**評估**：事實性數據，無過度引導。

---

## 4. `llm_agent/config.yaml`

### 4.1 Persona 0 — Youth Group（lines 44–56）

**改動後（現行版本）**：

```yaml
- id: 0
  name: "Youth Group"
  display_name: "Youth Group (age ≤20)"
  age_group: "youth"
  role: >
    You represent the youth demographic (age 20 and under). You are energetic
    with strong learning ability, though your skills are not yet mature.
    You move nimbly with low labor cost. You have limited experience and are
    still developing your capabilities.
  skill_pareto_alpha: 6.0
  labor_cost_modifier: 0.7
  endowment_coin_min: 0
  endowment_coin_max: 0
```

> **📌 改動前原文**：
>
> ```yaml
>   role: >
>     You represent the youth demographic (age 20 and under). You are energetic
>     with strong learning ability, though your skills are not yet mature.
>     You move nimbly with low labor cost. You are curious about market trading
>     but lack experience.
> ```

**修正要點**：
- `"curious about market trading but lack experience"` → `"limited experience and still developing your capabilities"`
- 把「對市場交易好奇」的策略傾向換成中性的成長期描述（經驗不足、尚在發展），讓模型不會預設年輕 agent 偏好交易

**過度引導處**：
- ~~<span style="color:red">"You are curious about market trading but lack experience."</span>~~ ✅ **已修正（Step B）**

---

### 4.2 Persona 1 — Young Adult Group（lines 58–69）

**改動後（現行版本）**：

```yaml
- id: 1
  name: "Young Adult Group"
  display_name: "Young Adult Group (age 21-40)"
  age_group: "young_adult"
  role: >
    You represent the young adult demographic (age 21-40). You have moderately
    high skills and abundant stamina, making you the most balanced and versatile
    group. You can sustain physical activity well and adapt to changing conditions.
  skill_pareto_alpha: 4.0
  labor_cost_modifier: 1.0
  endowment_coin_min: 0
  endowment_coin_max: 0
```

> **📌 改動前原文**：
>
> ```yaml
>   role: >
>     You represent the young adult demographic (age 21-40). You have moderately
>     high skills and abundant stamina, making you the most balanced and versatile
>     group. You are adept at flexibly switching between gathering, building,
>     and trading based on market conditions.
> ```

**修正要點**：
- `"adept at flexibly switching between gathering, building, and trading"` → `"sustain physical activity well and adapt to changing conditions"`
- 刪除「在 gather / build / trade 三類行動間靈活切換」的明示策略，改為「能持久進行體力活動並適應環境變化」的中性生理/行為描述

**過度引導處**：
- ~~<span style="color:red">"You are adept at flexibly switching between gathering, building, and trading based on market conditions."</span>~~ ✅ **已修正（Step B）**

---

### 4.3 Persona 2 — Middle-aged Group（lines 71–83） ★ HIGH

**改動後（現行版本）**：

```yaml
- id: 2
  name: "Middle-aged Group"
  display_name: "Middle-aged Group (age 41-60)"
  age_group: "middle_aged"
  role: >
    You represent the middle-aged demographic (age 41-60). You have mature
    skills and high building income, but your stamina is gradually declining.
    Physical effort is more costly for you than for younger agents, so stamina
    should be managed carefully.
  skill_pareto_alpha: 2.5
  labor_cost_modifier: 1.3
  endowment_coin_min: 20
  endowment_coin_max: 50
```

> **📌 改動前原文**：
>
> ```yaml
>   role: >
>     You represent the middle-aged demographic (age 41-60). You have mature
>     skills and high building income, but your stamina is gradually declining.
>     You tend to minimize unnecessary movement and make good use of market
>     trading to offset labor costs.
> ```

**修正要點**：
- 刪除 `"tend to minimize unnecessary movement and make good use of market trading to offset labor costs"` 這條被標記為 HIGH 的明文策略（「少移動、多交易以抵消勞動成本」），這正是實驗中中年 agent 傾向交易而非 Build 的根因
- 改為純粹的生理條件描述 `"Physical effort is more costly for you than for younger agents, so stamina should be managed carefully"`，只告訴模型「體力勞動代價較高、要留意體力」，不指定如何應對

**過度引導處**：
- ~~<span style="color:red">"You tend to minimize unnecessary movement and make good use of market trading to offset labor costs."</span>~~ ✅ **已修正（Step B）**

---

### 4.4 Persona 3 — Senior Group（lines 85–97） ★ HIGH

**改動後（現行版本）**：

```yaml
- id: 3
  name: "Senior Group"
  display_name: "Senior Group (age >60)"
  age_group: "senior"
  role: >
    You represent the senior demographic (age 60 and above). You have the most
    mature skills and highest building proficiency, but your stamina is the
    lowest with high movement costs. Physical activity is comparatively expensive
    for you, even though your building skill is high.
  skill_pareto_alpha: 1.5
  labor_cost_modifier: 1.8
  endowment_coin_min: 50
  endowment_coin_max: 100
```

> **📌 改動前原文**：
>
> ```yaml
>   role: >
>     You represent the senior demographic (age 60 and above). You have the most
>     mature skills and highest building proficiency, but your stamina is the
>     lowest with high movement costs. You prefer a stationary approach, relying
>     on market trading to acquire resources and building at favorable locations
>     for high income.
> ```

**修正要點**：
- 刪除「定點不動 + 依賴市場採購 + 高收益地點建造」這組被鎖定的策略三元組（"prefer a stationary approach, relying on market trading to acquire resources and building at favorable locations for high income"）
- 改為 `"Physical activity is comparatively expensive for you, even though your building skill is high"`，只陳述「身體活動昂貴、但建造技能高」的生理與技能條件，不指定策略解法

**過度引導處**：
- ~~<span style="color:red">"You prefer a stationary approach, relying on market trading to acquire resources and building at favorable locations for high income."</span>~~ ✅ **已修正（Step B）**

---

### 4.5 Planner Role（lines 99–123） ✅ **Step B+ 已修正**

> **註**：本小節在 Step B 原本保留未改（判斷 Planner 需要經濟機制事實作為決策根據）。後續 Step B+ 進一步檢討後，將 `"active markets signal a healthy economy"` 拆解為「事實層」與「價值判斷層」：事實層（交易會在 agent 之間重新分配 Coin 與資源）保留，價值判斷層（活躍＝健康）移除。`"building houses is the primary way to generate Coin"` 屬於系統事實（Build 是唯一淨產 Coin 來源），不構成對稅率方向的偏向，維持不改。

**改動後（現行版本）**：

```yaml
planner:
  name: "Social Planner"
  display_name: "Social Planner"
  role: >
    You are a Social Planner responsible for setting tax policy to promote
    a fair and productive economy. You observe the socioeconomic state at
    every step and set tax rates every 100 steps.

    Your two key objectives are FAIRNESS and PRODUCTIVITY.

    To assess fairness, observe:
    - Gini Index (0 = perfect equality, 1 = maximum inequality)
    - The gap in Coin holdings across agents (who is rich, who is poor)
    - Resource holdings (Wood, Stone) — uneven access may signal structural inequality
    - Whether the map's resource distribution creates unequal opportunities

    To assess productivity, observe:
    - Total Coin across all agents (overall economic output)
    - Mean Coin per agent (average standard of living)
    - Building activity (building houses is the primary way to generate Coin)
    - Market activity (whether agents are trading resources — market transactions redistribute Coin and resources between agents)

    Use these observable indicators to set tax brackets that balance
    equality and economic growth.
  tax_period: 100
```

> **📌 改動前原文**（Step B+ 修正處，僅 L119）：
>
>     - Market activity (whether agents are trading resources — active markets signal a healthy economy)

**修正要點**：
- `active markets signal a healthy economy` → `market transactions redistribute Coin and resources between agents`
- 刪除「活躍＝健康」的規範性判斷（real-world 中活躍交易也可能是恐慌拋售、投機、流動性危機），改為中性事實描述：市場是 agent 之間的 Coin / 資源再分配通道
- 保留 Planner 需要的經濟機制認知（「市場在做什麼」），但不再把「交易活躍」預先標記為正面指標，避免 Planner 將稅率決策偏向「鼓勵交易」
- `building houses is the primary way to generate Coin` **未動**：Build 是 AI Economist 系統中唯一的淨 Coin 產出來源（gather 不產 Coin、市場交易只是再分配），Planner 需要這個事實來推論稅基與稅率衝擊，且該陳述不對任何稅率方向（累進／比例／累退）構成偏向

**過度引導處**：
- ~~<span style="color:red">"active markets signal a healthy economy"</span>~~ ✅ **已修正（Step B+）**
- <span style="color:gray">"building houses is the primary way to generate Coin"</span> — 屬系統事實（Build 為唯一淨 Coin 產出），Planner 推理稅率所需，保留
- 結構上 OK：目標雙軸（公平＋生產力），未指定解法

---

## 5. `llm_agent/memory.py`

### 5.1 `build_consolidation_prompt`（lines 71–82）

```
You are {persona_name}. Below are your recent {N} decision thoughts:
[Step 1] {thought_1}
[Step 2] {thought_2}
...

Summarize your current strategic tendencies and resource situation in 1-2 sentences for long-term memory storage. Output only the summary text, no prefixes or suffixes.
```

**評估**：中性彙整指令，無策略引導。

---

### 5.2 PlannerMemory 格式化（lines 120–124, 137–140）

```
[Observation] {obs_summary}
[Prior Context] {prev_entry}
```
```
--- Memory {i+1}/{total} ---
{entry}
```

**評估**：純格式，無引導。

---

## 6. `llm_agent/llm_client.py`

### 6.1 Consolidation System Prompt（line 243）

```
You are a memory management assistant. Precisely summarize the provided past decision thoughts.
```

**評估**：中性工具角色。

---

### 6.2 Context Hand-off（lines 160–167）

```python
messages = [{"role": "system", "content": system_prompt}]
if context:
    messages.append({"role": "user", "content": context})
    messages.append({
        "role": "assistant",
        "content": "Understood. Please provide the current state."
    })
messages.append({"role": "user", "content": user_prompt})
```

**注意**：偽造的 assistant 訊息 <span style="color:red">`"Understood. Please provide the current state."`</span> 不算策略引導，但這種 message injection 對小模型可能造成角色混淆。低優先級。

---

### 6.3 JSON Schema Field Descriptions（lines 24–111）

**AGENT_JSON_SCHEMA**
```json
{
  "thought": {"type": "string", "description": "Reasoning based on current observations and resource status"},
  "action_id": {"type": "integer", "description": "Must be a valid Action ID defined by the environment (0-49)"}
}
```

**PLANNER_TAX_JSON_SCHEMA**
```json
{
  "thought": {"type": "string", "description": "Reasoning and social considerations for tax rate setting"},
  "tax_brackets": {"type": "array", "items": {"type": "integer"}, "description": "Tax rate index for each bracket (0-21), length must match the number of brackets"}
}
```

**PLANNER_OBSERVE_JSON_SCHEMA**
```json
{
  "thought": {"type": "string", "description": "The Social Planner's observation and reasoning"},
  "society_comment": {"type": "string", "description": "Assessment and insights on the current socioeconomic state"}
}
```

**CONSOLIDATION_SCHEMA**
```json
{
  "summary": {"type": "string", "description": "Concise summary of past decision thoughts (1-2 sentences)"}
}
```

**評估**：description 中性，無策略引導。

---

## 過度引導熱區總表（排除 action_map.py 的 88% 交易結構問題）

| # | 檔案 | 行號 | 內容摘要 | 症狀 | 嚴重度 | 狀態 |
|---|------|------|----------|------|--------|------|
| 1 | planner.py | 70 (原) | "Progressive taxation is recommended" | Planner 永遠輸出累進稅 | **CRITICAL** | ✅ Step B 已修正 |
| 2 | agent.py | 108 (原) | "labor...is an investment — it pays off in Coin" | 價值判斷，偏向 gather/build | HIGH | ✅ Step B 已修正 |
| 3 | agent.py | 106 (原) | "Build houses OR Sell on market to earn Coin" | 混淆 Build/Sell 經濟本質 | MEDIUM | ✅ Step B 已修正 |
| 4 | agent.py | 47 (原 Ex 4) | "bid 4 to match and buy immediately" | 示範即時成交 | HIGH | ✅ Step B 已修正 |
| 5 | agent.py | 50 (原 Ex 5) | "ask 5 to sell immediately" | 同上 | HIGH | ✅ Step B 已修正 |
| 6 | agent.py | 52 (原 Ex 6) | "only when truly idle" | NOOP 被污名化 | MEDIUM | ✅ Step B 已修正（NOOP 範例整個移除） |
| 7 | translator.py | 214 (原) | "To buy quickly / To sell quickly" | 教快速成交策略 | HIGH | ✅ Step B 已修正 |
| 8 | translator.py | 215 (原) | "Bidding at price 0 almost never works" | 勸阻撿便宜策略 | MEDIUM | ✅ Step B 已修正 |
| 9 | translator.py | 260–274 (原) | Dynamic "Quick reference" | 每步動態交易提示 | **HIGH** | ✅ Step B 已修正（整塊程式碼刪除） |
| 10 | translator.py | 205–217 (原) | Game Rules: Build 1 行 vs Trading 7 行 | 資訊密度失衡 | MEDIUM | ⏸ 部分緩解（Trading 壓到 5 行，結構性失衡保留） |
| 11 | config.yaml | 51–52 (原) | Youth: "curious about market trading" | persona 策略引導 | MEDIUM | ✅ Step B 已修正 |
| 12 | config.yaml | 65–66 (原) | Young Adult: "adept at flexibly switching..." | persona 策略引導 | MEDIUM | ✅ Step B 已修正 |
| 13 | config.yaml | 79–80 (原) | Middle-aged: "make good use of market trading" | persona 策略引導 | **HIGH** | ✅ Step B 已修正 |
| 14 | config.yaml | 93–95 (原) | Senior: "relying on market trading" | persona 策略引導 | **HIGH** | ✅ Step B 已修正 |
| 15 | config.yaml | 119 | Planner: "active markets signal a healthy economy" | 把交易活躍視為健康訊號 | MEDIUM | ✅ Step B+ 已修正 |

> **圖例**：
> - ✅ **Step B 已修正**：對應 prompt/程式碼已在 Step B 完成去引導化
> - ⏸ **保留未改**：Step B 刻意未動，現行程式碼仍保留原文
> - 「行號」欄標 `(原)` 表示行號以改動前版本為準；Step B 改動後有些行號已位移，詳情見對應小節

**額外追蹤項目（不在原 15 列熱區表內）**：

| 檔案 | 行號 (原) | 內容 | 狀態 |
|------|---------|------|------|
| planner.py | 66 | `"Set the optimal tax brackets"`（"optimal" 暗示標準答案） | ✅ Step B++ 已修正（改為 `"Choose tax brackets justified by your observations."`，句式與 Line 70 `choose ... justified by your observations` 對齊） |

---

## 修改優先順序建議

1. **P0（立即）**：planner.py 的 "Progressive taxation is recommended" — 唯一的明文策略引導
2. **P1（高）**：config.yaml 4 個 persona 去策略化 + Planner role 中性化
3. **P1（高）**：translator.py 移除 Quick reference 動態提示 + 刪除 "To buy/sell quickly" 三行
4. **P2（中）**：agent.py system prompt 的「投資」價值判斷中性化 + few-shot 範例平衡化

---

## Step B 完成記錄

Step B（prompt 去引導化）已完成下列 9 項 prompt / 程式碼改動。對應的詳細 before/after 與修正要點見各小節。

| 編號 | 小節 | 檔案 | 一行摘要 |
|---|---|---|---|
| **B1** | §1.1 | `llm_agent/agent.py` | `FEW_SHOT_EXAMPLES` 從 6 例含 NOOP 改為 7 例無 NOOP，Build 放 primacy + recency，移除即時成交措辭 |
| **B2** | §1.2 | `llm_agent/agent.py` | `_build_system_prompt` 幸福感區塊移除 `Build houses or Sell` 並列 + 移除 `labor is an investment` 的正向價值判斷，改為中性事實陳述 |
| **B3** | §2.2 | `llm_agent/planner.py` | `_tax_system_prompt` 刪除 CRITICAL 的 `Progressive taxation is recommended` 引導，改為 flat/progressive/regressive 三選列舉；本輪額外刪除冗餘的 `Explain your reasoning...` 行 |
| **B4** | §3.1 | `llm_agent/translator.py` | Game Rules 刪除 `To buy/sell quickly` 與 `Bidding at price 0 almost never works`，`Ways to earn Coin` 改為中性 `Coin can change through building activity and market trades` |
| **B5** | §3.2 | `llm_agent/translator.py` | Dynamic Pricing `Quick reference` 區塊整塊刪除，僅保留 `[Market Status]` 事實數據 |
| **B6** | §4.1 | `llm_agent/config.yaml` | Persona 0 Youth：`curious about market trading but lack experience` → `limited experience and still developing your capabilities` |
| **B7** | §4.2 | `llm_agent/config.yaml` | Persona 1 Young Adult：`adept at flexibly switching between gathering, building, and trading` → `sustain physical activity well and adapt to changing conditions` |
| **B8** | §4.3 | `llm_agent/config.yaml` | Persona 2 Middle-aged：刪除 `minimize unnecessary movement and make good use of market trading to offset labor costs`，改為純生理描述 |
| **B9** | §4.4 | `llm_agent/config.yaml` | Persona 3 Senior：刪除 `prefer a stationary approach, relying on market trading...`，改為 `Physical activity is comparatively expensive for you, even though your building skill is high` |
| **B+1** | §4.5 | `llm_agent/config.yaml` | Planner role：`active markets signal a healthy economy` → `market transactions redistribute Coin and resources between agents`，刪除「活躍＝健康」價值判斷，保留中性事實描述；`building houses is the primary way to generate Coin` 維持不動（系統事實） |
| **B++1** | §2.2 | `llm_agent/planner.py` | `_tax_system_prompt` Line 66：`"Set the optimal tax brackets based on your observations."` → `"Choose tax brackets justified by your observations."`，去除 "optimal" 暗示存在單一最佳解的語氣，句式改為與 Line 70 `"choose the pattern justified by your observations"` 一致的「由觀察導出決策」框架 |

**Step B 刻意未觸碰的項目**：

- `config.yaml` Planner role `building houses is the primary way to generate Coin`（§4.5）— 系統事實（Build 為唯一淨 Coin 產出來源），Planner 稅率推理所需，保留
- `agent.py` Decision rules 1-4、`translator.py` Wellbeing Sense / Direction Language / `translate_planner_obs`、`memory.py`、`llm_client.py` — 原本即中性，無需改動
