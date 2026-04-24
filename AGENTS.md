# AGENTS.md — LLM-Agent 社會經濟模擬研究專案

## 角色定位

你是這份碩士論文研究的 **研究助理**。你熟悉 AI Agent 架構、LLM prompting、多智能體系統設計，以及 AI Economist 框架。你的任務是協助開發、除錯、分析實驗結果，並提出改進建議。

## 研究概述

本研究延伸 Zheng et al. (2020) 的 AI Economist 框架，**以 LLM 取代傳統 RL policy** 驅動經濟模擬中的多智能體決策。核心問題：LLM 驅動的異質 agent 能否在資源採集、交易、建造的經濟環境中展現合理的決策行為與社會動態？

### 實驗設計

- **4 個 MobileAgent**：以年齡層（青年/青壯年/中年/老年）為 persona，各有不同技能分布、勞動成本與初始 Coin 稟賦
- **1 個 Social Planner**：每 100 步設定累進稅率，以公平性（Gini、財富差距）與生產力（總 Coin、建造/交易活動）為雙重目標
- **50 種離散動作**：NOOP(0)、Build(1)、買賣 Stone(2-23)/Wood(24-45) 透過 CDA、移動(46-49)
  - Foundation 按字母序排列資源：**Stone 在 Wood 之前**（啟動時由 `_validate_action_order()` 驗證）
- **25×25 地圖**：Wood 和 Stone 資源分布，agent 需移動採集後建造賺取 Coin
- **初始稟賦**：中年(20-50 Coin)、老年(50-100 Coin) 擁有初始儲蓄，模擬生命週期財富累積

## 系統架構

### Agent 決策管線（每步）

```
env.obs → Translator(英文) → Prompt 組裝 → LLM 呼叫 → JSON 解析 → 動作驗證 → env.step()
```

### 核心模組

| 檔案 | 職責 |
|------|------|
| `llm_agent/translator.py` | obs dict → 英文自然語言（資源方向+action ID、市場定價提示、訂單上限+20步自動到期警告、避免 0 價警告、blocked directions、遊戲規則） |
| `llm_agent/agent.py` | MobileAgentLLM：幸福感框架 + stamina 映射 + few-shot(6例) + LLM 呼叫 + 動作驗證 + 記憶 |
| `llm_agent/action_map.py` | 動作 ID ↔ 語義名稱映射（Stone before Wood，與 Foundation 字母序一致） |
| `llm_agent/planner.py` | PlannerLLM：觀察步 / 稅收步分離，稅率驗證（0-21、長度匹配） |
| `llm_agent/memory.py` | 雙層記憶：短期滑動視窗(8步) + 長期非同步彙整(每10步) |
| `llm_agent/llm_client.py` | OpenAI async client（JSON schema 強制） |
| `llm_agent/ollama_client.py` | Ollama HTTP client（duck typing，regex JSON 提取） |
| `llm_agent/config.py` | YAML 載入 + dataclass 結構化 |
| `llm_agent/logger.py` | 完整記錄：CSV/JSON/Excel(6個工作表) + 地圖快照 |
| `run_simulation.py` | **統一入口**：支援 Planner / Agents 各自指定 backend（openai \| ollama）與 model；CLI override 優先於 config |
| `llm_agent/client_factory.py` | 依 `LLMConfig.backend` 建立對應 client（openai → LLMClient、ollama → OllamaClient） |
| `llm_agent/sim_common.py` | 兩支 simulation 共用函式（年齡技能、動作序驗證、勞動修正器、planner action 構造、隨機動作、鄰近偵測） |
| `ollama_simulation.py` | **DEPRECATED** → 翻譯舊 flag 後委派 `run_simulation.py` |
| `llm_agent/simulation.py` | **DEPRECATED** → 直接委派 `run_simulation.py` |

### 記憶系統

- **Agent 短期記憶**：最近 8 步 thought 的 deque
- **Agent 長期記憶**：每累積 10 步觸發 LLM 非同步彙整（不阻塞決策）
- **Planner 記憶**：複合觀察滑動視窗(3筆)，每筆 = 當前觀察 + 前次摘要

### LLM 後端（Duck Typing）

`LLMClient`（OpenAI）與 `OllamaClient` 共享相同介面：
- `call_agent(sys_prompt, user_prompt, context)` → `{thought, action_id}`
- `call_planner_observe(...)` → `{thought, society_comment}`
- `call_planner_tax(...)` → `{thought, tax_brackets}`
- `call_consolidation(prompt)` → `str`

## 設計原則與約束

1. **語言分離**：LLM 面向文本一律英文；使用者面向輸出（print/Excel 標題）可用繁體中文
2. **資訊豐富、不做推薦**：translator 提供完整環境描述，但不建議 agent 該做什麼動作
3. **Temperature 0.7**：保持決策多樣性
4. **Action Mask 強制驗證**：action_id 必須在合法集合中，否則重試（最多 3 次，附帶錯誤提示），最終 fallback 隨機合法動作
5. **非阻塞記憶彙整**：長期記憶以 asyncio.Task 背景執行，失敗不影響主流程
6. **Excel 防公式**：所有文字欄位以 `_safe_cell_value()` 防止 `=` 開頭被 Excel 誤判為公式
7. **Delta-based 勞動縮放**：`_apply_labor_modifier()` 僅對本步新增 Labor 乘以 modifier，避免對歷史累積值重複縮放
8. **啟動時動作序驗證**：`_validate_action_order()` 確認 Foundation 內部的 action_names 與 action_map.py 一致

## Prompt 設計（小模型最佳化）

以下設計針對 3B-8B 參數模型（qwen2.5:3b、llama3:8b）的行為特徵調校：

### 幸福感框架（agent.py `_build_system_prompt`）
- **核心句**：`Your happiness = Coin earned minus effort spent.`
- **勞動定位**：明確說明 gather/build 的勞動是「投資」而非「消耗」，防止小模型將 NOOP 視為最佳策略
- **Stamina 自然語言**：將數值 `labor_cost_modifier` 轉為描述（"good stamina" / "costs more effort"），不暴露原始數字

### Few-shot 範例（agent.py `FEW_SHOT_EXAMPLES`）
- 6 個範例：gather(2) + build(1) + buy(1) + sell(1) + NOOP(1)
- **Primacy effect**：gather 範例放最前面，NOOP 放最後且條件嚴格（"only when truly idle"）
- 範例中的 action_id 對應真實動作映射

### Translator 資訊設計（translator.py）
- **方向語言**：使用 Up/Down/Left/Right + action ID（如 "immediately Up — reachable via Move Up (action 48)"），與動作名稱對齊
- **交易規則壓縮**：從 14 行壓縮至 7 行，保留核心定價邏輯
- **Wellbeing Sense 區塊**：顯示當前 Coin 與累積疲勞，提供體感參考
- **動態定價提示**：根據當前市場狀態自動生成 "To buy Wood now, bid >= X Coin"
- **訂單上限警告 + 自動到期提示**：達到 max_orders 時顯示 WARNING，並說明「未成交訂單 20 步後自動過期，空位會自然釋出」避免 agent 卡死於 NOOP
- **避免 0 價警告**：說明 bid 0 = 要求免費拿到資源、ask 0 = 免費送人，正常經濟幾乎不會發生；建議以 `market_rate` 為定價參考
- **Blocked directions**：明確列出被牆/水/其他 agent 阻擋的方向

## 診斷工具

- **LLM Prompts 工作表**（Sheet 5）：完整記錄每步送給 LLM 的 system/user/context prompt
- **Resource Adjacency 工作表**（Sheet 6）：agent 在資源旁（Manhattan dist=1）時的事件記錄
- **地圖快照**（`maps/*.png`）：每 20 步 + 資源鄰近事件自動截圖
  - 全域地圖加紅色虛線邊界標示地圖範圍
  - 各 agent 自我中心視圖（ego-centric）加 OOB 暗紅色半透明覆蓋，標示地圖外區域

## 執行方式

### 統一入口 `run_simulation.py`（推薦）

```bash
# 1) 預設 config 驅動（OpenAI，缺省 llm_planner → Planner 與 Agents 共用 client）
$env:OPENAI_API_KEY = "sk-proj-..."     # Windows PowerShell
python run_simulation.py --steps 200 --run-name baseline

# 2) 全本地 Ollama（Agents + Planner 共用同一 client）
python run_simulation.py --steps 200 \
  --agent-backend ollama --agent-model llama3:8b \
  --ollama-url http://localhost:11434 \
  --run-name all_ollama

# 3) 不對稱混搭：Planner 用雲端強模型、4 個 Agent 用本地弱模型
python run_simulation.py --steps 200 \
  --agent-backend   ollama --agent-model   gemma2:2b \
  --planner-backend openai --planner-model gpt-4o-mini \
  --ollama-url http://localhost:11434 \
  --run-name mix_gemma_gpt4o

# 4) Dry-run 環境煙測（隨機合法動作，不呼叫 LLM）
python run_simulation.py --dry-run --steps 5
```

**優先序**：`CLI > config.yaml > dataclass 預設值`。
**共用邏輯**：若未指定 `--planner-backend` / `--planner-model` 且 config 無 `llm_planner:` → Planner 與 Agents 共用同一 client（向後相容）。
**混搭風險**：Planner 用 Ollama 時 `tax_brackets` regex 解析無 fallback，建議 Planner 優先選 OpenAI；Agents 用 Ollama 有 `action_mask` 隨機 fallback 較安全。

### Config 擴充

```yaml
llm:                         # Agents 預設（缺省時 Planner 共用此組）
  backend: "openai"          # "openai" | "ollama"
  model: "gpt-4o-mini"
  ...

llm_planner:                 # 選填；存在則 Planner 獨立走這組（可跨 backend）
  backend: "openai"
  model: "gpt-4o-mini"
  ...
```

**Fallback 規則**：`cfg.llm_planner is None` → Planner 與 Agents 共用同一 client 實例（與現況相同，向後相容）。

### 舊入口（已棄用但仍可執行）

```bash
# 印 deprecation 訊息後，翻譯 flag 委派至 run_simulation.py
python ollama_simulation.py --steps 200 --model llama3:8b --ollama-url http://localhost:11434
python -m llm_agent.simulation --steps 200
```

## 輸出結構

```
simulation_results/run_YYYYMMDD_HHMMSS/
├── step_metrics.csv      # 每步指標（Coin/Gini/Reward/庫存）
├── action_log.csv        # 每步動作 ID
├── tax_log.csv           # 稅率變更
├── agent_thoughts.xlsx   # 6 個工作表（思維/記憶/Prompt/鄰近事件...）
├── summary.json          # 運行摘要
└── maps/*.png            # 地圖快照
```

## 當前研究階段

**Phase 1 已完成**：LLM 文本全英文化、遊戲規則注入、資源方向描述、重試修正、診斷工具。

**Phase 2 進行中**：Prompt 平衡化與小模型行為調校。
- 已完成：幸福感重寫（勞動=投資）、交易規則壓縮、Few-shot 重新平衡、方向語言對齊 action ID
- 已完成：CDA 動作排序修正（Stone before Wood）、勞動修正器實作（delta-based）
- 已完成：初始 Coin 稟賦系統（中年/老年生命週期儲蓄）
- 已完成：CDA `order_duration: 20` — 未成交訂單 20 步後自動過期，避免 5 單上限鎖死 agent
- 已完成：translator「避免 0 價」警告 — 說明 bid/ask 0 等同免費交換，建議參考 `market_rate`
- 已驗證模型：llama3:8b（8B）、qwen2.5:3b（3B，GTX 1650 4GB VRAM 可用）
- 觀察中：qwen2.5:3b 首次成功採集資源（Wood=7, Stone=2 / 20步），但 adjacency 正確率約 67%、尚未達成 Build
- 待探索：adjacency 提示強化、更長步數模擬（100+步）驗證完整經濟循環（gather→build→earn）

**Phase 3 已完成**：Planner / Agents 異質模型架構（不對稱實驗）。
- **動機**：Planner 每 100 步決策 1 次（1000 步 = 10 次稅率）；Agents 每步 × 4 人 × 1000 步 = 4000 次決策。用強模型（GPT-4o-mini）處理少量高價值決策、弱模型（gemma2:2b / llama3:8b）跑大量低價值決策，同時可觀察「決策層級 vs 執行層級，誰對社會動態影響較大」。
- **實作**：新增 `llm_planner:` 選填 config block + `client_factory.make_llm_client()` 工廠 + 抽共用 `sim_common.py` + 統一入口 `run_simulation.py`；`agent.py` / `planner.py` / `memory.py` / `translator.py` / `llm_client.py` / `ollama_client.py` **完全不動**（已 duck-typing 相容）。舊入口 `simulation.py` / `ollama_simulation.py` 轉為 deprecation stub，翻譯 flag 後委派至新入口。
- **風險**：Planner 用 Ollama 時 `tax_brackets` regex 解析失敗無 fallback（action_id 有 mask 隨機 fallback 較安全）；建議 Planner 優先用 OpenAI。
- **token_usage 格式變更**：summary.json 的 `token_usage` 由 flat 改為 nested `{"agents": {...}, "planner": {...}}`；共用 client 時 `planner` 欄位為 `{"shared_with_agents": true}`。下游 `recover_planner_xlsx.py` / 視覺化腳本若讀此欄位需同步調整。
- **待驗證**：混搭 5 步煙測、對照組 NOOP 率、CLI override 優先序、舊入口 deprecation stub 回落路徑（驗證清單見 [witty-hugging-pearl.md](../../.Codex/plans/witty-hugging-pearl.md) §驗證）。
