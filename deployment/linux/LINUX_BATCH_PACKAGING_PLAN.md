# Linux 打包與大量模擬執行計畫

## 目標

本計畫目標是在不修改原始研究程式碼的前提下，新增 Linux 部署、打包、平行執行、續跑、重跑與結果驗證工具，讓專案能在 Linux GPU 環境中穩定執行大量模擬。

保持不動的核心研究程式：

- `run_simulation.py`
- `llm_agent/`
- `ai_economist/`
- `llm_agent/config.yaml`

Linux 端新增工具集中放在：

```text
deployment/linux/
```

核心原則：

- 模擬邏輯、prompt、AI Economist 環境與 logger 維持一致。
- Linux 工具只負責部署、批次調度、完整性檢查、續跑、重跑、監控與打包。
- Linux 結果存放於獨立的 `linux_simulation_results/`，不與既有 Windows `simulation_results/` 混用。

## 新增檔案

```text
deployment/linux/
├── LINUX_BATCH_PACKAGING_PLAN.md
├── README_linux_run.md
├── .env.example
├── run_parallel.py
├── run_smoke_test.sh
├── pack_project.ps1
└── pack_project.sh
```

## 打包範圍

保留：

- `run_simulation.py`
- `run_experiment.py`
- `random_tax_simulation.py`
- `run_random_calibration.py`
- `run_saez_experiment.py`
- `saez_simulation.py`
- `preview_saez_schedule.py`
- `validate_calibration_csv.py`
- `requirements.txt`
- `README.md`
- `AGENTS.md`
- `llm_agent/`
- `ai_economist/`
- `deployment/linux/`

排除：

- `.git/`
- `venv/`
- `.venv/`
- `__pycache__/`
- `simulation_results/`
- `linux_simulation_results/`
- `thesis/`
- `*.pdf`
- `*.docx`
- `*.doc`
- `.env`
- `*.env`
- `credentials.json`

預期輸出檔名：

```text
llm_econ_linux_package_YYYYMMDD_HHMMSS.tar.gz
```

## Linux 安裝流程

上傳：

```bash
scp llm_econ_linux_package_YYYYMMDD_HHMMSS.tar.gz user@server:/data/user/
```

解壓：

```bash
ssh user@server
cd /data/user
tar -xzf llm_econ_linux_package_YYYYMMDD_HHMMSS.tar.gz
cd llm_econ_project
```

建立 Python 環境：

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -e ./ai_economist
```

確認 Ollama：

```bash
ollama --version
ollama list
curl http://localhost:11434/api/tags
```

若模型不存在：

```bash
ollama pull gemma4:e2b
```

## OpenAI 金鑰設定

不將 OpenAI API key 寫入程式碼或打包檔。

臨時輸入：

```bash
read -s OPENAI_API_KEY
export OPENAI_API_KEY
```

或使用 `.env`：

```bash
cp deployment/linux/.env.example .env
nano .env
chmod 600 .env
set -a
source .env
set +a
```

## Smoke Test

dry-run：

```bash
python run_simulation.py --dry-run --steps 5 --run-name linux_dry_test
```

LLM smoke test：

```bash
python run_simulation.py \
  --steps 5 \
  --agent-backend ollama \
  --agent-model gemma4:e2b \
  --planner-backend openai \
  --planner-model gpt-5.4-mini \
  --ollama-url http://localhost:11434 \
  --run-name linux_llm_test
```

或直接：

```bash
bash deployment/linux/run_smoke_test.sh
```

## 批次執行設計

`run_parallel.py` 以 subprocess 呼叫既有的 `run_simulation.py`，不直接 import `run_episode()`。

這樣可以避免：

- 修改核心研究程式
- `load_config()` 全域 `_CONFIG` 快取污染
- 多個 episode 共用同一 Python process 狀態

Pilot：

```bash
python deployment/linux/run_parallel.py \
  --episodes 2 \
  --parallel 1 \
  --steps 1000 \
  --agent-backend ollama \
  --agent-model gemma4:e2b \
  --planner-backend openai \
  --planner-model gpt-5.4-mini \
  --ollama-url http://localhost:11434 \
  --output-dir /data/user/linux_simulation_results
```

正式：

```bash
python deployment/linux/run_parallel.py \
  --episodes 500 \
  --parallel 4 \
  --steps 1000 \
  --agent-backend ollama \
  --agent-model gemma4:e2b \
  --planner-backend openai \
  --planner-model gpt-5.4-mini \
  --ollama-url http://localhost:11434 \
  --output-dir /data/user/linux_simulation_results
```

`run_parallel.py` 目前預設輸出目錄為：

```text
linux_simulation_results
```

## Linux 輸出結構

維持與 Windows 相容的研究輸出：

```text
linux_simulation_results/
└── episode_0001_agent_gemma4_e2b_planner_gpt_5_4_mini/
    ├── step_metrics.csv
    ├── action_log.csv
    ├── tax_log.csv
    ├── summary.json
    ├── agent_thoughts.xlsx
    └── maps/
```

Linux wrapper 額外新增：

```text
linux_simulation_results/
├── logs/
│   ├── batch.log
│   ├── episode_0001_agent_gemma4_e2b_planner_gpt_5_4_mini.log
│   └── ...
├── completed_experiments.txt
└── episode_0001_agent_gemma4_e2b_planner_gpt_5_4_mini/
    ├── COMPLETED.json
    ├── FAILED.json
    ├── INTERRUPTED.json
    └── RUNNING.json
```

## 完成判斷規則

不能只看 `summary.json` 是否存在，因為 `run_simulation.py` 每 100 steps checkpoint 就會寫出 `summary.json`。

單一 episode 被視為完成時，必須同時滿足：

- `summary.json` 存在
- `summary.total_steps == expected_steps`
- `summary.final_metrics.step == expected_steps - 1`
- `step_metrics.csv` rows 等於 `expected_steps`
- `action_log.csv` rows 等於 `expected_steps`
- `tax_log.csv` rows 等於預期稅收次數
- `summary.token_usage.agents` 存在
- `summary.token_usage.planner` 存在
- `COMPLETED.json` 存在
- `COMPLETED.json.status == "completed"`

以 `steps=1000`、`tax_period=100` 為例：

```text
step_metrics.csv rows = 1000
action_log.csv rows = 1000
tax_log.csv rows = 10
summary.final_metrics.step = 999
```

## Marker 檔案設計

執行中：

```text
RUNNING.json
```

內容包含：

- `status`
- `run_name`
- `episode_index`
- `started_at`
- `steps`
- `agent_model`
- `planner_model`
- `command`
- `pid`
- `log_path`

成功完成：

```text
COMPLETED.json
```

內容包含：

- `status`
- `run_name`
- `episode_index`
- `started_at`
- `ended_at`
- `duration_seconds`
- `summary_path`
- `total_steps`
- `tax_count`
- `has_token_usage`
- `step_metrics_rows`
- `action_log_rows`
- `tax_log_rows`

失敗：

```text
FAILED.json
```

中斷：

```text
INTERRUPTED.json
```

補充：

- `RUNNING.json` 在 episode 完成、失敗或中斷後會移除。
- archive 資料夾不會被重新算進 `completed_experiments.txt`。

## 續跑與重跑

### 續跑

不加 `--force-rerun`，直接重跑同一條 command。

系統會：

- 跳過已完成 episode
- 只補跑未完成 episode
- 不會因為只有 `summary.json` 就誤判完成

### 重跑

加上：

```bash
--force-rerun
```

系統會：

- 先 archive 同名舊 run
- 再重新跑新的同名 run

archive 例子：

```text
episode_0001_agent_gemma4_e2b_planner_gpt_5_4_mini.force_rerun_20260424_120000/
```

## Review Findings 對應處理

### Finding 1：`summary.json` 不能單獨當作續跑完成依據

處理：

- wrapper 不用 `summary.json exists` 當完成條件
- 必須通過完整性檢查並存在 `COMPLETED.json`

### Finding 2：完整跑完後仍可能缺 `token_usage`

處理：

- `COMPLETED.json` 只在 token usage 檢查通過後才寫入
- `completed_experiments.txt` 只列出 token usage 完整的 run

### Finding 3：`total_coin` 與 `coin_agent_*` 定義不一致

處理：

- 不修改 `logger.py`
- 在 README 與 plan 中標明兩者口徑不同

### Finding 4：相同 `run_name` 會覆蓋既有結果

處理：

- wrapper 使用唯一 run name
- 既有 run 若需重跑，先 archive 再重跑
- archive 資料夾不納入完成清單

### Finding 5：`load_config()` 全域快取會影響同一 process 多組設定

處理：

- wrapper 一律使用 subprocess 執行 `run_simulation.py`
- 不在同一 process 內直接呼叫 `run_episode()`

## 監控方式

建議使用 `tmux`：

```bash
tmux new -s econ_sim
```

離開但不中斷：

```text
Ctrl+B
D
```

重新進入：

```bash
tmux attach -t econ_sim
```

查看：

- `tail -f /data/user/linux_simulation_results/logs/batch.log`
- `tail -f /data/user/linux_simulation_results/logs/episode_0001_agent_gemma4_e2b_planner_gpt_5_4_mini.log`
- `nvidia-smi`
- `ollama ps`
- `find /data/user/linux_simulation_results -name COMPLETED.json | wc -l`
- `cat /data/user/linux_simulation_results/completed_experiments.txt`

## 中斷方式

正常中斷：

```text
Ctrl+C
```

從外部停止 wrapper：

```bash
pkill -f run_parallel.py
```

停止子程序：

```bash
pkill -f run_simulation.py
```

## 後續畫圖與統計分析

Linux 輸出與 Windows 相容，可直接用：

```bash
echo "Analysis and visualization scripts are local thesis utilities and are not included in this Linux experiment package."
```

建議正式分析時以 `completed_experiments.txt` 列出的 episode 為主。

## 測試計畫

### 打包測試

確認壓縮包不包含：

- `.git/`
- `venv/`
- `.venv/`
- `simulation_results/`
- `linux_simulation_results/`
- `thesis/`
- PDF/docx
- 真實 `.env`
- API key

### Linux 安裝測試

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ./ai_economist
```

### Dry-run 測試

```bash
python run_simulation.py --dry-run --steps 5 --run-name linux_dry_test
```

### LLM smoke test

```bash
python run_simulation.py \
  --steps 5 \
  --agent-backend ollama \
  --agent-model gemma4:e2b \
  --planner-backend openai \
  --planner-model gpt-5.4-mini \
  --ollama-url http://localhost:11434 \
  --run-name linux_llm_test
```

### Pilot batch 測試

```bash
python deployment/linux/run_parallel.py \
  --episodes 2 \
  --parallel 1 \
  --steps 1000 \
  --agent-model gemma4:e2b \
  --planner-model gpt-5.4-mini \
  --output-dir /data/user/linux_simulation_results
```

確認：

- 產生 2 個 episode 資料夾
- 每個資料夾有 `COMPLETED.json`
- `completed_experiments.txt` 只列出有效完成的 run
- CSV row 數與 summary 一致

### 中斷續跑測試

流程：

1. 啟動 1000-step episode
2. 中途 `Ctrl+C`
3. 確認沒有 `COMPLETED.json`
4. 重新執行同一 command
5. 確認未完成 episode 被補跑
6. 確認已完成 episode 被跳過

## Assumptions

- 原始研究程式碼不修改
- Linux wrapper 只新增在 `deployment/linux/`
- wrapper 使用 subprocess 呼叫 `run_simulation.py`
- planner 維持目前每步 observe 設計
- agent model 使用 `gemma4:e2b`
- planner model 使用 `gpt-5.4-mini`
- 真實 OpenAI API key 不寫入 repo 或打包檔
- Linux 輸出與 Windows 保持相容，可直接支援後續畫圖與統計分析
