# Linux 部署與大量模擬完整使用手冊

這份手冊是給不熟 Linux 的使用者準備的。目標是把這個研究專案完整搬到 Linux GPU 環境，並安全地執行大量模擬，同時保留續跑、重跑、監控與後續分析能力。

本手冊對應的 Linux 工具位於：

```text
deployment/linux/
```

這些工具的設計原則是：

- 不修改原始研究程式碼。
- 仍然透過既有的 `run_simulation.py` 執行模擬。
- Linux wrapper 只負責批次調度、續跑、重跑、完整性檢查與打包。

不會修改的核心研究檔案：

- `run_simulation.py`
- `llm_agent/`
- `ai_economist/`
- `llm_agent/config.yaml`

## 1. Linux 工具檔案說明

```text
deployment/linux/
├── README_linux_run.md
├── LINUX_BATCH_PACKAGING_PLAN.md
├── .env.example
├── run_parallel.py
├── run_smoke_test.sh
├── pack_project.ps1
└── pack_project.sh
```

各檔案用途：

- `README_linux_run.md`：你現在正在看的完整使用手冊。
- `LINUX_BATCH_PACKAGING_PLAN.md`：設計與驗證原則、review findings、完成條件總整理。
- `.env.example`：環境變數範例，不含真實金鑰。
- `run_parallel.py`：Linux 大量模擬批次執行器。
- `run_smoke_test.sh`：Linux 環境的快速檢查腳本。
- `pack_project.ps1`：Windows 端打包腳本。
- `pack_project.sh`：Linux 或 macOS 端打包腳本。

## 2. 核心觀念先看懂

### 2.1 Linux 結果會放在哪裡

Linux 執行結果不應混進你過去 Windows 的：

```text
simulation_results/
```

Linux 端應該使用獨立的新資料夾，例如：

```text
/data/your_user/linux_simulation_results
```

如果你沒有另外指定 `--output-dir`，`run_parallel.py` 預設會輸出到：

```text
linux_simulation_results
```

也就是專案根目錄下的：

```text
project/linux_simulation_results/
```

正式大量實驗仍建議手動指定絕對路徑，避免把結果塞在 home 或專案資料夾裡。

### 2.2 什麼叫續跑

續跑的意思是：

- 保留已經完整跑完的 episode
- 不重跑它們
- 只補跑尚未完成的 episode

這個判斷不是只看 `summary.json`，而是看：

- `COMPLETED.json`
- `summary.json`
- `step_metrics.csv`
- `action_log.csv`
- `tax_log.csv`
- `token_usage`

是否全部一致。

### 2.3 什麼叫重跑

重跑的意思是：

- 你明確要求某些已存在的 episode 再跑一次
- 舊結果不直接覆蓋
- 舊結果會先搬到 archive 形式的資料夾
- 然後再建立一個新的同名 run

### 2.4 archive 是什麼

archive 是為了保留舊資料，不是為了續跑。

例如原本有：

```text
episode_0001_agent_gemma4_e2b_planner_gpt_5_4_mini/
```

如果你用 `--force-rerun` 重跑，它會先被改名成：

```text
episode_0001_agent_gemma4_e2b_planner_gpt_5_4_mini.force_rerun_20260424_120000/
```

然後再建立新的：

```text
episode_0001_agent_gemma4_e2b_planner_gpt_5_4_mini/
```

現在的 `run_parallel.py` 會排除 archive 資料夾，不會把它們重新算進 `completed_experiments.txt`。

## 3. 先在 Windows 打包

### 3.1 最簡單打包方式

在 Windows PowerShell 進入專案根目錄後執行：

```powershell
powershell -ExecutionPolicy Bypass -File deployment\linux\pack_project.ps1
```

它會產生：

```text
llm_econ_linux_package_YYYYMMDD_HHMMSS.tar.gz
```

### 3.2 指定打包輸出位置

例如你想輸出到 `D:\packages`：

```powershell
powershell -ExecutionPolicy Bypass -File deployment\linux\pack_project.ps1 -OutputDir D:\packages
```

現在腳本如果發現資料夾不存在，會自動建立，不需要你先手動建。

### 3.3 打包內容包含什麼

保留：

- `run_simulation.py`
- `run_experiment.py`
- `visualize_experiments.py`
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

## 4. 將打包檔傳到 Linux

假設打包檔在本機，檔名為：

```text
llm_econ_linux_package_20260424_120000.tar.gz
```

上傳方式：

```bash
scp llm_econ_linux_package_20260424_120000.tar.gz user@server:/data/user/
```

登入 Linux：

```bash
ssh user@server
```

解壓：

```bash
cd /data/user
tar -xzf llm_econ_linux_package_20260424_120000.tar.gz
cd llm_econ_project
```

## 5. 在 Linux 建立 Python 環境

第一次進到新環境時執行：

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -e ./ai_economist
```

之後每次重新登入 Linux，先做：

```bash
cd /data/user/llm_econ_project
source .venv/bin/activate
```

## 6. 設定 OpenAI API key

不要把真實 API key 寫進 repo 或打包檔。

### 6.1 臨時輸入

```bash
read -s OPENAI_API_KEY
export OPENAI_API_KEY
```

### 6.2 使用 `.env`

建立 `.env`：

```bash
cp deployment/linux/.env.example .env
nano .env
chmod 600 .env
```

載入 `.env`：

```bash
set -a
source .env
set +a
```

## 7. 設定 Ollama

先確認 Ollama 是否可用：

```bash
ollama --version
ollama list
curl http://localhost:11434/api/tags
```

如果模型不存在：

```bash
ollama pull gemma4:e2b
```

如果 `curl` 失敗，先啟動 Ollama：

```bash
ollama serve
```

正式大量實驗建議用 service、`tmux` 或 `screen` 讓 Ollama 常駐。

## 8. 先跑 smoke test

### 8.1 一鍵 smoke test

```bash
bash deployment/linux/run_smoke_test.sh
```

這個腳本會依序做：

- dry-run 5 steps
- LLM smoke test 5 steps

### 8.2 手動執行 smoke test

```bash
python run_simulation.py --dry-run --steps 5 --run-name linux_dry_test
```

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

如果這兩個都成功，再進入 pilot 或正式批次。

## 9. 用 `run_parallel.py` 執行 batch

`run_parallel.py` 不直接 import 核心模擬函式，而是用 subprocess 呼叫：

```bash
python run_simulation.py ...
```

這樣做的理由：

- 不動原始研究程式碼
- 避免 `load_config()` 全域快取污染不同 episode
- 每個 episode 都是獨立 process，比較穩定

### 9.1 先跑 1-2 集 pilot

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

### 9.2 正式跑 500 集、4 個並行

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

## 10. Linux 結果資料夾結構

每個完成的 episode 都會保留原本 Windows 版本就有的研究輸出：

```text
/data/user/linux_simulation_results/
└── episode_0001_agent_gemma4_e2b_planner_gpt_5_4_mini/
    ├── step_metrics.csv
    ├── action_log.csv
    ├── tax_log.csv
    ├── summary.json
    ├── agent_thoughts.xlsx
    └── maps/
```

Linux wrapper 額外加入：

```text
/data/user/linux_simulation_results/
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

補充：

- `RUNNING.json` 只會在執行中存在，完成或失敗後會移除。
- `COMPLETED.json` 才是 batch runner 視為完成的正式標記。

## 11. `COMPLETED.json` 為什麼重要

不能只看 `summary.json`，因為 `run_simulation.py` 每 100 steps checkpoint 就會先寫出一次 `summary.json`。

如果只看 `summary.json` 是否存在，中途中斷的 run 也可能被誤判成完成。

因此 `run_parallel.py` 的完成條件是：

- `summary.json` 存在
- `summary.total_steps == expected_steps`
- `summary.final_metrics.step == expected_steps - 1`
- `step_metrics.csv` row 數正確
- `action_log.csv` row 數正確
- `tax_log.csv` row 數正確
- `summary.token_usage.agents` 存在
- `summary.token_usage.planner` 存在
- `COMPLETED.json.status == "completed"`

以 `steps=1000`、`tax_period=100` 為例，預期應為：

```text
step_metrics.csv rows = 1000
action_log.csv rows = 1000
tax_log.csv rows = 10
summary.final_metrics.step = 999
```

## 12. 續跑怎麼做

如果實驗中途中斷，例如：

- SSH 斷線
- 你手動按 `Ctrl+C`
- Linux 重開
- 某個子程序失敗

要續跑時，直接重新執行同一條 command，不要加 `--force-rerun`。

例如：

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

它會：

- 跳過已完成且驗證通過的 episode
- 不會因為只有 `summary.json` 就誤判完成
- 只補跑尚未完成的 episode

## 13. 重跑怎麼做

如果你不是要續跑，而是想重跑已存在的 episode，才使用：

```bash
--force-rerun
```

例如：

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
  --output-dir /data/user/linux_simulation_results \
  --force-rerun
```

它會：

- 先把同名舊 run archive 起來
- 再建立新的同名 run

## 14. 如何監控正在執行的 batch

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

看 batch log：

```bash
tail -f /data/user/linux_simulation_results/logs/batch.log
```

看單一 episode log：

```bash
tail -f /data/user/linux_simulation_results/logs/episode_0001_agent_gemma4_e2b_planner_gpt_5_4_mini.log
```

看 GPU：

```bash
nvidia-smi
```

看 Ollama：

```bash
ollama ps
```

看完成數：

```bash
find /data/user/linux_simulation_results -name COMPLETED.json | wc -l
```

看完成清單：

```bash
cat /data/user/linux_simulation_results/completed_experiments.txt
```

## 15. 如何中斷

正常停止 batch：

```text
Ctrl+C
```

從外部停止 wrapper：

```bash
pkill -f run_parallel.py
```

如果只有子程序卡住：

```bash
pkill -f run_simulation.py
```

下次重新執行同一條 command 時，系統會依 `COMPLETED.json` 做續跑。

## 16. 如何確認輸出可做後續分析

Linux 的輸出欄位與 Windows 相容，所以你仍然可以用原本的：

```bash
python visualize_experiments.py \
  --base /data/user/linux_simulation_results \
  --all
```

或只分析完成清單中的實驗：

```bash
python visualize_experiments.py \
  --experiments \
  /data/user/linux_simulation_results/episode_0001_agent_gemma4_e2b_planner_gpt_5_4_mini \
  /data/user/linux_simulation_results/episode_0002_agent_gemma4_e2b_planner_gpt_5_4_mini
```

建議正式分析時，以 `completed_experiments.txt` 中列出的路徑為主，避免誤讀中斷資料或 archive 資料夾。

圖表輸出位置通常會是：

```text
/data/user/linux_simulation_results/comparison_charts/
/data/user/linux_simulation_results/episode_0001_agent_gemma4_e2b_planner_gpt_5_4_mini/charts/
```

## 17. 分析時的口徑提醒

目前專案中：

`total_coin` 使用 `total_endowment("Coin")`，代表 agent 的總 Coin 持有量，包含 inventory 與市場中尚未成交、暫時存放於 escrow 的 Coin；`coin_agent_*` 則使用 agent inventory，代表 agent 當下可立即支配的 Coin，不包含已掛單後暫時保留在 escrow 的 Coin。因此兩者不是完全相同的統計口徑，後續分析時不要直接把 `coin_agent_*` 加總後視為 `total_coin` 的完全對應值。

## 18. 建議的實際操作順序

第一次部署時，照這個順序最穩：

1. 在 Windows 執行 `pack_project.ps1` 打包。
2. 用 `scp` 傳到 Linux。
3. 在 Linux 解壓。
4. 建立 `.venv`。
5. 安裝 `requirements.txt` 與 `ai_economist`。
6. 設定 `OPENAI_API_KEY`。
7. 確認 `ollama serve` 與 `ollama pull gemma4:e2b`。
8. 跑 `bash deployment/linux/run_smoke_test.sh`。
9. 跑 `--episodes 2 --parallel 1` 的 pilot。
10. 確認 `COMPLETED.json`、`completed_experiments.txt`、`summary.json` 都正常。
11. 再開始 500 集正式實驗。

## 19. 建議的正式實驗前檢查

正式跑大批次前，至少先確認：

- `OPENAI_API_KEY` 已正確載入
- `ollama list` 看得到 `gemma4:e2b`
- `curl http://localhost:11434/api/tags` 成功
- dry-run 5 steps 成功
- LLM smoke test 5 steps 成功
- 1-2 集 pilot 可完整跑完
- `completed_experiments.txt` 只列出有效完成的 run
- `linux_simulation_results` 沒和舊 `simulation_results` 混用

做到這裡，就已經是可正式部署並進入 Linux 大量模擬的狀態。
