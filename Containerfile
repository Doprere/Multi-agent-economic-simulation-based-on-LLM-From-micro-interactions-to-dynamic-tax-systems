# 使用輕量級的 Python 3.10 作為基底
FROM python:3.10-slim

# 設定容器內的工作目錄
WORKDIR /app

# 安裝系統層級的編譯依賴 (ai-economist 可能需要 C++ 編譯)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 先將宿主機「當前目錄」下的所有東西 (包含 ai-economist 與 llm_agent) 複製到容器內的 /app
COPY . .

# 安裝 Python 依賴 (這會自動讀取 requirements.txt 並安裝本地的 ai-economist)
RUN pip install --no-cache-dir -r requirements.txt

# 預設指令 (以 dry-run 測試環境是否建置成功)
CMD ["python", "-m", "llm_agent.simulation", "--dry-run", "--steps", "5"]