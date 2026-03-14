# 使用輕量級的 Python 3.10 作為基底
FROM python:3.10-slim

# 設定容器內的工作目錄
WORKDIR /app

# 安裝系統層級的編譯依賴 (這對底層 C++ 環境很重要)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# [步驟 A] 先單獨複製 requirements.txt 並安裝外部套件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# [步驟 B] 將宿主機當前目錄下的所有東西 (包含 ai_economist 與 llm_agent) 複製到容器內
COPY . .

# [步驟 C] 獨立安裝剛複製進來的本地環境 (注意這裡是底線 ai_economist)
RUN pip install -e ./ai_economist

# 預設指令 (以 dry-run 測試環境是否建置成功)
CMD ["python", "-m", "llm_agent.simulation", "--dry-run", "--steps", "5"]