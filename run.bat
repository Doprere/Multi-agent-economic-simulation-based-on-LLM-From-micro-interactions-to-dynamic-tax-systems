@echo off
:: run.bat — Execute simulation with proper UTF-8 console encoding
chcp 65001 > nul
set PYTHONPATH=%~dp0ai-economist
python -m llm_agent.simulation %*
