#!/usr/bin/env bash
# PsyTest Linux/macOS 安裝腳本
# 用法：bash scripts/setup.sh [--cpu-only]
set -euo pipefail

CPU_ONLY=0
if [[ "${1:-}" == "--cpu-only" ]]; then
  CPU_ONLY=1
fi

echo "[PsyTest] 安裝 Python 依賴..."
pip install -r requirements.txt

echo "[PsyTest] 檢查 llama-cpp-python 後端..."
python -c "import llama_cpp; print('llama-cpp-python OK')" || {
  echo "[PsyTest] 警告：llama-cpp-python 未正確安裝，請依 README 5.2 節以 CMAKE_ARGS 重新安裝。"
}

if [[ "$CPU_ONLY" == "1" ]]; then
  echo "[PsyTest] CPU 模式：略過 GPU 後端檢查。"
else
  echo "[PsyTest] 偵測硬體後端..."
  python -m src.backend.detector || true
fi

echo "[PsyTest] 驗證環境..."
python scripts/validate_setup.py || true

echo "[PsyTest] 完成。請確認 models/ 下已放置權重檔。"
