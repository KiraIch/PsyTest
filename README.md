# PsyTest

**繁體中文** | [English](README.en.md)

PsyTest 是一個本地端多模態心理諮商模擬數據生成系統，採用「快思慢想」架構，自動生成具備生態效度的心理諮商模擬數據。所有模型推理均可在本機完成，降低敏感資料外傳風險。本文件為專案正式說明文件，採定期更新制。

- 授權：Apache License 2.0，詳見 `LICENSE` 檔案。
- 維護者：KiraIch
- 文件版本：v1.1.2
- 最後更新：2026-10-16
- 專案狀態：全階段完成（Phase 1 至 Phase 7），進入維護期。詳細進度見 `HANDOVER.md`。

---

## 目錄

1. [專案簡介與架構](#1-專案簡介與架構)
2. [系統需求](#2-系統需求)
3. [安裝步驟](#3-安裝步驟)
4. [模型下載與放置](#4-模型下載與放置)
5. [Ollama 設定](#5-ollama-設定)
6. [快速開始](#6-快速開始)
7. [設定檔說明](#7-設定檔說明)
8. [輸出數據格式](#8-輸出數據格式)
9. [常見問題排解](#9-常見問題排解)
10. [倫理聲明與使用條款](#10-倫理聲明與使用條款)
11. [開發指南](#11-開發指南)
12. [測試](#12-測試)
13. [版本紀錄](#13-版本紀錄)
14. [授權與聯絡方式](#14-授權與聯絡方式)

---

## 1. 專案簡介與架構

### 1.1 簡介

本系統用於生成心理諮商教學與研究用的模擬數據，不提供臨床診斷功能。系統由五個層級組成：五感層負責多模態標註，快思層負責對話節奏與情緒觸發，慢想層負責扮演虛擬心理個案，聲音層負責情感語音合成，數據管線負責整合輸出為結構化 JSONL 數據庫。

### 1.2 架構圖

```text
                    ┌─────────────────────────────────┐
                    │        數據生成管線 (Pipeline)    │
                    │  session / batch / collector    │
                    │        output_formatter         │
                    └───────┬─────────┬───────┬───────┘
                            │         │       │
              ┌─────────────┘         │       └─────────────┐
              ▼                       ▼                     ▼
    ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
    │   五感層          │  │   快思層          │  │   慢想層          │
    │  Perception      │  │  Fast Think      │  │  Slow Think      │
    │                  │  │                  │  │                  │
    │ EmbeddingGemma 2 │  │ Laya 421M        │  │ Ollama Gemma 4   │
    │ audio / visual   │  │ rhythm / emotion │  │ persona/director │
    └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
             │                     │                     │
             └─────────────┬───────┴──────────┬──────────┘
                           ▼                  ▼
                 ┌──────────────────┐ ┌──────────────────┐
                 │   聲音層 (Voice)  │ │  督導報告        │
                 │ TTS / director   │ │  Supervisor      │
                 └──────────────────┘ └──────────────────┘
                           │
                           ▼
                 ┌──────────────────┐
                 │  硬體後端 (Backend) │
                 │ Metal/CUDA/ROCm/CPU │
                 └──────────────────┘
```

### 1.3 模組對應表

| 層級 | 目錄 | 關鍵模型 | 說明 |
|---|---|---|---|
| 五感層 | `src/perception/` | Ollama embeddinggemma-2（免授權首選） | 文字嵌入與音訊、視覺特徵分析；ST 與 GGUF 為備援 |
| 快思層 | `src/fast_think/` | 官方 laya 套件（multilingual） | 對話節奏控制、情緒路由、危機評估；GGUF 僅備援 |
| 慢想層 | `src/slow_think/` | Gemma 4 Uncensored（Ollama） | 虛擬個案扮演、人格管理、導演指令 |
| 聲音層 | `src/voice/` | Kokoro（sherpa-onnx）、ChatTTS、pyttsx3 | 情感語音合成，Orpheus 預設停用 |
| 數據管線 | `src/pipeline/` | 無（整合層） | 單場生成、批量排程、JSONL 輸出 |
| 後端 | `src/backend/` | 無 | Metal、CUDA、ROCm、CPU 自動偵測 |
| 督導報告 | `src/report/` | 無（確定性規則） | AI 督導分析與 Markdown 報告 |

---

## 2. 系統需求

### 2.1 最低需求

- 作業系統：macOS 13 以上（Apple Silicon）、Ubuntu 22.04 以上、Windows 10 以上（含 WSL2）。
- Python：3.10 以上（已驗證 3.13）。
- Git：2.40 以上。
- 磁碟空間：至少 15 GB 可用（Ollama 模型約 6.3GB，本機 GGUF 約 1.2GB，Kokoro 約 350MB，另需輸出與暫存空間）。
- 記憶體：最低 8 GB（純 CPU 模式），建議 16 GB 以上（同時載入多模型時）。

### 2.2 建議規格

| 用途 | 建議配備 |
|---|---|
| MacBook Air 開發 | Apple Silicon，16 GB 以上記憶體，參考 `configs/macbook_air.yaml` |
| NVIDIA 工作站 | CUDA 顯示卡，24 GB 以上記憶體，參考 `configs/cuda_workstation.yaml` |
| 無 GPU 環境 | 多核心 CPU，參考 `configs/cpu_only.yaml` |

### 2.3 外部服務

- Ollama：必須安裝並運行，用於慢想層 LLM 推理。詳見第 5 章。

---

## 3. 安裝步驟

### 3.1 通用步驟

建議使用專用 conda 環境隔離依賴：

```bash
git clone https://github.com/KiraIch/PsyTest.git
cd PsyTest
conda env create -f environment.yml
conda activate psytest
pip install -r requirements.txt
python -m src.backend.detector
python scripts/validate_setup.py
```

或使用快捷指令：

```bash
make setup
make check
```

### 3.2 各平台 llama-cpp-python 安裝

`llama-cpp-python` 必須依後端重新編譯，否則 GPU 加速不會啟用。

```bash
# macOS（Metal 自動啟用）
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# NVIDIA CUDA
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# AMD ROCm
CMAKE_ARGS="-DGGML_HIP=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# 純 CPU（使用 OpenBLAS 加速）
CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### 3.3 Windows

```bat
scripts\setup.bat
```

---

## 4. 模型下載與放置

模型權重檔不納入 Git 版本控制（已於 `.gitignore` 排除），需另行取得後放置於 `models/` 目錄。

### 4.1 模型檔案

將以下兩個權重檔放置於 `models/` 目錄（檔名須完全一致）：

| 檔案名稱 | 用途 | 大小（約） | 狀態 |
|---|---|---|---|
| `embeddinggemma-2-Q8_0.gguf` | 五感層嵌入模型 | 296 MB | llama.cpp 尚不支援，現以 ST 後端替代（需另行接受 HF 授權） |
| `Laya-BF16.gguf` | 快思層決策模型 | 805 MB | 僅備援，決策預設使用官方 laya 套件 |

### 4.2 放置與驗證

```bash
ls -lh models/
shasum -a 256 models/Laya-BF16.gguf
shasum -a 256 models/embeddinggemma-2-Q8_0.gguf
python -m src.backend.detector
```

詳細說明見 `models/README.md`。請勿將權重檔上傳至公開儲存庫或公開連結。

---

## 5. Ollama 設定

### 5.1 安裝與啟動

- macOS：安裝 Ollama 應用程式後開啟即可，位址為 `http://localhost:11434`。
- Linux：執行 `ollama serve` 啟動服務。
- Windows：啟動 Ollama 後確認 `http://localhost:11434` 可連線。
- 可經由環境變數 `OLLAMA_HOST` 覆寫位址。

### 5.2 拉取模型

```bash
ollama pull fredrezones55/Gemma-4-Uncensored-HauhauCS-Aggressive
ollama list
```

### 5.3 模型名稱偵測

系統會自動掃描已安裝模型，模糊匹配包含 `gemma`、`uncensored`、`hauhau`、`aggressive` 任一關鍵字的模型，支援 e2b 與 e4b 變體。優先順序為本專案鎖定標籤、e4b 明確標籤、e2b 標籤，其次為其他含關鍵字的模型。若找不到，將回退至其他可用的 4B 以上模型。

本專案已鎖定 e4b 變體（`fredrezones55/Gemma-4-Uncensored-HauhauCS-Aggressive:latest`，6.3GB），四份 `configs/*.yaml` 均已明確指定。也可在設定檔改回 `auto` 使用自動偵測，或以環境變數 `OLLAMA_MODEL` 覆寫。

---

## 6. 快速開始

### 6.1 環境檢查

```bash
python -m src.backend.detector
python scripts/validate_setup.py
```

### 6.2 生成第一場諮商

```bash
python scripts/generate_session.py \
  --persona data/personas/mdd_severe.yaml \
  --scenario data/scenarios/initial_session.yaml \
  --output data/output/ \
  --save-audio --verbose
```

### 6.3 批量生成

```bash
python scripts/batch_generate.py \
  --config configs/default.yaml \
  --personas mdd_severe,bpd_crisis,ptsd_combat \
  --sessions-per-persona 10 \
  --output data/output/ \
  --resume
```

### 6.4 互動測試

```bash
python scripts/interactive_session.py --persona data/personas/bpd_crisis.yaml
```

### 6.5 常用快捷指令

```bash
make check     # 硬體偵測與環境驗證
make test      # 執行測試
make generate  # 單場生成（預設 mdd_severe，可覆寫 PERSONA 與 SCENARIO）
make batch     # 批量生成
make report INPUT=<會談jsonl> OUTPUT=<報告md>  # 督導報告
make clean     # 清理輸出與暫存
```

更換個案（共三位：`mdd_severe` 林靜怡、`bpd_crisis` 陳曉彤、`ptsd_combat` 張志豪）：

```bash
make generate PERSONA=data/personas/bpd_crisis.yaml
python scripts/generate_session.py --persona data/personas/ptsd_combat.yaml --scenario data/scenarios/initial_session.yaml --output data/output/
```

---

## 7. 設定檔說明

設定檔位於 `configs/` 目錄，由 `src/config.py` 載入，支援環境變數覆寫（見 `.env.example`）。

| 檔案 | 用途 |
|---|---|
| `configs/default.yaml` | 預設設定，後端自動偵測 |
| `configs/macbook_air.yaml` | MacBook Air 優化（溫度上限 80 度，批次縮小） |
| `configs/cuda_workstation.yaml` | NVIDIA 工作站（最大吞吐） |
| `configs/cpu_only.yaml` | 純 CPU（功能完整，速度較慢） |

主要區段：`project`、`backend`（`force` 可為 null、metal、cuda、rocm、cpu）、`models`（embedding、laya、llm、tts）、`pipeline`、`thermal`、`logging`。

環境變數覆寫對照：`OLLAMA_HOST`、`OLLAMA_MODEL`、`PSYTEST_OUTPUT_DIR`、`PSYTEST_SAVE_AUDIO`、`PSYTEST_SAVE_EMBEDDINGS`、`PSYTEST_MAX_MEMORY_GB`、`PSYTEST_GPU_LAYERS`、`PSYTEST_TEMPERATURE_LIMIT`、`PSYTEST_LOG_LEVEL`。複製 `.env.example` 為 `.env` 後填寫（`.env` 禁止提交）。

個案設定位於 `data/personas/`（YAML，含診斷、背景、防衛機制、觸發點等），場景設定位於 `data/scenarios/`。

---

## 8. 輸出數據格式

主要格式為 JSONL：首行為會談 meta（含編號、個案、摘要），其後每輪對白各佔一行，可串流讀取與逐行檢查。每輪包含諮商師文本、個案文本與導演指令、音訊特徵、視覺特徵、心理標註、Laya 決策與三組 768 維嵌入向量。完整欄位定義見規格書 4.6 節與 `src/pipeline/output_formatter.py`。

輸出結構：

```text
data/output/
├── manifest.json
├── sessions/session_<12位id>.jsonl
├── audio/session_<12位id>_turn_<輪次>.wav
└── annotations/session_<12位id>_annotations.json
```

`manifest.json` 記錄數據集名稱、場次、輪數、音訊總時長、個案清單、模態、嵌入維度、硬體與使用模型版本。驗證指令：`python scripts/validate_output.py --input data/output/batch_001/ --check-embeddings --check-audio`。

---

## 9. 常見問題排解

### Q1：為什麼 `models/` 在 GitHub 上是空的？

權重檔體積過大且涉及散佈限制，已排除於版本控制。請依第 4 章另行取得。

### Q2：`git push` 出現檔案過大錯誤？

檢查是否誤加模型或資料檔：`git status --short`、`git check-ignore -v <路徑>`。誤加時以 `git rm --cached <路徑>` 移除，並檢查 `.gitignore`。

### Q3：llama-cpp-python 無法啟用 GPU？

必須依第 3.2 節以對應 `CMAKE_ARGS` 重新安裝，不可直接使用預編譯輪檔。

### Q4：Ollama 連線失敗？

確認服務是否運行：`curl http://localhost:11434/api/tags`。macOS 以應用程式啟動，Linux 執行 `ollama serve`，或設定 `OLLAMA_HOST`。

### Q5：找不到模型（model not found）？

執行 `ollama list` 確認已拉取 Gemma 4 變體。系統採模糊匹配，若名稱差異過大，請以 `OLLAMA_MODEL` 明確指定。

### Q6：MacBook Air 過熱？

改用 `configs/macbook_air.yaml`，系統會在 80 度以上降速。溫度持續超過 85 度應暫停生成。

### Q7：記憶體不足？

不要同時載入所有模型。系統採按需載入，批量生成時保持串行，必要時改用 `configs/cpu_only.yaml` 並縮小批次。

### Q8：語音引擎如何選擇與安裝？

選擇順序為 Kokoro、ChatTTS、pyttsx3，Orpheus 預設停用。pyttsx3 隨主依賴安裝即有基本語音。Kokoro 需另行安裝 `sherpa-onnx` 並下載模型（約 350MB，指令見 `models/README.md`）。ChatTTS 需另行安裝（授權 AGPLv3+，首次載入需由 HF 下載數 GB 模型）。各引擎缺失時會明確報錯，不會靜默失敗。

---

## 10. 倫理聲明與使用條款

1. 本系統產出之所有個案內容均為虛構教學材料，僅供臨床心理學教學與研究使用，不構成臨床診斷或治療建議。
2. 模擬結果不得直接用於實際個案處置，實際應用須由合格專業人員判斷並符合相關法規與倫理規範。
3. 危機個案模擬可表達自傷或自殺意念與情緒，可描述過去病史，但系統不得產出具體自傷方法的步驟、工具取得方式等可操作內容。
4. 受測者與真實個案資料屬於敏感性個人資料，禁止提交至版本控制。開發與測試僅可使用去識別化或合成資料。
5. 使用、修改或散佈本專案須遵守 Apache License 2.0，並保留授權與著作權聲明。模型權重另有其原始授權，使用前須確認其散佈與商用條款。

---

## 11. 開發指南

分支策略：`main` 為穩定分支，`develop` 為整合分支，功能分支命名為 `feature/<名稱>`，修正為 `fix/<名稱>`，文件為 `docs/<名稱>`。提交訊息採用 Conventional Commits（例如 `feat(pipeline): 實作單場生成器`）。合併至 `main` 或 `develop` 須經至少一人審查，涉及計分邏輯、個資處理或授權變更者須經負責人核准。全階段已完成，後續採維護期流程，詳見 `HANDOVER.md`。

---

## 12. 測試

```bash
pytest tests/ -v
```

測試對照：`test_backend.py` 驗證硬體偵測，`test_embedding.py`、`test_laya.py`、`test_ollama.py`、`test_tts.py`、`test_pipeline.py` 分別對應各層，`test_phase3.py` 至 `test_phase6.py` 覆蓋智能層、聲音層、管線與輔助功能。缺權重、缺服務或架構不支援的案例會明確跳過（目前僅剩 GGUF 架構一項）。提交前至少執行 `git diff --check` 與 `git status --short`。

---

## 13. 版本紀錄

| 日期 | 版本 | 說明 | 作者 |
|---|---|---|---|
| 2026-10-16 | v0.1.0 | 初版建立，確立文件結構與維護規範 | KiraIch |
| 2026-10-16 | v0.2.0 | 納入完整開發規格書，完成 Phase 1（基礎設施）與全模組骨架，對應快思慢想架構 | KiraIch |
| 2026-10-16 | v0.3.0 | 完成 Phase 2（嵌入引擎、決策引擎、Ollama 客戶端），鎖定 e4b 變體 | KiraIch |
| 2026-10-16 | v0.4.0 | 完成 Phase 3（情緒路由、節奏控制、人格管理、導演系統），版本對齊 | KiraIch |
| 2026-10-16 | v0.5.0 | 完成 Phase 4（Kokoro、ChatTTS、pyttsx3 三引擎，Orpheus 改 guarded），以 Kokoro 取代雲端方案 | KiraIch |
| 2026-10-16 | v0.6.0 | 完成 Phase 5（單場生成、輸出格式化、批量排程、四支 CLI），端到端驗證通過 | KiraIch |
| 2026-10-16 | v0.7.0 | 完成 Phase 6（溫度監控、記憶體管理、確定性督導報告） | KiraIch |
| 2026-10-16 | v1.0.0 | 完成 Phase 7：決策改用官方 laya 套件、嵌入加入 ST 替代路徑、文件定稿，進入維護期 | KiraIch |
| 2026-10-16 | v1.0.1 | 修復 pyttsx3 驅動重用空音訊、加入沉默守衛規則 | KiraIch |
| 2026-10-16 | v1.0.2 | JSONL 改為首行 meta 加逐輪分行，強化決策延遲測試 | KiraIch |
| 2026-10-16 | v1.0.3 | 修復裸 pytest 缺路徑、同步 pyproject 依賴、Makefile 支援指定個案 | KiraIch |
| 2026-10-16 | v1.1.0 | 個案擴充至 9 位（含 2 共病），新增對話風格、抗拒度、記憶錨點欄位 | KiraIch |
| 2026-10-16 | v1.1.1 | 嵌入預設改 ST 優先自動回退，管線自動接線嵌入引擎 | KiraIch |
| 2026-10-16 | v1.1.2 | 嵌入改走 Ollama 官方 embeddinggemma-2（免授權，768 維已驗證） | KiraIch |

---

## 14. 授權與聯絡方式

本專案採用 Apache License 2.0 授權，授權全文見 `LICENSE`，第三方聲明見 `NOTICE`。維護者為 KiraIch，儲存庫位址為 https://github.com/KiraIch/PsyTest.git。議題請經由 GitHub Issues 提交，並附重現步驟、環境資訊與已去敏的相關日誌。

---

*本文件採定期更新制，下次預定檢視日期依專案迭代排程，原則不超過三個月。更新時請同步調整文件版本、最後更新日期與第 13 章版本紀錄。*
