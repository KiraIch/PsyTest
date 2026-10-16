# PsyTest — 專案交接文件（HANDOVER）

> 本文件為專案交接與維運之正式依據，採定期更新制。任何交接、人員異動、架構變更或發佈前，均須更新本文件並經負責人確認。

- 文件版本：v0.6.0
- 最後更新：2026-10-16
- 現任維護者：KiraIch
- 前任維護者：（無，專案初建）
- 接手人：（待填）
- 專案狀態：Phase 5 完成，Phase 6 待啟動
- 下次預定檢視日期：2027-01-16（原則不超過三個月）

---

## 目錄

1. [交接摘要](#1-交接摘要)
2. [專案基本資訊](#2-專案基本資訊)
3. [目前進度](#3-目前進度)
4. [系統架構與關鍵檔案](#4-系統架構與關鍵檔案)
5. [環境建置步驟](#5-環境建置步驟)
6. [環境與忽略規則](#6-環境與忽略規則)
7. [帳號、權限與機敏資訊](#7-帳號權限與機敏資訊)
8. [日常維運](#8-日常維運)
9. [已知風險與限制](#9-已知風險與限制)
10. [待辦事項](#10-待辦事項)
11. [驗收標準](#11-驗收標準)
12. [聯絡資訊](#12-聯絡資訊)
13. [更新紀錄](#13-更新紀錄)
14. [交接簽收](#14-交接簽收)

---

## 1. 交接摘要

### 1.1 本次交接目的

- v0.2.0：納入完整開發規格書（本地端多模態心理諮商模擬數據生成系統，快思慢想架構），完成 Phase 1 基礎設施與全模組骨架，確立後續六個階段的實作基線。
- 適用情境：新成員加入、維護責任轉移、長期暫停後重啟。

### 1.2 關鍵結論

1. 規格書已定稿並記憶：五感層、快思層、慢想層、聲音層、數據管線，支援 Metal、CUDA、ROCm、CPU 四種後端，實作順序為 Phase 1 至 Phase 7。慢想層鎖定 e4b 變體。
2. Phase 1 至 Phase 5 已完成：基礎設施、核心引擎、智能層、聲音層、數據管線均已實作並通過驗證。
3. Phase 6 至 Phase 7 為骨架狀態（介面與 docstring 完整，方法體拋出 `NotImplementedError`），接手人首要任務為 Phase 6 輔助功能（見第 10 章）。
4. 模型權重檔（約 1.1 GB）與 Ollama 模型不納入版本控制，須另行交接。規格檔名與本機實際檔名存在別名對應，已由 `src/config.py` 自動處理。

### 1.3 需接手人優先閱讀的檔案

1. `README.md` v0.2.0 — 專案說明、安裝、Ollama 設定、倫理聲明。
2. `src/config.py` — 設定管理與模型別名邏輯。
3. `src/backend/detector.py` — 後端偵測邏輯。
4. `models/README.md` — 模型檔名對照與驗證。
5. 本文件第 3、9、10 章。

---

## 2. 專案基本資訊

| 項目 | 內容 |
|---|---|
| 專案名稱 | PsyTest |
| 專案定位 | 本地端多模態心理諮商模擬數據生成系統（快思慢想架構） |
| 儲存庫位址 | https://github.com/KiraIch/PsyTest.git |
| 授權 | Apache License 2.0 |
| 主要語言 | Python 3.10 以上（已驗證 3.13） |
| 預設分支 | `main` |
| 整合分支 | `develop`（待建立） |
| 議題管理 | GitHub Issues（待啟用標籤規範） |
| 文件語言 | 繁體中文為主，程式碼註解與提交訊息中英並行 |
| 文件規範 | 不使用表情符號、顏文字、繪文字，保持正式書面語 |
| Subagent 規範 | 一律使用 Muse Spark 1.3 Contributor，高或極高 effort（見 `opencode.json` 的 `subagents` 區段） |

---

## 3. 目前進度

### 3.1 已完成（v0.2.0）

- [x] Phase 1：專案結構建立（`src/`、`data/`、`tests/`、`configs/`、`scripts/`、`logs/`）。
- [x] Phase 1：`src/config.py`（YAML 載入、環境變數覆寫、模型別名、驗證）。
- [x] Phase 1：`src/backend/detector.py`（Metal→CUDA→ROCm→CPU 偵測、`--force-backend`、CLI）。
- [x] Phase 1：`src/utils/logger.py`（終端加輪替檔案日誌）。
- [x] Phase 1：`configs/` 四份設定（default、macbook_air、cuda_workstation、cpu_only）。
- [x] 全模組骨架：backend（base 加四後端）、perception、fast_think、slow_think（含 6 份提示模板）、voice、pipeline、report、utils（thermal、memory）。
- [x] 資料範例：3 份個案 YAML、2 份場景 YAML。
- [x] 測試 6 份：backend 全量可過，其餘缺權重或服務時自動跳過。
- [x] 根目錄配套：`requirements.txt`、`pyproject.toml`、`Makefile`、`.env.example`、`scripts/` 7 支（含 `validate_setup.py`）。
- [x] 文件更新：`README.md` v0.2.0（架構圖、跨平台安裝、Ollama、倫理聲明）、`models/README.md`（檔名對照）、`NOTICE`（第三方表）、`.gitignore` v0.2.0（輸出與檢查點規則）。
- [x] Phase 2：`perception/embedding_engine.py`（GGUF 載入、文字與批次嵌入、餘弦相似度、按需載入卸載）。
- [x] Phase 2：`fast_think/laya_engine.py`（四種決策、安全預設、危機保守合併、BF16 指引）。
- [x] Phase 2：`slow_think/ollama_client.py`（連線偵測、e4b 優先匹配、串流與非串流、3 次重試）。
- [x] 慢想層鎖定 e4b 變體，四份 `configs/*.yaml` 明確指定標籤，`detect_model` 優先 e4b。
- [x] Phase 3：`fast_think/emotion_router.py`（八情緒映射、中英關鍵字分類、<10ms 目標）。
- [x] Phase 3：`fast_think/rhythm_controller.py`（五狀態狀態機、Laya 節奏計畫、長沉默環境音）。
- [x] Phase 3：`slow_think/persona_manager.py`（YAML 載入驗證、系統提示組裝、風格參數）。
- [x] Phase 3：`slow_think/director.py`（指令解析與正規化、預設值兜底、TTS 標籤轉換）。
- [x] 版本對齊 0.4.0（套件、設定、文件一致）。
- [x] Phase 4：`voice/tts_engine.py`（自動選擇 kokoro→chattts→pyttsx3、雙標籤解析）。
- [x] Phase 4：`voice/kokoro_backend.py`（sherpa-onnx 中英 103 音色，中文驗證 5 秒）。
- [x] Phase 4：`voice/chatts_backend.py`（情感標籤、固定音色，中文驗證 3.75 秒）。
- [x] Phase 4：`voice/pyttsx3_backend.py`（系統語音保底，macOS 停頓支援）。
- [x] Phase 4：`voice/orpheus_backend.py` 改為 guarded（雲端專用，預設停用）。
- [x] Phase 4：`voice/voice_director.py`（四引擎參數轉換）。
- [x] 版本對齊 0.5.0（套件、設定、文件一致）。
- [x] Phase 5：`perception/audio_analyzer.py`（純 numpy 特徵，管線依賴回填）。
- [x] Phase 5：`perception/visual_analyzer.py`（無頭參數映射，管線依賴回填）。
- [x] Phase 5：`pipeline/data_collector.py`（輪次正規化與範圍檢查）。
- [x] Phase 5：`pipeline/output_formatter.py`（JSONL、WAV、標註、manifest、嵌入量化）。
- [x] Phase 5：`pipeline/session_generator.py`（模板諮商師、決策降級、危機中斷、錯誤恢復）。
- [x] Phase 5：`pipeline/batch_generator.py`（斷點續傳、串行排程）。
- [x] Phase 5：四支 CLI 完整實作（單場、批量、驗證、互動）與 `validate_setup` TTS 檢查。
- [x] 版本對齊 0.6.0（套件、設定、文件一致）。

### 3.2 進行中

- [ ] Phase 6：`utils/thermal_monitor.py`（溫度監控）。
- [ ] Phase 6：`utils/memory_manager.py`（模型生命週期）。
- [ ] Phase 6：`report/supervisor.py`（督導分析）。

### 3.3 尚未開始

- [ ] Phase 7：全量測試啟用、README 定稿、Makefile 定稿。
- [ ] 隱私影響評估與模型授權盤點。
- [ ] 隱私影響評估與模型授權盤點。

---

## 4. 系統架構與關鍵檔案

### 4.1 現行結構

```text
PsyTest/
├── README.md / HANDOVER.md / LICENSE / NOTICE / .env.example
├── .gitignore / opencode.json / requirements.txt / pyproject.toml / Makefile
├── models/                    ← 權重檔（不提交，見 models/README.md）
├── src/
│   ├── config.py              ← Phase 1 完成
│   ├── backend/               ← detector 完成，base 加四後端為骨架
│   ├── perception/            ← 骨架（embedding 計算相似度已實作）
│   ├── fast_think/            ← 骨架（含決策資料結構）
│   ├── slow_think/            ← 骨架（含 prompts 六份模板）
│   ├── voice/                 ← 骨架（含情緒映射表）
│   ├── pipeline/              ← 骨架
│   ├── report/                ← 骨架
│   └── utils/                 ← logger 完成，thermal/memory 為骨架
├── data/personas/ scenarios/  ← 範例 5 份
├── data/output/               ← 僅結構，不提交內容
├── tests/                     ← 6 份
├── configs/                   ← 4 份
└── scripts/                   ← 7 支
```

### 4.2 關鍵檔案說明

| 檔案 | 用途 | 修改注意事項 |
|---|---|---|
| `src/config.py` | 設定載入、環境覆寫、模型別名 | 新增設定鍵須同步更新四份 YAML 與 `.env.example` |
| `src/backend/detector.py` | 後端偵測唯一入口 | 修改偵測順序須同步更新測試與 README |
| `src/backend/base.py` | 後端統一介面 | 新增抽象方法須同步實作四個後端 |
| `configs/*.yaml` | 環境設定 | 機敏值禁止寫入，改用環境變數 |
| `opencode.json` | 模型路由與 subagent 規範 | `gemini-super` 為 medium；subagent 限定 muse-main 高或極高 effort |
| `.gitignore` | 排除規則 | 以 `git check-ignore -v` 驗證，修改後更新本文件第 6 章 |
| `models/README.md` | 檔名別名對照 | 新增模型時同步更新 `MODEL_ALIASES` |
| `LICENSE` / `NOTICE` | 授權全文與第三方聲明 | LICENSE 禁止修改；NOTICE 隨相依即時更新 |

### 4.3 架構決策紀錄（ADR 摘要）

| 日期 | 決策 | 原因 | 影響 |
|---|---|---|---|
| 2026-10-16 | 模型權重不進 Git，以 `.gitignore` 排除 | 單檔超過 GitHub 100 MB 限制，總計逾 1 GB | 須建立獨立分發管道，見第 5.3 節 |
| 2026-10-16 | 採用 Apache 2.0 授權 | 允許商業利用與修改，具專利授權條款 | 須保留 `LICENSE` 與 `NOTICE` |
| 2026-10-16 | 文件採定期更新制，不超過三個月檢視一次 | 避免文件與實作脫節 | 維護者須排程檢視，見第 8 章 |
| 2026-10-16 | 規格檔名與本機檔名以別名相容 | 本機已有 `Laya-BF16.gguf`，規格書用 `laya-421m-bf16.gguf` | `MODEL_ALIASES` 統一處理，禁止散佈兩套路徑寫法 |
| 2026-10-16 | 以 Kokoro（sherpa-onnx，中英 103 音色，Apache-2.0）取代 Orpheus 雲端方案 | 實測 PyPI orpheus-tts 為雲端 WebSocket 客戶端，與本地端原則衝突；Kokoro 經調研與實測皆符合需求 | Orpheus 欄位保留為 guarded，選擇順序改為 kokoro→chattts→pyttsx3 |
| 2026-10-16 | 慢想層鎖定 e4b 變體並明確寫入四份設定 | 本機已有 6.3GB 的 `:latest`（即 e4b 內容），明確指定可保證跨機器一致，`auto` 模式亦優先 e4b | 更換變體須同步修改設定、`PINNED_MODEL` 與文件 |
| 2026-10-16 | Phase 5 回填音訊與視覺分析器（先前階段遺漏） | 管線標註依賴兩者，無實作則數據缺聲學與視覺欄位 | 音訊採純 numpy 實作，視覺採無頭參數映射 |
| 2026-10-16 | Phase 2 至 Phase 7 先立骨架再填實作 | 保證結構一次到位，後續迭代不重排目錄 | 骨架方法一律拋出 `NotImplementedError` 並註明階段 |
| 2026-10-16 | 測試缺權重或服務時自動跳過 | CI 與新成員本機不一定有模型 | 不可將跳過改為失敗，完整案例隨階段啟用 |

後續架構決策請依此表格式追加，禁止覆寫既有紀錄。

---

## 5. 環境建置步驟

### 5.1 前置需求

- Git 2.40 以上，Python 3.10 以上，可用磁碟 5 GB 以上。
- 已取得遠端儲存庫讀取權限。
- llama-cpp-python 須依平台以 `CMAKE_ARGS` 編譯（見 README 3.2 節）。

### 5.2 標準建置流程

```bash
# 1. 複製專案
git clone https://github.com/KiraIch/PsyTest.git
cd PsyTest

# 2. 安裝依賴
pip install -r requirements.txt

# 3. 偵測後端與驗證環境
python -m src.backend.detector
python scripts/validate_setup.py

# 4. 確認忽略規則生效
git check-ignore -v models/Laya-BF16.gguf
git check-ignore -v models/embeddinggemma-2-Q8_0.gguf

# 5.（選用）建立 .env
cp .env.example .env
```

### 5.3 模型檔案還原

GGUF 權重由移交人經安全管道提供，Ollama 模型自行拉取：

```bash
ollama pull fredrezones55/Gemma-4-Uncensored-HauhauCS-Aggressive
```

還原後執行 `ls -lh models/` 與 `shasum -a 256` 比對雜湊（正式值見 `models/README.md`）。

### 5.4 驗證清單

- [ ] `git status` 顯示工作區乾淨。
- [ ] `python -m src.backend.detector` 輸出合法後端。
- [ ] `python scripts/validate_setup.py` 回傳 0。
- [ ] `pytest tests/ -v` 通過或僅有預期跳過。
- [ ] `models/` 下兩個權重檔存在（約 805 MB、296 MB）。

---

## 6. 環境與忽略規則

- 正式規則以 `.gitignore` 為準，本章僅摘要交接重點。
- `.gitignore` 版本：v0.2.0，最後更新 2026-10-16。

| 類別 | 代表規則 | 交接注意 |
|---|---|---|
| 模型權重 | `*.gguf`、`models/*`（保留 `.gitkeep`、`README.md`） | 禁止強制加入；遺失時依 5.3 節還原 |
| 資料輸出 | `data/output/*` 內容、`*.wav`、`*.checkpoint.json` | 含個資者禁止複製至個人裝置 |
| 機敏資訊 | `.env`、`*.pem`、`*.key` | 當面或以密碼管理器移交，不得經 Git 或明文傳遞 |
| 日誌暫存 | `logs/*`（保留 `.gitkeep`）、`__pycache__/`、`.pytest_cache/` | 接手人本機重建即可 |

修改 `.gitignore` 後，請以 `git check-ignore -v` 驗證並同步更新 `README.md` 常見問題章節。

---

## 7. 帳號、權限與機敏資訊

> 本章禁止記錄密碼、金鑰明文。僅記錄帳號位置、權限範圍與移交方式。

| 項目 | 說明 | 持有人 | 移交方式 | 狀態 |
|---|---|---|---|---|
| GitHub 儲存庫管理權 | `KiraIch/PsyTest` 的 Admin 權限 | KiraIch | 由持有人於 GitHub Settings 直接授權 | 待確認接手人帳號 |
| 模型檔案存取權 | GGUF 內部下載連結或實體媒體 | KiraIch | 依 5.3 節辦理 | 待移交 |
| Ollama 模型 | 公開拉取，無需帳號 | 無 | 接手人自行 `ollama pull` | 無需移交 |
| （範例）部署主機 SSH | 正式環境存取 | （待填） | 主管核准後重建金鑰，不沿用舊金鑰 | 未啟用 |

交接完成後，移交人應即時撤銷不再需要的共用權限，並於第 14 章簽收時註記。

---

## 8. 日常維運

### 8.1 定期更新排程

| 文件 | 更新觸發條件 | 最低頻率 | 負責人 |
|---|---|---|---|
| `README.md` | 功能發佈、架構變更、設定異動 | 每三個月檢視一次 | 維護者 |
| `HANDOVER.md` | 進度變更、人員異動、風險新增 | 每次迭代結束；交接前必須更新 | 維護者 |
| `.gitignore` | 新增依賴、工具、產出物類型 | 每次新增技術債時檢視 | 維護者 |
| `NOTICE` | 新增相依套件或模型 | 即時更新 | 維護者 |
| `models/README.md` | 模型增刪或別名變更 | 即時更新 | 維護者 |

### 8.2 例行檢查指令

```bash
git status --short
git diff --check
python -m src.backend.detector
python scripts/validate_setup.py
pytest tests/ -v
git check-ignore -v models/Laya-BF16.gguf
```

### 8.3 發佈前檢查清單

- [ ] `README.md` 版本號與日期已更新。
- [ ] `HANDOVER.md` 第 3、10、13 章已更新。
- [ ] `git diff --check` 無輸出。
- [ ] 無機敏資訊與大檔案誤入暫存區。
- [ ] `LICENSE` 與 `NOTICE` 內容正確。

---

## 9. 已知風險與限制

| 編號 | 風險 | 影響 | 緩解措施 | 負責人 | 狀態 |
|---|---|---|---|---|---|
| R-01 | 模型檔案體積大，分發不易 | 新成員建置耗時；誤傳導致推送失敗 | 維持 `.gitignore` 排除；別名相容減少改名成本 | KiraIch | 持續追蹤 |
| R-02 | 模型原始授權尚未完整盤點 | 散佈或商用可能侵權 | 盤點兩份 GGUF 與 Ollama 模型授權；結果記入 `NOTICE` | （待指派） | 待處理 |
| R-03 | Phase 6 至 Phase 7 尚未實作 | 骨架方法呼叫時拋出例外 | CLI 與測試在缺件時降級為驗證或跳過；依第 10 章順序填補 | （待指派） | 待處理 |
| R-04 | 心理諮商模擬涉及敏感議題 | 誤用或外流造成危害 | 禁止真實資料進 Git；危機模板設安全紅線；上線前完成法遵審查 | （待指派） | 持續追蹤 |
| R-05 | 模擬結果不可作為臨床診斷 | 誤用可能造成危害 | 文件與未來 UI 均須載明警語；審查報告產出文案 | （待指派） | 持續追蹤 |
| R-06 | BF16 在舊硬體或舊版 llama.cpp 可能不相容 | Laya 載入失敗 | Phase 2 實作自動轉 FP16 或提示下載 FP16 版本 | （待指派） | 待處理 |
| R-07 | 24GB 記憶體同時跑多模型吃緊 | OOM 中斷批量生成 | 按需載入與卸載；批量預設串行；溫度與記憶體監控 | （待指派） | 待處理 |

新增風險時請依序編號，禁止刪除已結案風險，結案時將狀態改為「已結案」並保留紀錄。

---

## 10. 待辦事項

### 10.1 高優先級（Phase 2，已完成）

- [x] `perception/embedding_engine.py`：載入 GGUF，實作文字、批次嵌入與相似度。
- [x] `fast_think/laya_engine.py`：載入 GGUF，實作四種決策，處理 BF16 相容性。
- [x] `slow_think/ollama_client.py`：連線偵測、e4b 優先匹配、串流與重試。
- [ ] 盤點兩份 GGUF 與 Ollama 模型授權，更新 `NOTICE` 與 `models/README.md` 雜湊值。
- [ ] 建立 `develop` 分支與 GitHub Issues 標籤規範。

### 10.2 中優先級（Phase 3 至 Phase 5 已完成，Phase 6）

- [x] Phase 3：emotion_router、rhythm_controller、persona_manager、director。
- [x] Phase 4：tts_engine、kokoro/chattts/pyttsx3 後端、voice_director。
- [x] Phase 5：session_generator、output_formatter、batch_generator 與 CLI 完整實作。
4. [ ] 補充 `.env.example` 實際使用案例。

### 10.3 低優先級（Phase 6 至 Phase 7）

1. [ ] Phase 6：thermal_monitor、supervisor、記憶體管理。
2. [ ] Phase 7：全量測試啟用、README 定稿、Makefile 定稿。
3. [ ] 群體統計報表、權限管理、部署與備份策略。

---

## 11. 驗收標準

本次交接（v0.6.0）之驗收條件如下，接手人確認後方得簽收：

1. 可依第 5 章步驟完成環境建置，驗證清單全數通過。
2. 可說明模型檔名別名機制，以及遺失時的還原管道。
3. 可說明 TTS 引擎選擇順序，以及 Orpheus 為何預設停用。
4. 可說明 Phase 5 完成範圍與 Phase 6 待辦。
5. 已取得必要的儲存庫權限與模型檔案（或已約定取得時間）。

---

## 12. 聯絡資訊

| 角色 | 姓名 / 帳號 | 聯絡方式 | 備註 |
|---|---|---|---|
| 現任維護者 | KiraIch | 經由 GitHub：https://github.com/KiraIch | 儲存庫管理者 |
| 接手人 | （待填） | （待填） | 簽收後填入 |
| 審查者 | （待填） | （待填） | 重大變更審查人 |

---

## 13. 更新紀錄

| 日期 | 版本 | 變更摘要 | 作者 |
|---|---|---|---|
| 2026-10-16 | v0.1.0 | 初版建立，確立交接架構、進度基線、風險清單與待辦事項 | KiraIch |
| 2026-10-16 | v0.2.0 | 納入完整開發規格書，完成 Phase 1 與全模組骨架，新增別名機制、風險 R-06、R-07 與 ADR 三筆 | KiraIch |
| 2026-10-16 | v0.3.0 | 完成 Phase 2 三大引擎，鎖定 e4b 變體並寫入設定，測試改以真實 Ollama 驗證 | KiraIch |
| 2026-10-16 | v0.4.0 | 完成 Phase 3 智能層四模組與 22 項測試，版本對齊 0.4.0 | KiraIch |
| 2026-10-16 | v0.5.0 | 完成 Phase 4 聲音層三引擎實測，以 Kokoro 取代 Orpheus 雲端方案 | KiraIch |
| 2026-10-16 | v0.6.0 | 完成 Phase 5 數據管線與四支 CLI，端到端驗證通過 | KiraIch |
| YYYY-MM-DD | v0.x.x | （範例）完成 Phase 2，更新第 3、4、10 章 | （待填） |

---

## 14. 交接簽收

| 項目 | 移交人 | 接手人 | 日期 |
|---|---|---|---|
| 文件已更新至最新狀態 | （簽名） | — | YYYY-MM-DD |
| 環境已依第 5 章驗證通過 | — | （簽名） | YYYY-MM-DD |
| 帳號權限已移交（見第 7 章） | （簽名） | （簽名） | YYYY-MM-DD |
| 風險與待辦已說明（見第 9、10 章） | （簽名） | （簽名） | YYYY-MM-DD |

簽收後請將本表掃描或拍照存檔，並於第 13 章新增一筆紀錄。

---

*提醒：本文件與 `README.md`、`.gitignore` 採連動維護。更新其中之一時，請檢視其餘兩份是否需同步修訂。*
