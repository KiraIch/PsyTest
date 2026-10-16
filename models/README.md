# models — 模型檔案說明

本目錄用於存放本機推論所需之模型權重檔。基於檔案體積與授權散佈限制，權重檔不納入 Git 版本控制，已於根目錄 `.gitignore` 中排除。

- 最後更新：2026-10-16
- 維護者：KiraIch

## 檔案清單

本目錄應包含以下兩個權重檔（檔名須完全一致）：

| 檔案名稱 | 用途 | 大小（約） | Git 狀態 | 載入狀態 |
|---|---|---|---|---|
| `Laya-BF16.gguf` | 快思層決策模型 | 805 MB | 已排除，不提交 | 僅備援，決策預設使用官方 laya 套件 |
| `embeddinggemma-2-Q8_0.gguf` | 五感層嵌入模型 | 296 MB | 已排除，不提交 | llama.cpp 尚不支援，改用 ST 後端 |

## 取得方式

1. 由專案維護者經由內部安全管道提供，不得使用公開連結散佈。
2. 取得後請放置於本目錄，檔名須與上表一致。
3. 放置完成後請執行完整性驗證（見下一節）。

## 完整性驗證

雜湊值（SHA-256）將於正式提供後補充。驗證範例：

```bash
shasum -a 256 models/Laya-BF16.gguf
shasum -a 256 models/embeddinggemma-2-Q8_0.gguf
```

| 檔案名稱 | SHA-256 | 提供日期 |
|---|---|---|
| `Laya-BF16.gguf` | （待補充） | （待補充） |
| `embeddinggemma-2-Q8_0.gguf` | （待補充） | （待補充） |

## 慢想層模型（Ollama）

慢想層使用 Ollama 管理的 e4b 變體，本專案已鎖定以下標籤
（本機 `:latest` 即 e4b 內容，6.3GB，已驗證存在）：

```bash
ollama pull fredrezones55/Gemma-4-Uncensored-HauhauCS-Aggressive
ollama list
```

- 鎖定標籤：`fredrezones55/Gemma-4-Uncensored-HauhauCS-Aggressive:latest`
- 對應變體：e4b（4B 參數，128K context，支援文字與影像）。
- 四份 `configs/*.yaml` 的 `llm.model_name` 均已明確指定該標籤；
  `OllamaClient.detect_model` 在 `auto` 模式下亦優先選擇 e4b。
- 如需更換變體（例如改用 e2b 節省記憶體），請同步修改四份設定、
  `PINNED_MODEL` 常數與本節說明，並記錄於 `HANDOVER.md` 第 13 章。

## 聲音層模型（Kokoro，存放於版本控制之外）

Kokoro 模型不存放於本目錄，預設位於 `~/.cache/psytest/kokoro-multi-lang-v1_1`
（可經 `PSYTEST_KOKORO_DIR` 覆寫）。缺檔時 `KokoroBackend` 會報出以下下載指令：

```bash
mkdir -p ~/.cache/psytest
curl -sL -o ~/.cache/psytest/kokoro-multi-lang-v1_1.tar.bz2 \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-multi-lang-v1_1.tar.bz2
tar -xjf ~/.cache/psytest/kokoro-multi-lang-v1_1.tar.bz2 -C ~/.cache/psytest/
```

ChatTTS 模型由 `ChatTTS.Chat().load(source="huggingface")` 自動取得，
存放於 HuggingFace 快取，無需手動放置。

## 嵌入替代方案（sentence-transformers，需授權）

`google/embeddinggemma-300m` 為授權管制模型，需兩步驟方可使用：

1. 以瀏覽器登入 HuggingFace，前往
   https://huggingface.co/google/embeddinggemma-300m 接受授權。
2. 本機登入後下載：
   ```bash
   pip install sentence-transformers huggingface_hub
   hf download google/embeddinggemma-300m
   ```

完成後以 `backend="st"` 使用：

```python
from src.perception.embedding_engine import EmbeddingEngine

with EmbeddingEngine("models/embeddinggemma2-q8.gguf", backend="st") as engine:
    vec = engine.embed_text("個案表達強烈的無助感")
```

## 授權注意

- 使用前須確認各模型原始授權允許本專案之使用方式。
- 授權資訊確認後將同步更新至根目錄 `NOTICE` 檔案。
- 禁止將本目錄內容上傳至公開儲存庫或公開雲端空間。

## 目錄佔位

`.gitkeep` 用於在權重檔缺席時維持目錄結構，請勿刪除。
