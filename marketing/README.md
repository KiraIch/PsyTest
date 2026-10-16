# 社群推廣作業區（本機執行，不含任何金鑰）

> 金鑰一律放 `.env`（已於 `.gitignore` 排除，禁止提交）。
> 發文流程為審核制：行事曆上 `approved: false` 的項目只會列印預覽，不會發出。

## 一、X API 申請流程

1. 前往 https://developer.x.com ，以你的 X 帳號登入。
2. 建立 Project（免費層即可，每月約有數百則發文額度，夠排程使用）。
3. 在 App 設定中將權限改為 Read and Write，並重新產生金鑰。
4. 取得四組值：API Key、API Key Secret、Access Token、Access Token Secret。
   注意 Access Token 必須是該 App 產生、綁定你帳號的那組（OAuth 1.0a 使用者身分）。
5. 寫入專案根目錄 `.env`：
   ```bash
   X_API_KEY=填入
   X_API_SECRET=填入
   X_ACCESS_TOKEN=填入
   X_ACCESS_SECRET=填入
   ```
6. 安裝依賴：`pip install tweepy`。

## 二、Threads API 申請流程

1. 以 Facebook 帳號登入 https://developers.facebook.com/apps ，建立應用程式。
2. 加入 Threads API 用品，完成必要的基本設定（隱私政策網址可用 repo 連結）。
3. 以你的 Threads 帳號授權，取得短期權杖後換長期權杖（約 60 天效期，到期重換）。
   同時記下 Threads 使用者 ID（以 `/me?fields=id` 查得）。
4. 寫入 `.env`：
   ```bash
   THREADS_ACCESS_TOKEN=填入
   THREADS_USER_ID=填入
   ```
5. 注意：Meta 流程與權限名稱常變動，以官方文件為準；部分權限需送審才可用於非測試帳號。

## 三、使用方式

```bash
# 預覽當日待發項目（不發送）
python marketing/post.py --dry-run

# 只發送已審核（approved: true）且日期到期的項目
python marketing/post.py --publish

# 指定日期預覽
python marketing/post.py --dry-run --date 2026-10-20
```

## 四、檔案說明

- `calendar.yaml`：發文行事曆，新增貼文時設 `approved: false`，你審過後改 `true`。
- `posts.md`：文案庫，含中英文草稿。
- `post.py`：排程發文腳本，預設試運行，發送紀錄寫入 `log.jsonl`。
- `SESSION.md`：新開發文專用 session 時貼上的啟動提示詞。

## 五、基本規則

1. 同一內容不同平台至少間隔一天，不洗版。
2. 機器只發審核過的內容，回留言一律由你本人回。
3. 發送失敗或 API 回傳限流時，腳本會停下並記錄，不硬重試。
