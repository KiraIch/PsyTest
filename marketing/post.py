"""社群排程發文腳本（審核制，預設只預覽不發送）。

用法：
    python marketing/post.py --dry-run [--date YYYY-MM-DD]
    python marketing/post.py --publish [--date YYYY-MM-DD]

金鑰來源：專案根目錄 .env（X_API_KEY、X_API_SECRET、
X_ACCESS_TOKEN、X_ACCESS_SECRET、THREADS_ACCESS_TOKEN、
THREADS_USER_ID）。X 發文需 pip install tweepy。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MARKETING = Path(__file__).resolve().parent
LOG_PATH = MARKETING / "log.jsonl"


def load_env() -> dict[str, str]:
    """讀取根目錄 .env（不存在則回傳空字典）。"""
    values: dict[str, str] = {}
    env_path = ROOT / ".env"
    if not env_path.exists():
        return values
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            values[key.strip()] = val.strip()
    return values


def load_posts() -> dict[str, str]:
    """由 posts.md 解析 ref 到文案的對照表。"""
    text = (MARKETING / "posts.md").read_text(encoding="utf-8")
    posts: dict[str, str] = {}
    current: str | None = None
    buffer: list[str] = []
    for line in text.splitlines():
        match = re.match(r"##\s+(P\d+)[：:]", line)
        if match:
            if current is not None:
                posts[current] = "\n".join(buffer).strip()
            current = match.group(1)
            buffer = []
        elif current is not None and not line.startswith(">") and line.strip() != "# PsyTest 文案庫":
            buffer.append(line)
    if current is not None:
        posts[current] = "\n".join(buffer).strip()
    return {key: val for key, val in posts.items() if val}


def load_calendar() -> list[dict]:
    """讀取行事曆。"""
    import yaml

    data = yaml.safe_load((MARKETING / "calendar.yaml").read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def post_x(text: str, env: dict[str, str]) -> str:
    """經 X API v2 發文，回傳貼文 ID。"""
    try:
        import tweepy
    except ImportError as exc:
        raise RuntimeError("缺少 tweepy，請執行 pip install tweepy") from exc
    missing = [key for key in ("X_API_KEY", "X_API_SECRET", "X_ACCESS_TOKEN", "X_ACCESS_SECRET") if not env.get(key)]
    if missing:
        raise RuntimeError(f".env 缺少：{', '.join(missing)}")
    client = tweepy.Client(
        consumer_key=env["X_API_KEY"],
        consumer_secret=env["X_API_SECRET"],
        access_token=env["X_ACCESS_TOKEN"],
        access_token_secret=env["X_ACCESS_SECRET"],
    )
    response = client.create_tweet(text=text)
    tweet_id = response.data.get("id") if response.data else None
    if not tweet_id:
        raise RuntimeError(f"X 發文無回傳 ID：{response}")
    return str(tweet_id)


def post_threads(text: str, env: dict[str, str]) -> str:
    """經 Threads API 發文（建草稿再發布），回傳貼文 ID。"""
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("缺少 requests，請執行 pip install -r requirements.txt") from exc
    if not env.get("THREADS_ACCESS_TOKEN") or not env.get("THREADS_USER_ID"):
        raise RuntimeError(".env 缺少 THREADS_ACCESS_TOKEN 或 THREADS_USER_ID")
    base = f"https://graph.threads.net/v1.0/{env['THREADS_USER_ID']}"
    created = requests.post(
        f"{base}/threads",
        data={"media_type": "TEXT", "text": text, "access_token": env["THREADS_ACCESS_TOKEN"]},
        timeout=30,
    ).json()
    creation_id = created.get("id")
    if not creation_id:
        raise RuntimeError(f"Threads 草稿建立失敗：{created}")
    published = requests.post(
        f"{base}/threads_publish",
        data={"creation_id": creation_id, "access_token": env["THREADS_ACCESS_TOKEN"]},
        timeout=30,
    ).json()
    post_id = published.get("id")
    if not post_id:
        raise RuntimeError(f"Threads 發布失敗：{published}")
    return str(post_id)


def log_result(item: dict, channel: str, ok: bool, detail: str) -> None:
    """寫入發送紀錄。"""
    record = {
        "date": str(date.today()),
        "id": item.get("id"),
        "channel": channel,
        "ok": ok,
        "detail": detail,
    }
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(description="社群排程發文（審核制）")
    parser.add_argument("--dry-run", action="store_true", help="只預覽不發送")
    parser.add_argument("--publish", action="store_true", help="發送已審核且到期項目")
    parser.add_argument("--date", default=None, help="指定日期 YYYY-MM-DD，預設今天")
    args = parser.parse_args(argv)
    if not args.dry_run and not args.publish:
        print("請指定 --dry-run 或 --publish", file=sys.stderr)
        return 2
    try:
        today = date.fromisoformat(args.date) if args.date else date.today()
    except ValueError:
        print("日期格式錯誤，應為 YYYY-MM-DD", file=sys.stderr)
        return 2
    try:
        posts = load_posts()
        calendar = load_calendar()
    except (OSError, ImportError) as exc:
        print(f"錯誤：讀取失敗：{exc}", file=sys.stderr)
        return 1
    env = load_env()
    due = [item for item in calendar if str(item.get("date", "")) <= today.isoformat()]
    if not due:
        print("今日無到期項目。")
        return 0
    failed = 0
    for item in due:
        ref = str(item.get("ref", ""))
        channel = str(item.get("channel", ""))
        text = posts.get(ref, "")
        tag = "DRY" if args.dry_run or not item.get("approved") else "SEND"
        print(f"[{tag}] {item.get('id')} ({channel}@{item.get('date')}) approved={bool(item.get('approved'))}")
        print(text)
        print("---")
        if args.dry_run or not item.get("approved"):
            continue
        try:
            if channel == "x":
                result = post_x(text, env)
            elif channel == "threads":
                result = post_threads(text, env)
            else:
                raise RuntimeError(f"未知平台：{channel}")
        except Exception as exc:  # noqa: BLE001
            print(f"發送失敗，已停止：{exc}", file=sys.stderr)
            log_result(item, channel, False, str(exc))
            return 1
        log_result(item, channel, True, result)
        print(f"已發送：{result}")
    return failed


if __name__ == "__main__":
    sys.exit(main())
