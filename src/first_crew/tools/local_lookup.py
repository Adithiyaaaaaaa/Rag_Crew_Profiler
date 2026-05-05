from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data"


def iter_json_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        first = handle.read(1)
        handle.seek(0)
        if first == "[":
            data = json.load(handle)
            yield from data
            return

        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def find_one(path: Path, key: str, value: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    for record in iter_json_records(path):
        if str(record.get(key)) == value:
            return record
    return None


def find_item(item_id: str) -> dict[str, Any] | None:
    return (
        find_one(DATA_DIR / "item_subset.json", "business_id", item_id)
        or find_one(DATA_DIR / "item_subset.json", "item_id", item_id)
        or find_one(DATA_DIR / "item_subset.jsonl", "business_id", item_id)
        or find_one(DATA_DIR / "item_subset.jsonl", "item_id", item_id)
    )


def find_reviews(user_id: str | None = None, item_id: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
    path = DATA_DIR / "review_subset.json"
    if not path.exists():
        return []

    matches: list[dict[str, Any]] = []
    for record in iter_json_records(path):
        record_user = str(record.get("user_id"))
        record_item = str(record.get("business_id") or record.get("item_id"))
        if (user_id and record_user == user_id) or (item_id and record_item == item_id):
            matches.append(record)
            if len(matches) >= limit:
                break
    return matches


def compact_user(record: dict[str, Any]) -> dict[str, Any]:
    compact = dict(record)
    compact.pop("friends", None)
    return compact


def dumps_record(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)
