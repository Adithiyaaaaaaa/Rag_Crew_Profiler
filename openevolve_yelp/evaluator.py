from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from first_crew.tools.local_lookup import DATA_DIR, find_item, find_one, find_reviews, iter_json_records

TEST_PATH = DATA_DIR / "test_review_subset.json"


def _load_program(program_path: str | Path):
    path = Path(program_path).resolve()
    spec = importlib.util.spec_from_file_location("candidate_program", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load program from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _business_id(record: dict[str, Any]) -> str:
    return str(record.get("business_id") or record.get("item_id") or "")


def _evaluate_records(program, max_cases: int | None = None) -> dict[str, float]:
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Missing test set: {TEST_PATH}")

    absolute_errors: list[float] = []
    exact_half_star_hits = 0
    evaluated = 0

    for record in iter_json_records(TEST_PATH):
        user_id = str(record.get("user_id") or "")
        item_id = _business_id(record)
        truth = record.get("stars")
        if not user_id or not item_id or truth is None:
            continue

        user = find_one(DATA_DIR / "user_subset.json", "user_id", user_id) or {}
        item = find_item(item_id) or {}
        user_reviews = find_reviews(user_id=user_id, limit=10)
        item_reviews = find_reviews(item_id=item_id, limit=10)

        prediction = float(program.predict_stars(user, item, user_reviews, item_reviews))
        prediction = max(1.0, min(5.0, prediction))
        error = abs(prediction - float(truth))
        absolute_errors.append(error)
        exact_half_star_hits += int(error <= 0.5)
        evaluated += 1

        if max_cases is not None and evaluated >= max_cases:
            break

    if not absolute_errors:
        return {
            "combined_score": 0.0,
            "mae": 4.0,
            "accuracy_within_0_5": 0.0,
            "evaluated_count": 0.0,
            "runs_successfully": 0.0,
        }

    mae = sum(absolute_errors) / len(absolute_errors)
    rmse = math.sqrt(sum(error * error for error in absolute_errors) / len(absolute_errors))
    within_half = exact_half_star_hits / len(absolute_errors)

    return {
        "combined_score": max(0.0, 1.0 - mae / 4.0),
        "mae": mae,
        "rmse": rmse,
        "accuracy_within_0_5": within_half,
        "evaluated_count": float(evaluated),
        "runs_successfully": 1.0,
    }


def evaluate(program_path: str) -> dict[str, float]:
    """OpenEvolve entrypoint."""
    try:
        program = _load_program(program_path)
        return _evaluate_records(program)
    except Exception as exc:
        return {
            "combined_score": 0.0,
            "mae": 4.0,
            "rmse": 4.0,
            "accuracy_within_0_5": 0.0,
            "evaluated_count": 0.0,
            "runs_successfully": 0.0,
            "error": str(exc),
        }


if __name__ == "__main__":
    candidate = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).with_name("initial_program.py"))
    print(json.dumps(evaluate(candidate), indent=2, ensure_ascii=False))
