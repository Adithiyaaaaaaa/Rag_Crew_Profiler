#!/usr/bin/env python
from __future__ import annotations

import json
import os
import re
import warnings
from pathlib import Path
from typing import Any

from first_crew.crew import FirstCrew
from first_crew.tools.local_lookup import DATA_DIR, find_item, find_one, find_reviews, iter_json_records

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

BASE_DIR = Path(__file__).resolve().parents[2]
TEST_SET_PATH = BASE_DIR / "data" / "test_review_subset.json"
REPORT_PATH = BASE_DIR / "report.json"


def _read_first_json_record(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        first = handle.readline().strip()
        if first:
            return json.loads(first)

        handle.seek(0)
        data = json.load(handle)
        if isinstance(data, list):
            if not data:
                raise ValueError(f"{path} is empty")
            return data[0]
        return data


def _business_id(test_case: dict[str, Any]) -> str:
    value = test_case.get("business_id") or test_case.get("item_id")
    if value is None:
        raise KeyError("test case must contain business_id or item_id")
    return str(value)


def extract_json_from_output(raw_output: str) -> dict[str, Any]:
    """Extract and sanitize JSON from an LLM response."""
    text = str(raw_output).strip().replace("{{", "{").replace("}}", "}")
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if "review" in parsed and "text" not in parsed:
                parsed["text"] = parsed.pop("review")
            return parsed
        except json.JSONDecodeError:
            pass
    return {"stars": None, "text": text, "_parse_error": True}


def _static_baseline_report(inputs: dict[str, Any]) -> dict[str, Any]:
    user_id = inputs["user_id"]
    business_id = inputs["business_id"]
    user = find_one(DATA_DIR / "user_subset.json", "user_id", user_id) or {}
    item = find_item(business_id) or {}
    reviews = find_reviews(user_id=user_id, item_id=business_id, limit=8)

    star_signals: list[float] = []
    for value in (user.get("average_stars"), item.get("stars")):
        if value is not None:
            star_signals.append(float(value))
    star_signals.extend(float(review["stars"]) for review in reviews if review.get("stars") is not None)
    predicted_stars = round((sum(star_signals) / len(star_signals) if star_signals else 3.5) * 2) / 2
    predicted_stars = max(1.0, min(5.0, predicted_stars))

    item_name = item.get("name", "this business")
    categories = item.get("categories", "the listed category")
    review_text = (
        f"I would probably rate {item_name} around {predicted_stars:.1f} stars. "
        f"Based on the local Yelp subset, it fits into {categories}, and the prediction balances "
        "the user's rating history, the business profile, and nearby historical review evidence."
    )

    return {
        "input": {
            "user_id": user_id,
            "business_id": business_id,
        },
        "prediction": {
            "stars": predicted_stars,
            "text": review_text,
            "rationale": "Fast static baseline using direct local JSON retrieval; chroma_index-001 is wired for the optional full CrewAI RAG path.",
        },
        "ground_truth": {
            "stars": inputs["ground_truth_stars"],
            "text": inputs["ground_truth_text"],
        },
        "retrieved_context": {
            "user_found": bool(user),
            "business_found": bool(item),
            "review_count": len(reviews),
        },
    }


def run() -> None:
    """Run Milestone 1 against the first record in data/test_review_subset.json."""
    if not TEST_SET_PATH.exists():
        raise FileNotFoundError(
            f"Missing {TEST_SET_PATH}. Add the AgentSociety test set before running the crew."
        )

    test_case = _read_first_json_record(TEST_SET_PATH)
    inputs = {
        "user_id": str(test_case["user_id"]),
        "business_id": _business_id(test_case),
        "ground_truth_stars": test_case.get("stars"),
        "ground_truth_text": test_case.get("text") or test_case.get("review"),
    }
    inputs["item_id"] = inputs["business_id"]

    if os.getenv("RUN_CREWAI_FULL", "0") != "1":
        print("Running fast static baseline. Set RUN_CREWAI_FULL=1 to run the full CrewAI RAG pipeline.")
        report = _static_baseline_report(inputs)
    else:
        print(f"Starting Milestone 1 prediction for user={inputs['user_id']} business={inputs['business_id']}")
        result = FirstCrew().crew().kickoff(inputs=inputs)
        prediction = extract_json_from_output(result.raw)

        report = {
            "input": {
                "user_id": inputs["user_id"],
                "business_id": inputs["business_id"],
            },
            "prediction": prediction,
            "ground_truth": {
                "stars": inputs["ground_truth_stars"],
                "text": inputs["ground_truth_text"],
            },
        }
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Milestone 1 Result ===")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWrote {REPORT_PATH}")


def predict_for_task(task: dict[str, Any], use_full_crew: bool | None = None) -> dict[str, Any]:
    """Return AgentSociety-compatible prediction for one simulator task."""
    business_id = str(task.get("business_id") or task.get("item_id") or "")
    inputs = {
        "user_id": str(task.get("user_id", "")),
        "business_id": business_id,
        "item_id": business_id,
        "ground_truth_stars": task.get("stars"),
        "ground_truth_text": task.get("text") or task.get("review"),
    }
    if use_full_crew is None:
        use_full_crew = os.getenv("RUN_CREWAI_FULL", "0") == "1"

    if use_full_crew:
        result = FirstCrew().crew().kickoff(inputs=inputs)
        prediction = extract_json_from_output(result.raw)
        return {
            "stars": float(prediction.get("stars", 4.0)),
            "review": str(prediction.get("review") or prediction.get("text") or "Good."),
        }

    report = _static_baseline_report(inputs)
    return {
        "stars": float(report["prediction"]["stars"]),
        "review": str(report["prediction"]["text"]),
    }


def train() -> None:
    pass


def replay() -> None:
    pass


def test() -> None:
    pass


def run_with_trigger() -> None:
    pass
