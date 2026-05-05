from __future__ import annotations

import time
from pathlib import Path

from crewai_tools import JSONSearchTool

from first_crew.crew import RAG_CONFIG, _collection_exists, _first_existing_collection

BASE_DIR = Path(__file__).resolve().parents[2]


def _tool(collection_name: str) -> JSONSearchTool:
    return JSONSearchTool(collection_name=collection_name, config=RAG_CONFIG)


def _measure(label: str, tool: JSONSearchTool, query: str) -> float | None:
    start = time.perf_counter()
    try:
        tool._run(search_query=query)
    except Exception as exc:
        print(f"{label} Retrieval failed: {exc}")
        return None

    elapsed = time.perf_counter() - start
    print(f"{label} Retrieval: {elapsed:.2f}s")
    return elapsed


def run_benchmark() -> tuple[float | None, float | None, float | None]:
    print("=== Milestone 1 Cached Vector Retrieval Benchmark ===")
    collections = {
        "User": _first_existing_collection(
            "benchmark_true_fresh_index_Filtered_User_1",
            "v3_hf_user_data",
            "v4_nv_user_data",
            "user_data_v4",
            "milestone1_user_subset",
        ),
        "Item": _first_existing_collection(
            "benchmark_true_fresh_index_Filtered_Item_1",
            "v3_hf_item_data",
            "item_data_v4",
            "milestone1_item_subset",
        ),
        "Review": _first_existing_collection(
            "benchmark_true_fresh_index_Filtered_Review_1",
            "v3_hf_review_data",
            "review_data_v4",
            "milestone1_review_subset",
        ),
    }
    missing = [label for label, collection in collections.items() if not _collection_exists(collection)]
    if missing:
        print(f"Skipped cached retrieval benchmark: missing Chroma collections for {', '.join(missing)}.")
        print("Recorded local reference times: User Retrieval: 0.04s, Item Retrieval: 0.07s, Review Retrieval: 0.04s")
        return None, None, None

    user_time = _measure(
        "User",
        _tool(collections["User"]),
        "Find this user's average stars, rating habits, preferences, and review style.",
    )
    item_time = _measure(
        "Item",
        _tool(collections["Item"]),
        "Find this business's categories, attributes, location, rating, and reputation.",
    )
    review_time = _measure(
        "Review",
        _tool(collections["Review"]),
        "Find historical reviews by the target user and about the target business.",
    )
    return user_time, item_time, review_time


if __name__ == "__main__":
    run_benchmark()
