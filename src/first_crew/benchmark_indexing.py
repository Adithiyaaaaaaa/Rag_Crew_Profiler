from __future__ import annotations

import time
from pathlib import Path

from crewai_tools import JSONSearchTool

from first_crew.crew import RAG_CONFIG

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"


def _first_existing_path(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def benchmark_single_tool(label: str, json_path: Path, collection_prefix: str, run_id: int) -> None:
    print(f"\n=== Fresh Indexing Benchmark: {label} ===")
    if not json_path.exists():
        print(f"[{label}] Skipped: missing {json_path}")
        return

    collection_name = f"milestone1_fresh_{collection_prefix}_{run_id}"
    start = time.perf_counter()
    rag_tool = JSONSearchTool(
        json_path=str(json_path),
        collection_name=collection_name,
        config=RAG_CONFIG,
    )
    index_time = time.perf_counter() - start
    print(f"[{label}] Fresh Indexing Time: {index_time:.2f}s")

    start = time.perf_counter()
    rag_tool._run(search_query="Find relevant Yelp recommendation evidence.")
    retrieval_time = time.perf_counter() - start
    print(f"[{label}] Retrieval After Fresh Index: {retrieval_time:.2f}s")


def run_indexing_benchmark() -> None:
    run_id = int(time.time())
    print("=== Milestone 1 Fresh Indexing Benchmark ===")
    benchmark_single_tool("User", DATA_DIR / "user_subset.json", "user_subset", run_id)
    benchmark_single_tool("Item", _first_existing_path(DATA_DIR / "item_subset.json", DATA_DIR / "item_subset.jsonl"), "item_subset", run_id)
    benchmark_single_tool("Review", DATA_DIR / "review_subset.json", "review_subset", run_id)


if __name__ == "__main__":
    run_indexing_benchmark()
