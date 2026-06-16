from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any, List

from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import JSONSearchTool
from dotenv import load_dotenv
from crewai.tools import tool

from first_crew.tools.local_lookup import DATA_DIR as LOCAL_DATA_DIR
from first_crew.tools.local_lookup import compact_user, dumps_record, find_item, find_one, find_reviews

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = BASE_DIR / "docs"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RAG_CONFIG = {
    "embedding_model": {
        "provider": "sentence-transformer",
        "config": {"model_name": EMBEDDING_MODEL},
    }
}


def configure_llm_provider() -> None:
    """Configure LiteLLM-compatible environment variables for CrewAI."""
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    if provider == "nvidia":
        os.environ["MODEL"] = f"openai/{os.getenv('NVIDIA_MODEL_NAME', 'minimaxai/minimax-m2.7')}"
        os.environ["OPENAI_API_BASE"] = os.getenv("NVIDIA_API_BASE", "https://integrate.api.nvidia.com/v1")
        os.environ["OPENAI_API_KEY"] = os.getenv("NVIDIA_API_KEY", "")
    else:
        os.environ["MODEL"] = os.getenv("OLLAMA_MODEL", "ollama/phi3")
        os.environ.setdefault("OPENAI_API_KEY", "NA")


def _crewai_chroma_db() -> Path:
    from crewai.utilities.paths import db_storage_path

    return Path(db_storage_path()) / "chroma.sqlite3"


def _collection_exists(collection_name: str) -> bool:
    db_file = _crewai_chroma_db()
    if not db_file.exists():
        return False

    try:
        with sqlite3.connect(db_file) as connection:
            row = connection.execute(
                "select id from collections where name = ?",
                (collection_name,),
            ).fetchone()
    except sqlite3.Error:
        return False
    return row is not None


def _missing_data_tool(json_path: Path, name: str, description: str) -> Any:
    from crewai.tools import tool

    @tool(name)
    def missing_data_search(search_query: str) -> str:
        """Explain that the local Milestone 1 JSON data file is missing."""
        return (
            f"Cannot search yet because {json_path} is missing. "
            "Add the AgentSociety Milestone 1 data files under data/ and rerun."
        )

    missing_data_search.description = description
    return missing_data_search


def _first_existing_path(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def _first_existing_collection(*collection_names: str) -> str:
    for collection_name in collection_names:
        if _collection_exists(collection_name):
            return collection_name
    return collection_names[0]


def create_rag_tool(json_path: Path, collection_name: str, name: str, description: str) -> Any:
    """
    Create a JSONSearchTool with a Chroma cache fast path.

    If the collection already exists, mounting by collection_name avoids CrewAI Tools re-reading
    and re-chunking the whole JSON file on every run.
    """
    if not json_path.exists() and not _collection_exists(collection_name):
        return _missing_data_tool(json_path, name, description)

    if _collection_exists(collection_name):
        from crewai_tools.tools.json_search_tool.json_search_tool import FixedJSONSearchToolSchema

        tool = JSONSearchTool(collection_name=collection_name, config=RAG_CONFIG)
        tool.args_schema = FixedJSONSearchToolSchema
    else:
        tool = JSONSearchTool(json_path=str(json_path), collection_name=collection_name, config=RAG_CONFIG)

    tool.name = name
    tool.description = description
    return tool


configure_llm_provider()

_USER_RAG_TOOL: Any | None = None
_ITEM_RAG_TOOL: Any | None = None
_REVIEW_RAG_TOOL: Any | None = None


def get_user_rag_tool() -> Any:
    global _USER_RAG_TOOL
    if _USER_RAG_TOOL is None:
        _USER_RAG_TOOL = create_rag_tool(
            json_path=DATA_DIR / "user_subset.json",
            collection_name=_first_existing_collection(
                "benchmark_true_fresh_index_Filtered_User_1",
                "v3_hf_user_data",
                "v4_nv_user_data",
                "user_data_v4",
                "milestone1_user_subset",
            ),
            name="search_user_profile_data",
            description=(
                "Search Yelp user profile records. Input must be a natural language search_query string, "
                "for example: 'Find profile and rating habits for user USER_ID'."
            ),
        )
    return _USER_RAG_TOOL


def get_item_rag_tool() -> Any:
    global _ITEM_RAG_TOOL
    if _ITEM_RAG_TOOL is None:
        _ITEM_RAG_TOOL = create_rag_tool(
            json_path=_first_existing_path(DATA_DIR / "item_subset.json", DATA_DIR / "item_subset.jsonl"),
            collection_name=_first_existing_collection(
                "benchmark_true_fresh_index_Filtered_Item_1",
                "v3_hf_item_data",
                "item_data_v4",
                "milestone1_item_subset",
            ),
            name="search_restaurant_feature_data",
            description=(
                "Search Yelp business/item records from the Desktop latest_ai_development knowledge data. "
                "Input must be a natural language search_query string, for example: "
                "'Find categories, attributes, stars, and location for item_id BUSINESS_ID'."
            ),
        )
    return _ITEM_RAG_TOOL


def get_review_rag_tool() -> Any:
    global _REVIEW_RAG_TOOL
    if _REVIEW_RAG_TOOL is None:
        _REVIEW_RAG_TOOL = create_rag_tool(
            json_path=DATA_DIR / "review_subset.json",
            collection_name=_first_existing_collection(
                "benchmark_true_fresh_index_Filtered_Review_1",
                "v3_hf_review_data",
                "review_data_v4",
                "milestone1_review_subset",
            ),
            name="search_historical_reviews_data",
            description=(
                "Search Yelp historical reviews. Input must be a natural language search_query string, "
                "for example: 'Find reviews by user USER_ID or about business BUSINESS_ID'."
            ),
        )
    return _REVIEW_RAG_TOOL


@tool("lookup_user_by_id")
def lookup_user_by_id(user_id: str) -> str:
    """Exact lookup for a Yelp user_id. Use this before semantic RAG when the ID is known."""
    record = find_one(LOCAL_DATA_DIR / "user_subset.json", "user_id", str(user_id))
    return dumps_record(compact_user(record)) if record else f"No exact user found for user_id={user_id}"


@tool("lookup_item_by_id")
def lookup_item_by_id(item_id: str) -> str:
    """Exact lookup for a Yelp item_id/business_id. Use this before semantic RAG when the ID is known."""
    record = find_item(str(item_id))
    return dumps_record(record) if record else f"No exact item found for item_id={item_id}"


@tool("lookup_reviews_by_user_id")
def lookup_reviews_by_user_id(user_id: str) -> str:
    """Exact lookup for historical reviews written by a user_id."""
    records = find_reviews(user_id=str(user_id), limit=10)
    return dumps_record(records) if records else f"No exact reviews found for user_id={user_id}"


@tool("lookup_reviews_by_business_id")
def lookup_reviews_by_business_id(business_id: str) -> str:
    """Exact lookup for historical reviews about an item_id/business_id."""
    records = find_reviews(item_id=str(business_id), limit=10)
    return dumps_record(records) if records else f"No exact reviews found for business_id={business_id}"

schema_path = DOCS_DIR / "Yelp Data Translation.md"
schema_knowledge = StringKnowledgeSource(
    content=schema_path.read_text(encoding="utf-8") if schema_path.exists() else "Yelp schema guide unavailable.",
    metadata={"source": str(schema_path)},
)


@CrewBase
class FirstCrew:
    """Milestone 1 Yelp recommendation crew."""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def user_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["user_analyst"],  # type: ignore[index]
            tools=[lookup_user_by_id, lookup_reviews_by_user_id, get_user_rag_tool(), get_review_rag_tool()],
            verbose=True,
            max_rpm=int(os.getenv("AGENT_MAX_RPM", "15")),
        )

    @agent
    def item_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["item_analyst"],  # type: ignore[index]
            tools=[lookup_item_by_id, lookup_reviews_by_business_id, get_item_rag_tool(), get_review_rag_tool()],
            verbose=True,
            max_rpm=int(os.getenv("AGENT_MAX_RPM", "15")),
        )

    @agent
    def prediction_modeler(self) -> Agent:
        return Agent(
            config=self.agents_config["prediction_modeler"],  # type: ignore[index]
            verbose=True,
            max_rpm=int(os.getenv("AGENT_MAX_RPM", "15")),
        )

    @task
    def analyze_user_task(self) -> Task:
        return Task(config=self.tasks_config["analyze_user_task"])  # type: ignore[index]

    @task
    def analyze_item_task(self) -> Task:
        return Task(config=self.tasks_config["analyze_item_task"])  # type: ignore[index]

    @task
    def predict_review_task(self) -> Task:
        return Task(
            config=self.tasks_config["predict_review_task"],  # type: ignore[index]
            output_file="report.json",
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            knowledge_sources=[schema_knowledge],
            embedder={
                "provider": "huggingface",
                "config": {"model": EMBEDDING_MODEL},
            },
            verbose=True,
            max_rpm=int(os.getenv("CREW_MAX_RPM", "15")),
        )
