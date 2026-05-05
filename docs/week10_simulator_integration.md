# Week 10 Simulator Integration Notes

This repo now follows the Week 10 direction:

- Uses cached Chroma collections from `chroma_index-001`.
- Uses exact ID lookup tools first, because Yelp IDs are random strings and semantic RAG is unreliable for exact ID lookup.
- Keeps RAG tools as fallback/search tools.
- Provides a simulator adapter at `crewai_simulation_agent.py`.

## Local Run

```powershell
uv run first_crew
```

## RAG Benchmark

```powershell
uv run python src/first_crew/benchmark_rag.py
```

Observed after extracting `chroma_index-001`:

```text
User Retrieval: 1.52s
Item Retrieval: 0.26s
Review Retrieval: 0.27s
```

## Simulator Contract

The official simulator expects an agent class with `workflow()` returning:

```python
{
    "stars": 4.0,
    "review": "Generated review text"
}
```

This repo exposes that class:

```python
from crewai_simulation_agent import CrewAISimulationAgent
```

`CrewAISimulationAgent.workflow()` calls `first_crew.main.predict_for_task()`.

By default it uses the fast static baseline. To use the full CrewAI RAG path:

```powershell
$env:RUN_CREWAI_FULL="1"
```
