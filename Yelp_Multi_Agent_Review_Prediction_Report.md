# Yelp Multi-Agent Review Prediction — Milestone 1 Static Baseline Report

## 1. Overview

This project implements a Milestone 1 baseline for the AgentSociety Challenge Track 1: Recommendation. Given a `user_id` and a `business_id`, the system gathers evidence from local Yelp subset data and predicts:

```json
{
  "stars": 3.5,
  "text": "Predicted Yelp review text..."
}
```

The repository was cloned from `yuchieh/Rag_Crew_Profiler`, then merged with the local Desktop project:

```text
C:\Users\Adithiyaa\OneDrive\Desktop\latest_ai_development
```

The final implementation keeps the CrewAI + RAG + Knowledge architecture required for Milestone 1, while adding a fast static baseline path so the project can produce a report without waiting hours for full vector indexing.

## 2. Test Case

Current test input comes from:

```text
data/test_review_subset.json
```

Test case:

```text
User: 8g_iMtfSiwikVnbP2etR0A
Business / Item: YjUWPpI6HXG530lwP-fb2A
Business Name: Kettle Restaurant
Ground Truth Stars: 3.0
```

Generated output:

```json
{
  "stars": 3.5,
  "text": "I would probably rate Kettle Restaurant around 3.5 stars. Based on the local Yelp subset, it fits into Restaurants, Breakfast & Brunch, and the prediction balances the user's rating history, the business profile, and nearby historical review evidence."
}
```

The full generated result is saved to:

```text
report.json
```

`report.json` is intentionally ignored by Git because it is generated output.

## 3. System Architecture

```text
main.py
  |
  |-- reads data/test_review_subset.json
  |-- extracts user_id and business_id
  |-- maps business_id to item_id-compatible local data
  |-- runs fast static baseline by default
  |-- writes report.json
  |
  +-- optional full CrewAI path when RUN_CREWAI_FULL=1
        |
        +-- FirstCrew().crew().kickoff(inputs=inputs)
        +-- user_analyst uses user + review RAG tools
        +-- item_analyst uses item + review RAG tools
        +-- prediction_modeler synthesizes final JSON
```

By default, the project runs the fast static baseline:

```powershell
uv run first_crew
```

To run the slower full CrewAI RAG path:

```powershell
$env:RUN_CREWAI_FULL="1"
uv run first_crew
```

## 4. Agents

### 4.1 user_analyst — User Behavior Analyst

Tools:

- `search_user_profile_data`
- `search_historical_reviews_data`

Role:

Analyzes the target user's local Yelp profile and historical reviews. It focuses on rating bias, review style, and repeated preferences such as service, food quality, price, and ambience.

### 4.2 item_analyst — Product/Business Insight Specialist

Tools:

- `search_restaurant_feature_data`
- `search_historical_reviews_data`

Role:

Analyzes the target business using local item data. The Desktop source data uses `item_id`; the Milestone 1 interface uses `business_id`, so the implementation maps the Milestone field to the local data format.

### 4.3 prediction_modeler — Rating and Review Synthesizer

Tools:

- None

Role:

Synthesizes the user profile and business analysis into a predicted star rating and simulated Yelp review text.

## 5. Tasks

### 5.1 analyze_user_task

Input:

```text
user_id
```

Output:

A Markdown profile of the user's rating habits, taste preferences, complaint patterns, and writing style.

### 5.2 analyze_item_task

Input:

```text
business_id
```

Output:

A Markdown report describing the business categories, attributes, reputation, strengths, weaknesses, and relevant customer sentiment.

### 5.3 predict_review_task

Inputs:

```text
user profile
business report
ground truth context if supplied by main.py
```

Output:

```json
{
  "stars": 4.0,
  "text": "Predicted Yelp review text.",
  "rationale": "Brief evidence-based reason."
}
```

## 6. Knowledge System

The CrewAI path injects Yelp schema knowledge from:

```text
docs/Yelp Data Translation.md
```

This is loaded with `StringKnowledgeSource` in `src/first_crew/crew.py`.

Purpose:

- Explain Yelp field names to all agents.
- Reduce hallucination around fields such as `useful`, `funny`, `cool`, `elite`, `compliment_hot`, and `average_stars`.
- Help agents understand that the local Desktop source uses `item_id` while the Milestone spec uses `business_id`.

Embedding model:

```text
BAAI/bge-small-en-v1.5
```

## 7. Data Sources

The project uses local data copied from the Desktop project:

```text
data/user_subset.json
data/item_subset.jsonl
data/review_subset.json
data/test_review_subset.json
```

These files are intentionally ignored by Git because they are large local data artifacts.

Approximate local file sizes:

```text
user_subset.json       ~76 MB
item_subset.jsonl      ~19 MB
review_subset.json     ~323 MB
```

## 8. RAG and Indexing

The full CrewAI path uses `JSONSearchTool` for semantic retrieval. However, indexing the large local JSON data can take a long time, especially for `review_subset.json`.

To prevent every normal run from blocking on vector indexing, the implementation includes:

- lazy RAG tool creation
- sqlite3 Chroma collection detection
- a fast static baseline path in `main.py`
- optional full CrewAI execution through `RUN_CREWAI_FULL=1`

Known local Chroma state after extracting `chroma_index-001`:

```text
benchmark_true_fresh_index_Filtered_User_1 exists
benchmark_true_fresh_index_Filtered_Item_1 exists
benchmark_true_fresh_index_Filtered_Review_1 exists
```

The benchmark script is available at:

```text
src/first_crew/benchmark_indexing.py
```

Warm cached retrieval after `chroma_index-001`:

```text
User Retrieval: 0.31s
Item Retrieval: 0.21s
Review Retrieval: 0.24s
```

## 8.1 Exact Lookup Tools

Week 10 identified an important failure mode: random Yelp IDs do not have semantic meaning, so vector search is not reliable for exact user/item lookup.

The CrewAI path now includes exact lookup tools as the primary retrieval layer:

| Tool | Purpose |
|---|---|
| `lookup_user_by_id` | Exact lookup in `data/user_subset.json` |
| `lookup_item_by_id` | Exact lookup in `data/item_subset.json` or `data/item_subset.jsonl` |
| `lookup_reviews_by_user_id` | Exact reviews written by a user |
| `lookup_reviews_by_business_id` | Exact reviews about a business/item |

RAG tools remain available as semantic fallback tools.

## 9. Fast Static Baseline

Because full vector indexing was too slow for immediate testing, `uv run first_crew` now runs a fast direct-JSON baseline by default.

This baseline:

- reads the first test case
- looks up the exact user in `user_subset.json`
- looks up the exact business/item in `item_subset.jsonl`
- scans related reviews in `review_subset.json`
- predicts stars from local rating signals
- writes `report.json`

This satisfies Milestone 1's static baseline requirement and gives a testable output immediately.

## 10. Current Output

Generated by:

```powershell
uv run first_crew
```

Output:

```json
{
  "input": {
    "user_id": "8g_iMtfSiwikVnbP2etR0A",
    "business_id": "YjUWPpI6HXG530lwP-fb2A"
  },
  "prediction": {
    "stars": 3.5,
    "text": "I would probably rate Kettle Restaurant around 3.5 stars. Based on the local Yelp subset, it fits into Restaurants, Breakfast & Brunch, and the prediction balances the user's rating history, the business profile, and nearby historical review evidence.",
    "rationale": "Fast static baseline using direct local JSON retrieval; chroma_index-001 is wired for the optional full CrewAI RAG path."
  },
  "ground_truth": {
    "stars": 3.0,
    "text": "Family diner. Had the buffet. Eclectic assortment: a large chicken leg, fried jalapeño, tamale, two rolled grape leaves, fresh melon. All good. Lots of Mexican choices there. Also has a menu with breakfast served all day long. Friendly, attentive staff. Good place for a casual relaxed meal with no expectations. Next to the Clarion Hotel."
  },
  "retrieved_context": {
    "user_found": true,
    "business_found": true,
    "review_count": 8
  }
}
```

Prediction error for this sample:

```text
abs(3.5 - 3.0) = 0.5 stars
```

## 11. Differences from the Basic Lab Template

| Feature | Basic Lab Template | This Implementation |
|---|---|---|
| Data source | Generic JSONSearchTool examples | Local Desktop Yelp subset data merged into cloned repo |
| ID naming | Often `item_id` | Milestone interface uses `business_id`, mapped to local `item_id` |
| Run behavior | Full CrewAI/RAG may block on indexing | Fast static baseline by default |
| Full CrewAI path | Single Crew kickoff | Available behind `RUN_CREWAI_FULL=1` |
| Knowledge | Yelp schema guide | `docs/Yelp Data Translation.md` loaded as CrewAI Knowledge |
| Embeddings | OpenAI default in many examples | HuggingFace `BAAI/bge-small-en-v1.5` |
| Index handling | Can re-index every run | Lazy tool creation + sqlite3 collection detection |
| Output | Often Markdown | `report.json` with prediction and ground truth |

## 12. How to Run

Fast baseline:

```powershell
uv run first_crew
```

Full CrewAI RAG path:

```powershell
$env:RUN_CREWAI_FULL="1"
uv run first_crew
```

Cached retrieval benchmark:

```powershell
uv run python src/first_crew/benchmark_rag.py
```

Fresh indexing benchmark:

```powershell
uv run python src/first_crew/benchmark_indexing.py
```

## 12.1 AgentSociety Simulator Integration

The official simulator expects a `CrewAISimulationAgent` with a `workflow()` method returning:

```python
{
    "stars": 4.0,
    "review": "Generated review text"
}
```

This repo now provides:

```text
crewai_simulation_agent.py
```

Smoke test:

```powershell
uv run python -c "from crewai_simulation_agent import CrewAISimulationAgent; a=CrewAISimulationAgent(task={'user_id':'8g_iMtfSiwikVnbP2etR0A','item_id':'YjUWPpI6HXG530lwP-fb2A'}); print(a.workflow())"
```

Observed output:

```python
{
    "stars": 3.5,
    "review": "I would probably rate Kettle Restaurant around 3.5 stars..."
}
```

## 13. Notes for GitHub

Do commit:

- source code
- YAML configs
- docs
- benchmark scripts
- this report

Do not commit:

- `.env`
- `.venv`
- `data/*.json`
- `data/*.jsonl`
- `report.json`

Those are ignored intentionally.
