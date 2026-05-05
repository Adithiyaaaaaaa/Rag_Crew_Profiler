# 🚀 WWW'25 AgentSociety Challenge: CrewAI Implementation

This project integrates the WWW'25 AgentSociety Challenge with the CrewAI multi-agent framework. It focuses on building intelligent LLM Agents for user behavior simulation and recommendation systems using advanced orchestration patterns.

## 🧠 Model

This project is optimized to run with **NVIDIA NIM** (NVIDIA Inference Microservices) or local models. 
The default configuration utilizes high-performance models via the NVIDIA API:
- **Default LLM:** `openai/minimaxai/minimax-m2.7` or `nvidia/llama-3.3-nemotron-super-49b-v1`
- **Embeddings:** `BAAI/bge-small-en-v1.5` (running locally via HuggingFace for cost-efficient, fast ChromaDB indexing)

## 🤖 Agents

The core architecture operates with specialized AI agents, each serving a distinct role in the simulation ecosystem:

| Agent | Role | Tools | Description |
|-------|------|-------|-------------|
| **user_analyst** | Yelp User Profiler | `Interaction Tool Wrapper` | Acts as an expert behavior analyst. It uses the interaction tool to query a target user's historical reviews and profile data to understand their taste, rating habits, and tone. |
| **item_analyst** | Yelp Restaurant Analyst | `Interaction Tool Wrapper` | Acts as a restaurant critic. It uses the interaction tool to query a target restaurant's profile and historical reviews to identify its strengths, weaknesses, and public sentiment. |
| **prediction_modeler** | Review Prediction Expert | — | Synthesizes the outputs from the User Analyst and Item Analyst. Using deep behavioral psychology, it accurately predicts the exact star rating (1.0 - 5.0) and generates the mock review text the user would write. |

*(Note: Additional experimental agents like `web_researcher`, `eda_specialist`, and `manager_agent` are available for Collaborative and Hierarchical architectures).*

## ⚙️ How It Works (The Process)

The simulation runs through a highly structured CrewAI pipeline orchestrating the agents:

1. **Task Initialization:** 
   The Official AgentSociety Simulator provides a `user_id` and an `item_id`. The `CrewAISimulationAgent` adapter receives these IDs and initializes an `InferenceState`.
2. **Tool Injection:**
   The simulator's interaction tool is dynamically injected into the CrewAI environment via the `Interaction Tool Wrapper`. This allows the agents to fetch live environment data without breaking CrewAI's sandbox.
3. **User Profiling (Task 1):** 
   The `user_analyst` uses the wrapper to query `{"query_type": "user"}` and `{"query_type": "review_by_user"}`. It analyzes the user's average stars, sentiment, and common vocabulary to generate a detailed User Profile markdown report.
4. **Item Profiling (Task 2):** 
   The `item_analyst` queries `{"query_type": "item"}` and `{"query_type": "review_by_item"}`. It compiles a comprehensive report on the restaurant's features, categories, and pros/cons.
5. **Prediction & Synthesis (Task 3):**
   The `prediction_modeler` ingests both the User Profile and Item Report. It evaluates if the restaurant's features align with the user's historical preferences, outputting a strict JSON dictionary containing the predicted `stars` and simulated `review`.
6. **Result Submission:** 
   The output is captured by the Serving Flow and passed back to the official simulator for evaluation against hidden Ground Truth data.

## 🔍 RAG Tools

All three RAG tools are backed by ChromaDB collections indexed with `BAAI/bge-small-en-v1.5` embeddings:

| Tool | Collection | Source Data |
|------|------------|-------------|
| `search_user_profile_data` | `v3_hf_user_data` | `user_subset.json` |
| `search_restaurant_feature_data` | `v3_hf_item_data` | `item_subset.json` |
| `search_historical_reviews_data` | `v3_hf_review_data` | `review_subset.json` |

> **Note:** Each tool accepts only a plain natural language string as `search_query`. Passing structured JSON objects will cause a `FixedJSONSearchToolSchema` validation error.

## 🏛️ Crew Architectures

This project explores three distinct CrewAI architectural patterns to optimize agent collaboration and task execution:

### 1. Sequential Crew — Cascade Pattern
- **Logic:** Tasks are executed in a fixed linear order.
- **Workflow:** Each agent is assigned a specific task. Once a task is completed, its output is passed downstream as context to the next agent.
- **Best For:** Straightforward pipelines where each step depends directly on the result of the previous one (e.g., User Analysis → Item Analysis → Final Prediction).
- **Process Type:** `Process.sequential`

### 2. Collaborative Single-Task Crew
- **Logic:** A single, shared "Master Task" is owned by a lead agent.
- **Workflow:** The lead agent (e.g., `prediction_modeler`) has `allow_delegation=True`. It dynamically pulls in specialists (User Analyst, Item Analyst, etc.) to perform sub-tasks or provide data as needed before synthesizing the final output.
- **Best For:** Complex tasks that require a central "mind" to coordinate multiple specialists without the overhead of a dedicated manager agent.
- **Process Type:** `Process.sequential` (with delegation enabled)

### 3. Hierarchical Crew — Manager-Delegated Pattern
- **Logic:** A dedicated `manager_agent` (or an LLM-managed process) orchestrates all work.
- **Workflow:** The manager agent receives the high-level goal and dynamically delegates tasks to workers independently. It can assign tasks in parallel, manage dependencies, and ensure the final output meets all requirements.
- **Best For:** Highly complex workflows where the order of execution might change based on intermediate results, or where parallel execution of sub-tasks is critical for efficiency.
- **Process Type:** `Process.hierarchical`

## 📊 Performance

| Architecture | Preference Estimation (%) | Review Generation (%) | Overall Quality (%) |
|--------------|---------------------------|-----------------------|---------------------|
| Sequential | 81.37 | 79.75 | 80.56 |
| Collaborative | 80.42 | 82.18 | 81.30 |
| Hierarchical | 82.44 | 79.47 | 80.95 |

## 📂 Project Structure

- `src/`: Contains the core logic and Crew definitions.
- `crews/`: Implementations of the different Crew architectures.
- `flows/`: Higher-level orchestration (e.g., ServingFlow) that manages the simulation lifecycle.
- `config/`: YAML files defining Agent roles and Task descriptions, strictly separated from Python logic.
- `websocietysimulator/`: The underlying simulation environment and tools (Interaction, Evaluation).
- `data/`: Processed Yelp/Amazon/Goodreads datasets.

## 🛠️ Quick Start

### 1. Prerequisites
- [Astral uv](https://docs.astral.sh/uv/) (for blazing fast dependency management)
- Python 3.10+

### 2. Installation
```bash
uv sync
```

### 3. Environment Setup
Create a `.env` file and add your API keys:

```dotenv
LLM_PROVIDER=nvidia # LLM Selection: nvidia, ollama, or groq
OPENAI_API_KEY=nvidia_api_key
OPENAI_API_BASE=https://integrate.api.nvidia.com/v1

NVIDIA_API_KEY=nvidia_api_key
NVIDIA_MODEL_NAME=nvidia_model_name # e.g. meta/llama-3.1-8b-instruct
NVIDIA_API_BASE=https://integrate.api.nvidia.com/v1
```

### 4. Run Simulation

```bash
# Verify setup with Mock Mode (zero cost)
uv run python run_simulator_test.py --mock

# Run full simulation
uv run python run_simulator_test.py

# If you already have the output
## By default in run_simulator_test.py, the save_dir is simulator_results
uv run python run_simulator_test.py --eval-only
```

### Example Output
```json
{
  "task": {
    "description": "This is a simulation task. \n            You are a simulation agent that simulates a user's rating and review with an item. \n            There is a user with id and an item with id. ",
    "user_id": "JTqQ9C9S2Qc8_aHtO1965g",
    "item_id": "QGYzYUMsQe6k7__LD91E5w"
  },
  "output": {
    "stars": 4.5,
    "review": "I'm so glad I stumbled upon this hidden gem in Chinatown! The milk tea is rich and creamy, and the price is unbeatable at $2. The staff is friendly and attentive, and the atmosphere is lively and fun. I've been here a few times now, and I always leave feeling satisfied and happy. The only reason I'm not giving it 5 stars is because the service can be a bit slow at times, but overall, I highly recommend this place to anyone looking for a delicious and affordable drink in the heart of Chinatown."
  }
}
```

## 📖 References
- [AgentSociety Challenge Official](https://www.agentsocietychallenge.com/)
- [AgentSocietyChallenge_w_CrewAI Repository](https://github.com/yuchieh/AgentSocietyChallenge_w_CrewAI)
- [CrewAI Documentation](https://docs.crewai.com/)
