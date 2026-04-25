# Milestone 1: Static Baseline System (CrewAI + RAG + Knowledge)

本指南專注於引導學生完成 **AgentSociety Challenge - Milestone 1**。目標是基於目前的 `src/first_crew/` 架構，結合 CrewAI 的 Knowledge 與 RAG Tools，打造一個能預測 `data/test_review_subset.json` 中使用者星等與評論內容的多智能體系統 (Multi-Agent System)。

## 教學與實作步驟 (Step-by-Step Guide)

### Step 1: 了解目標與資料集 (Data & Objective)

- **輸入 (Input):** 從 `data/test_review_subset.json` 中抽取出的一組 `{user_id}` 與 `{business_id}`。
- **輸出 (Output):** AI Agents 綜合判斷後預測出的 `{stars}` (星等) 與 `{text}` (評論內容)。
- **檢索池 (Retrieval Pool):** `data/user_subset.json`, `data/item_subset.json`, `data/review_subset.json`。

### Step 2: 注入全局背景知識 (CrewAI Knowledge)

目的：讓所有 Agents 都能「先天」看懂 Yelp JSON 的欄位名稱定義，避免在讀取資料時對 `compliment_hot`、`useful`、`funny`、`cool` 等欄位產生解釋錯誤的幻覺 (Hallucination)。

實作方法：在 `src/first_crew/crew.py` 中使用 `StringKnowledgeSource` 讀入 `docs/Yelp Data Translation.md`，並在 `Crew(...)` 中透過 `knowledge_sources=[schema_knowledge]` 進行全局綁定。

### Step 3: 配置主動檢索武器 (CrewAI RAG Tools)

目的：為 Agents 裝備能夠搜尋本機 JSON 檔案的工具。

```python
user_rag_tool = JSONSearchTool(json_path="data/user_subset.json")
item_rag_tool = JSONSearchTool(json_path="data/item_subset.json")
review_rag_tool = JSONSearchTool(json_path="data/review_subset.json")
```

本專案另外加入 sqlite3 collection cache 檢查：若 ChromaDB 已有 Desktop `latest_ai_development` 既有 collection (`user_data_v4`, `item_data_v4`, `review_data_v4`)，工具會直接掛載 collection，避免重新索引。

### Step 4: 嚴格分離的 Agent 定義 (`config/agents.yaml`)

- `user_analyst` (使用者輪廓分析師): 專注於分析特定 user 的歷史行為、給分習慣與口味偏好。
- `item_analyst` (店家分析師): 專注於分析特定 business 的設施、類別、地點與總體評價。
- `prediction_modeler` (評論預測家): 讀取前兩者的分析報告，進行最終星等與評論內容預測。

### Step 5: 嚴格分離的 Task 定義 (`config/tasks.yaml`)

- `analyze_user_task`: 使用 `{user_id}` 尋找使用者的歷史紀錄，輸出其「給分習慣與口味偏好」。
- `analyze_item_task`: 使用 `{business_id}` 尋找店家的特徵與別人的評論，輸出「餐廳優劣勢總結」。
- `predict_review_task`: 綜合上述兩份報告，輸出 JSON，內部包含預測的 `stars` 與模擬寫出的 `text`。

### Step 6: 系統組裝與工具綁定 (`src/first_crew/crew.py`)

透過 Decorator 將 RAG Tools 掛載給正確的 Agent：

- `user_analyst`: `tools=[user_rag_tool, review_rag_tool]`
- `item_analyst`: `tools=[item_rag_tool, review_rag_tool]`
- `prediction_modeler`: 不直接檢索，只使用前兩個 Agent 的 context 進行綜合判斷
- `Crew(...)`: 加入 `knowledge_sources=[schema_knowledge]` 與本地 embedding 設定

### Step 7: 測試執行腳本 (`src/first_crew/main.py`)

`main.py` 會讀取 `data/test_review_subset.json` 的第一筆測試資料，擷取 `user_id` 與 `business_id`，將其作為 inputs 丟入：

```python
FirstCrew().crew().kickoff(inputs=inputs)
```

最後將模型輸出的星等/評論與 JSON 內的 Ground Truth 進行比對，並寫入 `report.json`，完成 Milestone 1 的 Static Baseline。

## 本地量測記錄

- User Retrieval: 0.04s
- Item Retrieval: 0.07s
- Review Retrieval: 0.04s

Fresh indexing benchmark 請執行：

```bash
uv run python src/first_crew/benchmark_indexing.py
```
