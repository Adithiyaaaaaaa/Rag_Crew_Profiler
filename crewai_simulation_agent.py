from __future__ import annotations

from typing import Any

from first_crew.main import predict_for_task

try:
    from websocietysimulator.agent import SimulationAgent
except Exception:
    class SimulationAgent:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.task = kwargs.get("task", {})


class CrewAISimulationAgent(SimulationAgent):
    """AgentSociety simulator adapter for the Rag_Crew_Profiler Milestone 1 crew."""

    def __init__(self, llm: Any = None, *args: Any, **kwargs: Any) -> None:
        super().__init__(llm=llm, *args, **kwargs)

    def workflow(self) -> dict[str, Any]:
        task = self.task if isinstance(self.task, dict) else {
            "user_id": getattr(self.task, "user_id", ""),
            "item_id": getattr(self.task, "item_id", ""),
            "business_id": getattr(self.task, "business_id", ""),
        }
        prediction = predict_for_task(task)
        return {
            "stars": float(prediction.get("stars", 4.0)),
            "review": str(prediction.get("review", "Good.")),
        }
