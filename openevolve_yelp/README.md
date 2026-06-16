# OpenEvolve Harness for Yelp Rating Prediction

This folder prepares the project for the Week 11 direction: make the crew evolutionary.

Instead of evolving the whole CrewAI stack, OpenEvolve evolves the deterministic rating strategy used by the fast static baseline. That gives us a clear quantitative metric: lower MAE against `data/test_review_subset.json`.

## Files

- `initial_program.py`: seed algorithm that predicts stars from user, item, and review signals.
- `evaluator.py`: computes MAE, RMSE, accuracy within 0.5 stars, and `combined_score`.
- `config.yaml`: OpenEvolve config targeting NVIDIA NIM / MiniMax.

`initial_program.py` contains exactly one evolvable region:

```python
# EVOLVE-BLOCK-START
def predict_stars(...):
    ...
# EVOLVE-BLOCK-END
```

Helpers and imports stay outside the block so OpenEvolve only mutates the rating strategy.

## Local Evaluation

From the repo root:

```powershell
uv run python openevolve_yelp/evaluator.py openevolve_yelp/initial_program.py
```

## OpenEvolve Run

After cloning OpenEvolve separately:

```powershell
git clone https://github.com/algorithmicsuperintelligence/openevolve.git
```

For this project, run from the `Rag_Crew_Profiler` repo root and point to OpenEvolve's runner:

```powershell
uv run --env-file .env python ..\openevolve\openevolve-run.py openevolve_yelp\initial_program.py openevolve_yelp\evaluator.py --config openevolve_yelp\config.yaml
```

Adjust the relative path to `openevolve-run.py` depending on where you cloned OpenEvolve.

## Why This Matters

OpenEvolve needs a real quantitative evaluator. For our project, the evaluator is:

```text
fitness = 1 - MAE / 4
```

This lets an evolutionary coding agent modify the rating algorithm and keep versions that produce better predictions on the validation set.
