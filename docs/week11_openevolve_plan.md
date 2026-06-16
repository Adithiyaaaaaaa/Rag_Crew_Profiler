# Week 11 OpenEvolve Plan

The Week 11 goal is to make the Yelp recommendation crew evolutionary.

## What We Can Evolve

The full CrewAI agent workflow is expensive to evaluate repeatedly. OpenEvolve works best when each candidate can be scored quantitatively and quickly.

So this repo evolves the deterministic rating strategy first:

```text
candidate program -> predict_stars(...) -> compare against ground truth -> MAE -> fitness
```

## Metric

Primary metric:

```text
MAE = mean(abs(predicted_stars - ground_truth_stars))
```

OpenEvolve score:

```text
combined_score = max(0, 1 - MAE / 4)
```

Higher is better.

## Files

```text
openevolve_yelp/initial_program.py
openevolve_yelp/evaluator.py
openevolve_yelp/config.yaml
```

`initial_program.py` follows the OpenEvolve requirement of exactly one evolvable block:

```python
# EVOLVE-BLOCK-START
def predict_stars(...):
    ...
# EVOLVE-BLOCK-END
```

Everything outside that block is helper code and should not be mutated.

## Local Test

```powershell
uv run python openevolve_yelp/evaluator.py openevolve_yelp/initial_program.py
```

## Relationship to CrewAI

The evolved rating strategy can later be plugged back into:

```text
src/first_crew/main.py
```

as the fast baseline scorer, while CrewAI remains responsible for explanation, review text generation, and simulator integration.

## Presentation Talking Point

We are not evolving prompts blindly. We created a measurable optimization loop:

```text
Generate candidate code -> run on validation reviews -> score with MAE -> keep better algorithms
```

This is the same idea behind AlphaEvolve/OpenEvolve, adapted to Yelp preference prediction.

## Config Notes

`openevolve_yelp/config.yaml` uses:

```yaml
database:
  feature_dimensions:
    - "score"
    - "complexity"
```

The feature dimensions are a list, as required by the updated OpenEvolve guide.
