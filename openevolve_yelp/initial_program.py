from __future__ import annotations

from statistics import mean
from typing import Any


def _clip_rating(value: float) -> float:
    value = max(1.0, min(5.0, value))
    return round(value * 2) / 2


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# EVOLVE-BLOCK-START
def predict_stars(
    user: dict[str, Any],
    item: dict[str, Any],
    user_reviews: list[dict[str, Any]],
    item_reviews: list[dict[str, Any]],
) -> float:
    """
    Initial evolvable program for Yelp rating prediction.

    OpenEvolve should modify this function to reduce MAE on test_review_subset.json.
    Keep the function signature unchanged.
    """
    user_average = _safe_float(user.get("average_stars"), 3.5) or 3.5
    item_average = _safe_float(item.get("stars"), 3.5) or 3.5
    user_recent = [_safe_float(review.get("stars")) for review in user_reviews]
    item_recent = [_safe_float(review.get("stars")) for review in item_reviews]
    user_recent = [score for score in user_recent if score is not None]
    item_recent = [score for score in item_recent if score is not None]

    user_history_signal = mean(user_recent) if user_recent else user_average
    item_history_signal = mean(item_recent) if item_recent else item_average

    prediction = (
        0.35 * user_average
        + 0.25 * item_average
        + 0.20 * user_history_signal
        + 0.20 * item_history_signal
    )
    return _clip_rating(prediction)
# EVOLVE-BLOCK-END


def generate_review(
    user: dict[str, Any],
    item: dict[str, Any],
    predicted_stars: float,
) -> str:
    """Simple review generator paired with the rating predictor."""
    business_name = item.get("name") or "this business"
    categories = item.get("categories") or "its category"
    sentiment = "positive" if predicted_stars >= 4 else "mixed" if predicted_stars >= 3 else "disappointing"
    return (
        f"I would probably have a {sentiment} experience at {business_name}. "
        f"It fits into {categories}, and based on my usual rating style I would give it "
        f"around {predicted_stars:.1f} stars."
    )
