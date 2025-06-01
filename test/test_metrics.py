import pytest
from metrics import compute_metrics

def test_metrics():
    episode_data = {"rescued_victims": 2, "total_victims": 3, "steps": 50, "collisions": 1}
    metrics = compute_metrics(episode_data)
    assert metrics["rescue_success_rate"] == 2/3
    assert metrics["time_to_complete"] == 50
    assert metrics["collisions"] == 1