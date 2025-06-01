def compute_metrics(episode_data):
    rescue_success = episode_data["rescued_victims"] / episode_data["total_victims"]
    time_to_complete = episode_data["steps"]
    collisions = episode_data["collisions"]
    return {
        "rescue_success_rate": rescue_success,
        "time_to_complete": time_to_complete,
        "collisions": collisions
    }