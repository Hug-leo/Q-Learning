import numpy as np
from matplotlib import pyplot as plt
import os

def get_moving_avgs(arr, window, convolution_mode):
    """Compute moving average to smooth noisy data."""
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    ) / window

# Load saved training statistics
stats_path = os.getenv("STATS_PATH", "training_stats.npz")
if not os.path.exists(stats_path):
    raise FileNotFoundError(
        f"Could not find '{stats_path}'. Run Training_the_Agent.py first to generate it, or set STATS_PATH to the npz file."
    )

data = np.load(stats_path)
rewards = data.get("rewards")
lengths = data.get("lengths")
training_error = data.get("training_error")

# Smooth over a 500-episode window (cap to series length)
rolling_length = min(500, len(rewards)) if rewards is not None else 500
fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

# Episode rewards (win/loss performance)
axs[0].set_title("Episode rewards")
if rewards is not None and len(rewards) >= 2:
    reward_moving_average = get_moving_avgs(rewards, rolling_length, "valid")
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
axs[0].set_ylabel("Average Reward")
axs[0].set_xlabel("Episode")

# Episode lengths (how many actions per hand)
axs[1].set_title("Episode lengths")
if lengths is not None and len(lengths) >= 2:
    length_moving_average = get_moving_avgs(lengths, rolling_length, "valid")
    axs[1].plot(range(len(length_moving_average)), length_moving_average)
axs[1].set_ylabel("Average Episode Length")
axs[1].set_xlabel("Episode")

# Training error (how much we're still learning)
axs[2].set_title("Training Error")
if training_error is not None and len(training_error) > 0:
    training_error_moving_average = get_moving_avgs(training_error, min(rolling_length, len(training_error)), "same")
    axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
axs[2].set_ylabel("Temporal Difference Error")
axs[2].set_xlabel("Step")

plt.tight_layout()
plt.show()