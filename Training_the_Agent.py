import os
import gymnasium as gym
from tqdm import tqdm  # Progress bar
from Q_Learning_Agent_OOP import QAgent
import numpy as np
import pickle
import time


def main():
    # Training hyperparameters
    learning_rate = 0.01        # How fast to learn (higher = faster but less stable)
    n_episodes = int(os.getenv("N_EPISODES", 100_000))  # Number of hands to practice
    start_epsilon = 1.0         # Start with 100% random actions
    epsilon_decay = start_epsilon / (n_episodes / 2)  # Reduce exploration over time
    final_epsilon = 0.1         # Always keep some exploration

    # Rendering during training is optional (off by default to keep training fast)
    # Enable by setting RENDER_TRAINING=1 (uses text-based "human" rendering for Blackjack)
    render_training = os.getenv("RENDER_TRAINING", "0").lower() in {"1", "true", "yes", "on"}
    render_delay = float(os.getenv("RENDER_DELAY", "0.05"))  # seconds to pause after each render

    # Create environment and agent
    if render_training:
        env = gym.make("FrozenLake-v1", render_mode="human")
    else:
        env = gym.make("FrozenLake-v1")
    env = gym.wrappers.RecordEpisodeStatistics(env)

    agent = QAgent(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
    )

    for episode in tqdm(range(n_episodes)):
        # Start a new hand
        obs, info = env.reset()
        done = False

        # Play one complete hand
        while not done:
            # Agent chooses action (initially random, gradually more intelligent)
            action = agent.get_action(obs)

            # Take action and observe result
            next_obs, reward, terminated, truncated, info = env.step(action)

            # Learn from this experience
            agent.update(obs, action, reward, terminated, next_obs)

            # Move to next state
            done = terminated or truncated
            obs = next_obs
            # Only render if explicitly enabled; avoids Gym warning and speeds up training
            if render_training:
                env.render()
                time.sleep(render_delay)
        # Reduce exploration rate (agent becomes less random over time)
        agent.decay_epsilon()

    # Save training statistics for offline analysis
    try:
        rewards = list(env.return_queue)
        lengths = list(env.length_queue)
        np.savez(
            "training_stats.npz",
            rewards=np.array(rewards, dtype=float),
            lengths=np.array(lengths, dtype=float),
            training_error=np.array(agent.training_error, dtype=float),
            n_episodes=np.array([n_episodes], dtype=int),
        )
    except Exception as e:
        # Non-fatal; analysis script can still be run if user captures stats another way
        print(f"Warning: Failed to save training stats: {e}")

    # Save learned policy (Q-table) for evaluation
    try:
        q_serializable = {state: np.asarray(values, dtype=float) for state, values in agent.q_values.items()}
        payload = {
            "q_values": q_serializable,
            "action_space_n": env.action_space.n,
            "metadata": {
                "discount_factor": agent.discount_factor,
                "learning_rate": agent.lr,
                "episodes": n_episodes,
            },
        }
        with open("policy.pkl", "wb") as f:
            pickle.dump(payload, f)
    except Exception as e:
        print(f"Warning: Failed to save policy: {e}")


if __name__ == "__main__":
    main()