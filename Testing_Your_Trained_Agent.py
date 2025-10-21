import os
import pickle
import numpy as np
import gymnasium as gym
from Discrete_Q_Learning_Agent_OOP import DeepQAgent


# Test the trained agent
def test_agent(agent, env, num_episodes=1000):
    """Test agent performance without learning or exploration."""
    total_rewards = []

    # Temporarily disable exploration for testing
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0  # Pure exploitation

    for _ in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        total_rewards.append(episode_reward)

    # Restore original epsilon
    agent.epsilon = old_epsilon

    win_rate = np.mean(np.array(total_rewards) > 0)
    average_reward = np.mean(total_rewards)

    print(f"Test Results over {num_episodes} episodes:")
    print(f"Win Rate: {win_rate:.1%}")
    print(f"Average Reward: {average_reward:.3f}")
    print(f"Standard Deviation: {np.std(total_rewards):.3f}")

def load_policy(path="policy.pkl"):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find policy file at '{path}'. Run Training_the_Agent.py first to create it."
        )
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


if __name__ == "__main__":
    # Create evaluation environment
    render_eval = os.getenv("RENDER_EVAL", "0").lower() in {"1", "true", "yes", "on"}
    if render_eval:
        env = gym.make("CartPole-v1", render_mode="human")
    else:
        env = gym.make("CartPole-v1")

    # Construct agent (with any hyperparams; we'll load Q-table next)
    agent = DeepQAgent(
        env=env,
        learning_rate=0.0,    # no learning during evaluation
        initial_epsilon=0.0,   # we'll set epsilon=0 for test anyway
        epsilon_decay=0.0,
        final_epsilon=0.0,
        discount_factor=0.95,
    )

    # Load saved policy
    policy_path = os.getenv("POLICY_PATH", "policy.pkl")
    payload = load_policy(policy_path)

    # Restore Q-table
    q_loaded = payload.get("q_values", {})
    # Ensure numpy arrays
    agent.q_values.clear()
    for state, values in q_loaded.items():
        agent.q_values[state] = np.asarray(values, dtype=float)

    # Evaluate with greedy policy
    episodes = int(os.getenv("TEST_EPISODES", 1000))
    test_agent(agent, env, num_episodes=episodes)