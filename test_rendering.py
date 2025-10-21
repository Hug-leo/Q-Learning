"""Quick test to verify CartPole rendering works."""
import gymnasium as gym

# Create environment with rendering enabled
env = gym.make("CartPole-v1", render_mode="human")

print("Starting CartPole rendering test...")
print("You should see a window with a cart and pole!")
print("Press Ctrl+C to stop.\n")

# Run a few episodes
for episode in range(3):
    obs, info = env.reset()
    total_reward = 0
    step = 0
    done = False
    
    print(f"Episode {episode + 1}:")
    while not done:
        # Take random actions
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        done = terminated or truncated
        
        # Render the environment
        env.render()
    
    print(f"  Finished after {step} steps, total reward: {total_reward}\n")

env.close()
print("Test complete! If you saw the CartPole animation, rendering is working.")
