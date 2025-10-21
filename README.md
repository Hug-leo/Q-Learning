# Q-Learning with Gymnasium Environments

A flexible reinforcement learning framework implementing both **table-based Q-learning** and **discretized Q-learning** for various Gymnasium environments. This project demonstrates core RL concepts with clear, educational code.

## 🎯 Features

- **Multiple Agent Types**:
  - `QAgent`: Pure Q-learning for discrete state spaces (FrozenLake, Blackjack, Taxi)
  - `DiscreteQAgent`: Q-learning with state discretization for continuous environments (CartPole, MountainCar)
  
- **Supported Environments**:
  - ✅ FrozenLake-v1 (discrete, grid world)
  - ✅ Blackjack-v1 (discrete, card game)
  - ✅ CartPole-v1 (continuous → discretized)
  - ✅ MountainCar-v0 (continuous → discretized)
  - ✅ Taxi-v3 (discrete, navigation)

- **Training Features**:
  - ε-greedy exploration with decay
  - Configurable hyperparameters via environment variables
  - Training statistics tracking (rewards, episode lengths, TD errors)
  - Policy saving/loading for evaluation
  - Optional real-time rendering with adjustable speed

- **Analysis Tools**:
  - Automated plotting of training progress
  - Rolling averages for reward, episode length, and TD error
  - Performance evaluation metrics

## 📋 Requirements

- Python 3.10+
- Gymnasium
- NumPy
- Matplotlib
- Pygame (for rendering)
- tqdm (progress bars)

## 🚀 Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/Q_learning.git
cd Q_learning
```

2. Create a virtual environment (recommended):
```bash
python -m venv .venv
```

3. Activate the virtual environment:
```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Windows Command Prompt
.venv\Scripts\activate.bat

# Linux/Mac
source .venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎮 Usage

### Training an Agent

By default, trains on **FrozenLake-v1** for 100,000 episodes:

```powershell
python .\Training_the_Agent.py
```

#### Environment Variables

Customize training with these environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `N_EPISODES` | 100000 | Number of training episodes |
| `RENDER_TRAINING` | 0 | Enable visual rendering during training (1/true/yes/on) |
| `RENDER_DELAY` | 0.05 | Seconds to pause after each render (makes animation visible) |

**Examples:**

```powershell
# Quick test (1,000 episodes)
$env:N_EPISODES=1000; python .\Training_the_Agent.py

# Train with visible rendering (slow, for demonstration)
$env:RENDER_TRAINING=1; $env:N_EPISODES=10; $env:RENDER_DELAY=0.2; python .\Training_the_Agent.py

# Full training without rendering (fastest)
python .\Training_the_Agent.py
```

### Testing a Trained Agent

After training, evaluate your agent's performance:

```powershell
python .\Testing_Your_Trained_Agent.py
```

**With rendering** (watch your agent play):

```powershell
$env:RENDER_EVAL=1; python .\Testing_Your_Trained_Agent.py
```

**Custom test settings:**

```powershell
# Test with 100 episodes
$env:TEST_EPISODES=100; python .\Testing_Your_Trained_Agent.py

# Use a different policy file
$env:POLICY_PATH="my_policy.pkl"; python .\Testing_Your_Trained_Agent.py
```

### Analyzing Training Results

Visualize training progress with matplotlib:

```powershell
python .\Analyzing_Training_Results.py
```

This generates plots showing:
- Episode rewards (performance over time)
- Episode lengths (how long each episode lasted)
- Training error (temporal difference errors)

**Custom stats file:**

```powershell
$env:STATS_PATH="custom_stats.npz"; python .\Analyzing_Training_Results.py
```

### Quick Demo (No Training Required)

Watch a random agent play CartPole:

```powershell
python .\test_rendering.py
```

## 🧩 Project Structure

```
Q_learning/
├── Q_Learning_Agent_OOP.py          # Pure Q-learning agent (discrete states)
├── Discrete_Q_Learning_Agent_OOP.py # Q-learning with state discretization
├── Training_the_Agent.py            # Main training script
├── Testing_Your_Trained_Agent.py    # Evaluation script
├── Analyzing_Training_Results.py    # Training visualization
├── test_rendering.py                # Quick rendering demo
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── policy.pkl                       # Saved Q-table (after training)
└── training_stats.npz               # Training statistics (after training)
```

## 🔬 Switching Environments

To train on a different environment, modify `Training_the_Agent.py`:

### For Discrete Environments (FrozenLake, Blackjack, Taxi)

Use `QAgent`:

```python
from Q_Learning_Agent_OOP import QAgent

# In main():
env = gym.make("FrozenLake-v1")  # or "Blackjack-v1", "Taxi-v3"
agent = QAgent(
    env=env,
    learning_rate=0.01,
    initial_epsilon=1.0,
    epsilon_decay=epsilon_decay,
    final_epsilon=0.1,
)
```

### For Continuous Environments (CartPole, MountainCar)

Use `DiscreteQAgent`:

```python
from Discrete_Q_Learning_Agent_OOP import DiscreteQAgent

# In main():
env = gym.make("CartPole-v1")  # or "MountainCar-v0"
agent = DiscreteQAgent(
    env=env,
    learning_rate=0.01,
    initial_epsilon=1.0,
    epsilon_decay=epsilon_decay,
    final_epsilon=0.1,
    n_bins=10,  # Number of bins per dimension for discretization
)
```

**Don't forget to update `Testing_Your_Trained_Agent.py` to match!**

## 📊 How It Works

### Q-Learning Algorithm

This project implements the classic Q-learning algorithm:

```
Q(s, a) ← Q(s, a) + α [r + γ max Q(s', a') - Q(s, a)]
```

Where:
- `Q(s, a)`: Quality of taking action `a` in state `s`
- `α`: Learning rate (how quickly to update)
- `r`: Reward received
- `γ`: Discount factor (how much to value future rewards)
- `s'`: Next state

### State Discretization

For continuous environments, `DiscreteQAgent` discretizes the state space:

1. Divides each continuous dimension into `n_bins` discrete buckets
2. Maps continuous values to bin indices using `np.digitize()`
3. Creates hashable tuples for Q-table lookup

**Example:** CartPole state `[0.02, 0.15, -0.03, -0.21]` → discrete state `(5, 6, 4, 3)`

### Exploration vs Exploitation (ε-greedy)

- **Exploration**: Random action with probability ε
- **Exploitation**: Best known action with probability (1 - ε)
- ε decays linearly from 1.0 → 0.1 over training

## 🎓 Educational Notes

### When to Use Each Agent Type

| Agent Type | Best For | Examples |
|------------|----------|----------|
| **QAgent** | Small, discrete state spaces | FrozenLake, Blackjack, Taxi |
| **DiscreteQAgent** | Continuous states with few dimensions | CartPole, MountainCar |
| **Deep Q-Learning (DQN)** | Large/continuous state spaces, images | Atari, LunarLander, robotics |

### Performance Expectations

| Environment | Agent | Expected Performance |
|-------------|-------|---------------------|
| FrozenLake-v1 | QAgent | ~70-80% win rate |
| CartPole-v1 | DiscreteQAgent | 100-200 avg reward |
| MountainCar-v0 | DiscreteQAgent | Reaches goal in 150-200 steps |
| Blackjack-v1 | QAgent | ~42-45% win rate |

### Limitations

- **Table-based Q-learning** doesn't scale to high-dimensional or very large state spaces
- **State discretization** loses information and can hurt performance
- For complex environments (Atari, robotics), use **Deep Q-Networks (DQN)** instead

## 🛠️ Troubleshooting

### "ModuleNotFoundError: No module named 'pygame'"

Install pygame:
```bash
pip install pygame
```

### "TypeError: unhashable type: 'numpy.ndarray'"

You're using `QAgent` on a continuous environment. Switch to `DiscreteQAgent`:
```python
from Discrete_Q_Learning_Agent_OOP import DiscreteQAgent
```

### Rendering window appears but closes instantly

Add a delay:
```powershell
$env:RENDER_DELAY=0.2; $env:RENDER_TRAINING=1; python .\Training_the_Agent.py
```

### Training is too slow with rendering

Disable rendering or reduce episodes:
```powershell
$env:RENDER_TRAINING=0; $env:N_EPISODES=10000; python .\Training_the_Agent.py
```

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Add support for new environments
- Improve the agents (e.g., Double Q-Learning, prioritized replay)
- Add more analysis tools
- Improve documentation

## 📚 References

- [Sutton & Barto: Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Q-Learning Tutorial](https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/)

## 🙋 FAQ

**Q: Why are my rewards so low?**  
A: Q-learning needs many episodes to converge. Try training for 100,000+ episodes. Also check your hyperparameters (learning rate, epsilon decay).

**Q: Can I use this for Atari games?**  
A: No, Atari requires Deep Q-Learning (DQN) with convolutional neural networks. This project uses table-based Q-learning.

**Q: How do I save my trained agent?**  
A: It's automatic! After training, `policy.pkl` contains your Q-table. Load it with `Testing_Your_Trained_Agent.py`.

**Q: What's the difference between QAgent and DiscreteQAgent?**  
A: `QAgent` works only with discrete states (like FrozenLake). `DiscreteQAgent` can handle continuous states by discretizing them into bins (like CartPole).

---
**Result for CartPole-v1 (continuous → discretized)

<img width="597" height="425" alt="image" src="https://github.com/user-attachments/assets/5f3e89a9-f03f-4227-892a-f247afc6c9bb" />

**Result for FrozenLake-v1 (discrete, grid world)

<img width="254" height="285" alt="image" src="https://github.com/user-attachments/assets/6bc3aef1-83cf-4bde-9057-1c4a8e7ed608" />

**Happy Learning! 🎉**
