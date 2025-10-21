# Quick Start Guide

Get up and running in 5 minutes!

## 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Q_learning.git
cd Q_learning

# Create virtual environment
python -m venv .venv

# Activate it (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

## 2. Train Your First Agent

```powershell
# Quick 1,000 episode test on FrozenLake
$env:N_EPISODES=1000; python .\Training_the_Agent.py
```

This will:
- Train a Q-learning agent on FrozenLake-v1
- Save the trained policy to `policy.pkl`
- Save training statistics to `training_stats.npz`
- Take about 10-30 seconds

## 3. Test Your Agent

```powershell
# Evaluate performance
python .\Testing_Your_Trained_Agent.py
```

You'll see output like:
```
Test Results over 1000 episodes:
Win Rate: 74.2%
Average Reward: 0.742
Standard Deviation: 0.438
```

## 4. Visualize Training

```powershell
# Plot training progress
python .\Analyzing_Training_Results.py
```

A matplotlib window will show:
- Episode rewards over time
- Episode lengths
- Training error (TD error)

## 5. Watch Your Agent Play (Optional)

```powershell
# Re-run evaluation with rendering
$env:RENDER_EVAL=1; python .\Testing_Your_Trained_Agent.py
```

You'll see the agent navigate the frozen lake in real-time!

## Next Steps

### Try Different Environments

**CartPole (continuous state, needs discretization):**

1. Edit `Training_the_Agent.py`:
   ```python
   from Discrete_Q_Learning_Agent_OOP import DiscreteQAgent
   
   # Change environment
   env = gym.make("CartPole-v1")
   
   # Change agent
   agent = DiscreteQAgent(
       env=env,
       learning_rate=0.01,
       initial_epsilon=1.0,
       epsilon_decay=epsilon_decay,
       final_epsilon=0.1,
       n_bins=10,
   )
   ```

2. Update `Testing_Your_Trained_Agent.py` similarly

3. Train and test!

### Experiment with Hyperparameters

Try adjusting:
- `learning_rate`: 0.001 - 0.1 (how fast to learn)
- `n_episodes`: 10,000 - 1,000,000 (more = better learning)
- `epsilon_decay`: Control exploration rate
- `n_bins`: 5-20 (for discretized agents, affects granularity)

### Full Training

```powershell
# Full 100,000 episode training (3-5 minutes)
python .\Training_the_Agent.py
```

## Troubleshooting

**"ModuleNotFoundError: No module named 'pygame'"**
```bash
pip install pygame
```

**Agent performs poorly**
- Train for more episodes
- Try different learning rates
- For continuous environments, increase `n_bins`

**Rendering doesn't show**
- Make sure you set `RENDER_EVAL=1`
- Check that pygame is installed
- Add delay: `$env:RENDER_DELAY=0.2`

## Environment Cheat Sheet

| Environment | Agent Type | Training Time | Difficulty |
|-------------|-----------|---------------|-----------|
| FrozenLake-v1 | QAgent | Fast (~30s) | Easy |
| Blackjack-v1 | QAgent | Fast (~1min) | Medium |
| CartPole-v1 | DiscreteQAgent | Medium (~5min) | Medium |
| MountainCar-v0 | DiscreteQAgent | Slow (~10min) | Hard |
| Taxi-v3 | QAgent | Fast (~1min) | Easy |

**Happy Learning! 🚀**
