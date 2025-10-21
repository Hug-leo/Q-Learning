# Q-learning Agent

A simple Q-learning agent trained on Gymnasium's Blackjack-v1 environment.

## Setup

Install dependencies for your current Python interpreter:

```powershell
# From the repo root
python -m pip install --upgrade pip
python -m pip install -r .\requirements.txt
```

## Run training

By default it runs 100,000 episodes. You can lower it temporarily for quick tests.

```powershell
# Quick smoke test with 1,000 episodes
$env:N_EPISODES=1000; python .\Training_the_Agent.py
```

After training, a `training_stats.npz` file is created with rewards, episode lengths, and training errors.

### Rendering control

Rendering is disabled by default during training to keep it fast and avoid warnings from Gymnasium.

- To enable rendering during training (slower):

```powershell
$env:RENDER_TRAINING=1; python .\Training_the_Agent.py
```

- To enable rendering during evaluation only:

```powershell
$env:RENDER_EVAL=1; python .\Testing_Your_Trained_Agent.py
```

## Analyze results

Plot rolling averages from saved stats:

```powershell
python .\Analyzing_Training_Results.py
```

If your stats file is in a custom location/name:

```powershell
$env:STATS_PATH="D:\\AI\\Q_learning\\training_stats.npz"; python .\Analyzing_Training_Results.py
```

# Q-Learning
