from collections import defaultdict
import gymnasium as gym
import numpy as np


class DiscreteQAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
        n_bins: int = 10,
    ):
        """Initialize a Discrete-Q-Learning agent with state discretization for continuous environments.

        Args:
            env: The training environment
            learning_rate: How quickly to update Q-values (0-1)
            initial_epsilon: Starting exploration rate (usually 1.0)
            epsilon_decay: How much to reduce epsilon each episode
            final_epsilon: Minimum exploration rate (usually 0.1)
            discount_factor: How much to value future rewards (0-1)
            n_bins: Number of bins for discretizing continuous state dimensions
        """
        self.env = env

        # Q-table: maps (state, action) to expected reward
        # defaultdict automatically creates entries with zeros for new states
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor  # How much we care about future rewards

        # Exploration parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Track learning progress
        self.training_error = []

        # State discretization for continuous environments (like CartPole)
        self.n_bins = n_bins
        self._setup_discretization()

    def _setup_discretization(self):
        """Set up bins for discretizing continuous state spaces."""
        # Check if we have a continuous observation space
        if isinstance(self.env.observation_space, gym.spaces.Box):
            # Create bins for each dimension of the state space
            self.state_bins = []
            for i in range(len(self.env.observation_space.low)):
                low = self.env.observation_space.low[i]
                high = self.env.observation_space.high[i]
                
                # Handle infinite bounds (common in CartPole)
                if np.isinf(low):
                    low = -10.0
                if np.isinf(high):
                    high = 10.0
                
                # Create bins for this dimension
                bins = np.linspace(low, high, self.n_bins - 1)
                self.state_bins.append(bins)
        else:
            self.state_bins = None

    def _discretize_state(self, obs):
        """Convert continuous observation to discrete state tuple."""
        if self.state_bins is None:
            # Already discrete (like Blackjack) - return as tuple if array
            return tuple(obs) if isinstance(obs, np.ndarray) else obs
        
        # Discretize each dimension
        discrete_state = []
        for i, value in enumerate(obs):
            # Find which bin this value falls into
            bin_index = np.digitize(value, self.state_bins[i])
            discrete_state.append(bin_index)
        
        return tuple(discrete_state)

    def get_action(self, obs) -> int:
        """Choose an action using epsilon-greedy strategy.

        Returns:
            action: The selected action index
        """
        # Discretize the observation
        discrete_obs = self._discretize_state(obs)
        
        # With probability epsilon: explore (random action)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        # With probability (1-epsilon): exploit (best known action)
        else:
            return int(np.argmax(self.q_values[discrete_obs]))

    def update(
        self,
        obs,
        action: int,
        reward: float,
        terminated: bool,
        next_obs,
    ):
        """Update Q-value based on experience.

        This is the heart of Q-learning: learn from (state, action, reward, next_state)
        """
        # Discretize observations
        discrete_obs = self._discretize_state(obs)
        discrete_next_obs = self._discretize_state(next_obs)
        
        # What's the best we could do from the next state?
        # (Zero if episode terminated - no future rewards possible)
        future_q_value = (not terminated) * np.max(self.q_values[discrete_next_obs])

        # What should the Q-value be? (Bellman equation)
        target = reward + self.discount_factor * future_q_value

        # How wrong was our current estimate?
        temporal_difference = target - self.q_values[discrete_obs][action]

        # Update our estimate in the direction of the error
        # Learning rate controls how big steps we take
        self.q_values[discrete_obs][action] = (
            self.q_values[discrete_obs][action] + self.lr * temporal_difference
        )

        # Track learning progress (useful for debugging)
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)