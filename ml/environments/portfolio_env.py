"""Weekly portfolio allocation environment for RL training."""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Optional
import warnings

warnings.filterwarnings("ignore")


class WeeklyPortfolioEnv(gym.Env):
    """
    Weekly rebalancing environment for portfolio allocation with market regimes.
    
    State: [continuous_features, regime_posteriors, prev_allocation]
    Actions: 7 discrete portfolio templates
        0: 100% Cash
        1: 100% SPY
        2: 100% TLT
        3: 100% GLD
        4: 80% SPY / 20% TLT
        5: 60% SPY / 30% TLT / 10% GLD
        6: 20% SPY / 60% TLT / 20% GLD
    
    Reward: Return - cost*Turnover + incentive*Turnover - vol_penalty*RollingVol
    """
    
    # Portfolio action templates: {action_id: (SPY, TLT, GLD, CASH)}
    PORTFOLIO_TEMPLATES = {
        0: (0.0, 0.0, 0.0, 1.0),    # 100% Cash
        1: (1.0, 0.0, 0.0, 0.0),    # 100% SPY
        2: (0.0, 1.0, 0.0, 0.0),    # 100% TLT
        3: (0.0, 0.0, 1.0, 0.0),    # 100% GLD
        4: (0.8, 0.2, 0.0, 0.0),    # 80 SPY / 20 TLT
        5: (0.6, 0.3, 0.1, 0.0),    # 60 SPY / 30 TLT / 10 GLD
        6: (0.2, 0.6, 0.2, 0.0),    # 20% SPY / 60% TLT / 20% GLD
    }
    
    ACTION_NAMES = [
        "cash_only",
        "spy_only",
        "tlt_only",
        "gld_only",
        "spy_80_tlt_20",
        "balanced_60_30_10",
        "defensive_20_60_20",
    ]
    
    def __init__(self,
                 features: pd.DataFrame,
                 regime_posteriors: np.ndarray,
                 asset_returns: pd.DataFrame,
                 transaction_cost: float = 0.001,
                 turnover_incentive: float = 0.002,
                 volatility_penalty: float = 0.05,
                 lookback_vol: int = 4,
                 seq_len: int = 4,
                 start_step: Optional[int] = None,
                 end_step: Optional[int] = None,
                 initial_allocation: Optional[np.ndarray] = None):
        """
        Initialize portfolio environment.
        
        Args:
            features: Weekly feature DataFrame [n_weeks, n_features]
            regime_posteriors: Weekly regime probabilities [n_weeks, n_regimes]
            asset_returns: Weekly returns for portfolio assets [n_weeks, 4] (SPY, TLT, GLD, Cash)
            transaction_cost: Cost per unit of portfolio turnover
            turnover_incentive: Positive reward per unit of portfolio turnover
            volatility_penalty: Penalty for portfolio volatility
            lookback_vol: Lookback window for rolling volatility (weeks)
            seq_len: Sequence length for temporal attention
            start_step: First internal environment step. Reward is realized on row `start_step - 1`
            end_step: Exclusive environment end step. Reward is realized through row `end_step - 1`
            initial_allocation: Starting portfolio weights in SPY/TLT/GLD/CASH order
        """
        super().__init__()
        
        self.features = features.values  # [n_weeks, n_features]
        self.regime_posteriors = regime_posteriors  # [n_weeks, n_regimes]
        self.asset_returns = asset_returns.values  # [n_weeks, 4]
        
        self.transaction_cost = transaction_cost
        self.turnover_incentive = turnover_incentive
        self.volatility_penalty = volatility_penalty
        self.lookback_vol = lookback_vol
        self.seq_len = seq_len
        self.initial_allocation = np.array(
            initial_allocation if initial_allocation is not None else [0.0, 0.0, 0.0, 1.0],
            dtype=float,
        )
        
        # Dimensions
        self.n_features = features.shape[1]
        self.n_regimes = regime_posteriors.shape[1]
        self.state_dim = self.n_features + self.n_regimes + 4  # features + regime + prev_allocation
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(len(self.PORTFOLIO_TEMPLATES))
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.seq_len, self.state_dim),
            dtype=np.float32
        )
        
        # Environment state. `current_step` points one past the most recent
        # observable row, so reward is realized on `current_step - 1`.
        default_start = max(1, lookback_vol + seq_len) if start_step is None else int(start_step)
        default_end = len(features) + 1 if end_step is None else int(end_step)
        if default_start < 1:
            raise ValueError("start_step must be >= 1 so the first reward maps to a valid row.")
        if default_end > len(features) + 1:
            raise ValueError("end_step cannot exceed one step past the number of feature rows.")
        if default_start > default_end:
            raise ValueError("start_step must be <= end_step.")

        self.default_start_step = default_start
        self.current_step = self.default_start_step
        self.max_step = default_end
        self.prev_allocation = self.initial_allocation.copy()
        self.portfolio_returns = []
        self.actions_taken = []
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment and return initial state.
        
        Args:
            seed: Random seed (for gymnasium compatibility)
            options: Additional options (for gymnasium compatibility)
            
        Returns:
            observation: Initial state observation
            info: Info dictionary
        """
        # Set seed if provided (gymnasium requirement)
        if seed is not None:
            np.random.seed(seed)
            
        self.current_step = self.default_start_step
        self.prev_allocation = self.initial_allocation.copy()
        self.portfolio_returns = []
        self.actions_taken = []
        
        return self._get_observation(), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step of the environment.
        
        Args:
            action: Action ID (0-6)
            
        Returns:
            observation: New state observation
            reward: Reward for this step
            terminated: Whether episode ended (goal reached) - always False for portfolio env
            truncated: Whether episode ended (max steps reached)
            info: Additional info dict
        """
        # Convert action to int if it's a numpy array (from stable-baselines3 DummyVecEnv)
        if isinstance(action, np.ndarray):
            action = int(action.item() if action.ndim == 0 else action[0])
        else:
            action = int(action)
        
        decision_step = self.current_step

        # Get new allocation from action
        new_allocation = np.array(self.PORTFOLIO_TEMPLATES[action])
        
        # Compute reward components
        # 1. Portfolio return this week (credit current action directly)
        portfolio_return = np.dot(new_allocation, self.asset_returns[decision_step - 1])
        
        # 2. Turnover terms
        turnover = np.sum(np.abs(new_allocation - self.prev_allocation)) / 2
        turnover_cost = self.transaction_cost * turnover
        turnover_reward = self.turnover_incentive * turnover
        
        # 3. Volatility penalty (rolling volatility)
        vol_lookback = self.lookback_vol
        if decision_step - vol_lookback >= 0:
            recent_returns = self.asset_returns[decision_step - vol_lookback:decision_step]
            portfolio_vols = np.std(recent_returns, axis=0)
            portfolio_vol = np.dot(new_allocation ** 2, portfolio_vols ** 2) ** 0.5
        else:
            portfolio_vol = 0.0
            
        volatility_penalty = self.volatility_penalty * portfolio_vol
        
        # Total reward
        reward = portfolio_return - turnover_cost + turnover_reward - volatility_penalty
        
        # Update state
        self.prev_allocation = new_allocation
        self.portfolio_returns.append(portfolio_return)
        self.actions_taken.append(action)
        self.current_step += 1
        
        # Check if done (gymnasium uses terminated and truncated)
        truncated = self.current_step >= self.max_step
        terminated = False  # Portfolio env never reaches a goal, only runs out of data
        
        # Get next observation
        observation = self._get_observation()
        
        # Info dict
        info = {
            "portfolio_return": portfolio_return,
            "turnover": turnover,
            "turnover_cost": turnover_cost,
            "turnover_reward": turnover_reward,
            "turnover_net": turnover_reward - turnover_cost,
            "volatility": portfolio_vol,
            "allocation": new_allocation,
            "action_name": self.ACTION_NAMES[action],
            "week": decision_step - 1,
        }
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Build observation from current state.
        
        Observation sequence: last seq_len timesteps.
        For each timestep: [market_features, regime_posteriors, prev_allocation]
        """
        obs_list = []
        
        start_idx = max(0, self.current_step - self.seq_len)
        end_idx = self.current_step
        
        for t in range(start_idx, end_idx):
            if t < 0 or t >= len(self.features):
                # Pad with zeros if out of bounds
                step_obs = np.zeros(self.state_dim, dtype=np.float32)
            else:
                # [features, regime_posteriors, prev_allocation]
                step_obs = np.concatenate([
                    self.features[t],
                    self.regime_posteriors[t],
                    self.prev_allocation
                ]).astype(np.float32)
            
            obs_list.append(step_obs)
        
        # Pad if not enough history
        while len(obs_list) < self.seq_len:
            obs_list.insert(0, np.zeros(self.state_dim, dtype=np.float32))
        
        observation = np.array(obs_list[-self.seq_len:], dtype=np.float32)
        return observation
    
    def get_episode_stats(self) -> Dict:
        """Get statistics for current episode."""
        if len(self.portfolio_returns) == 0:
            return {}
        
        returns = np.array(self.portfolio_returns)
        total_return = np.sum(returns)
        cumulative_return = np.prod(1 + returns) - 1
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe = avg_return / (std_return + 1e-8) * np.sqrt(52)  # Annualized
        max_drawdown = self._compute_max_drawdown(returns)
        
        return {
            "total_return": total_return,
            "cumulative_return": cumulative_return,
            "avg_return": avg_return,
            "std_return": std_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "num_episodes": len(self.portfolio_returns),
        }
    
    @staticmethod
    def _compute_max_drawdown(returns: np.ndarray) -> float:
        """Compute maximum drawdown from returns."""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / (running_max + 1e-8)
        return np.min(drawdown)
    
    def render(self, mode: str = "human"):
        """Render environment state."""
        if mode == "human":
            print(f"Step: {self.current_step}/{self.max_step}")
            print(f"Allocation: {self.prev_allocation}")
            if self.portfolio_returns:
                print(f"Last return: {self.portfolio_returns[-1]:.4f}")
