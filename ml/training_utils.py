"""Training utilities using FinRL (stable-baselines3) for DQN agent."""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class ValidationCallback(BaseCallback):
    """Callback for validation-based early stopping during training."""
    
    def __init__(self, val_env: Any, eval_freq: int = 10, patience: int = 20, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.val_env = val_env
        self.eval_freq = eval_freq
        self.patience = patience
        self.best_val_reward = -np.inf
        self.patience_counter = 0
        self.val_history: List[Dict[str, float]] = []
        self.n_calls = 0
        
    def _on_step(self) -> bool:
        """Called after each step in the environment."""
        if self.n_calls % self.eval_freq == 0:
            # Run validation episode
            val_reward = self._evaluate_policy()
            self.val_history.append({
                'step': self.num_timesteps,
                'reward': val_reward
            })
            
            if self.verbose > 0:
                print(f"Validation at step {self.num_timesteps}: reward={val_reward:.4f}")
            
            # Early stopping check
            if val_reward > self.best_val_reward:
                self.best_val_reward = val_reward
                self.patience_counter = 0
                if self.verbose > 0:
                    print(f"  -> New best validation reward!")
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= self.patience:
                if self.verbose > 0:
                    print(f"Early stopping triggered at step {self.num_timesteps}")
                return False
        
        return True
    
    def _evaluate_policy(self, num_episodes: int = 1) -> float:
        """Evaluate policy on validation environment."""
        rewards: List[float] = []
        for _ in range(num_episodes):
            obs, _ = self.val_env.reset()  # gymnasium returns (obs, info) tuple
            done = False
            episode_reward = 0.0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.val_env.step(action)  # gymnasium returns 5 values
                episode_reward += reward
                done = terminated or truncated
            
            rewards.append(episode_reward)
        
        return np.mean(rewards)


def _as_int_action(action: Any) -> int:
    """Convert policy output to a scalar int action."""
    if isinstance(action, np.ndarray):
        return int(action.item() if action.ndim == 0 else action[0])
    return int(action)


def evaluate_episode(agent: Any,
                     env: Any,
                     max_steps: Optional[int] = None,
                     deterministic: bool = True) -> Dict[str, Any]:
    """
    Evaluate agent for one episode (no training, deterministic).
    
    Args:
        agent: FinRL DQN agent (stable-baselines3)
        env: Portfolio environment
        max_steps: Maximum steps per episode
        deterministic: Use deterministic policy (greedy)
        
    Returns:
        Episode statistics dict
    """
    obs, _ = env.reset()  # gymnasium returns (obs, info) tuple
    episode_reward = 0.0
    episode_length = 0
    actions_taken: List[Dict[str, Any]] = []
    
    while True:
        # Select action (deterministic/greedy)
        action, _ = agent.predict(obs, deterministic=deterministic)
        next_obs, reward, terminated, truncated, info = env.step(action)  # gymnasium returns 5 values
        action_id = _as_int_action(action)
        
        actions_taken.append({
            'action': action_id,
            'action_name': info.get('action_name', f'Action {action_id}'),
            'return': info.get('portfolio_return', reward),
            'turnover': info.get('turnover', 0.0),
        })
        
        obs = next_obs
        episode_reward += reward
        episode_length += 1
        
        done = terminated or truncated
        if done or (max_steps and episode_length >= max_steps):
            break
    
    # Get environment stats if available
    env_stats = env.get_episode_stats() if hasattr(env, 'get_episode_stats') else {}
    
    return {
        'reward': episode_reward,
        'length': episode_length,
        'avg_reward': episode_reward / episode_length if episode_length > 0 else 0,
        'actions': actions_taken,
        **env_stats,
    }


def train_dqn_finrl(train_env: Any,
                    val_env: Any,
                    total_timesteps: int = 100000000,
                    eval_freq: int = 2000,
                    early_stopping_patience: int = 10,
                    learning_rate: float = 1e-4,
                    exploration_fraction: float = 0.1,
                    exploration_final_eps: float = 0.05,
                    target_update_interval: int = 1000,
                    buffer_size: int = 10000,
                    batch_size: int = 32,
                    device: str = 'auto') -> Dict[str, Any]:
    """
    Train DQN agent using FinRL (stable-baselines3) with validation and early stopping.
    
    Args:
        train_env: Training environment
        val_env: Validation environment
        total_timesteps: Total training timesteps
        eval_freq: Evaluate every N timesteps
        early_stopping_patience: Early stopping patience (in evaluations)
        learning_rate: Learning rate for optimizer
        exploration_fraction: Fraction of training steps for exploration decay
        exploration_final_eps: Final epsilon for exploration
        target_update_interval: Update target network every N steps
        buffer_size: Replay buffer capacity
        batch_size: Batch size for training
        device: Device for computation ('auto', 'cuda', 'cpu')
        
    Returns:
        Training results dict with history and trained agent
    """
    
    # Initialize DQN agent from stable-baselines3
    agent = DQN(
        'MlpPolicy',
        train_env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=1000,
        batch_size=batch_size,
        tau=1.0,
        gamma=0.99,
        train_freq=1,
        target_update_interval=target_update_interval,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=1.0,
        exploration_final_eps=exploration_final_eps,
        device=device,
        verbose=1
    )
    
    print("=== Training DQN Agent with FinRL ===")
    print(f"Device: {agent.device}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Evaluation frequency: {eval_freq}")
    print(f"Early stopping patience: {early_stopping_patience}\n")
    
    # Setup validation callback
    val_callback = ValidationCallback(
        val_env=val_env,
        eval_freq=max(1, int(eval_freq)),
        patience=early_stopping_patience,
        verbose=1
    )
    
    # Train the agent
    agent.learn(
        total_timesteps=total_timesteps,
        callback=val_callback,
        progress_bar=True
    )
    
    # Collect training history (episodes completed per evaluation)
    train_history: List[Dict[str, Any]] = []
    
    return {
        'agent': agent,
        'train_history': train_history,
        'val_history': val_callback.val_history,
        'best_val_reward': val_callback.best_val_reward,
        'total_timesteps': agent.num_timesteps,
    }


def compare_regimes_aware_finrl(agent_aware: Any,
                                agent_naive: Any,
                                val_env_aware: Any,
                                val_env_naive: Any,
                                num_episodes: int = 50) -> Dict:
    """
    Compare regime-aware vs naive agent performance using FinRL agents.
    
    Args:
        agent_aware: Regime-aware FinRL DQN agent
        agent_naive: Naive FinRL DQN agent (no regime info)
        val_env_aware: Environment with regime info
        val_env_naive: Environment without regime info
        num_episodes: Number of evaluation episodes
        
    Returns:
        Comparison results dict
    """
    results_aware: List[Dict[str, Any]] = []
    results_naive: List[Dict[str, Any]] = []
    
    print(f"Comparing {num_episodes} episodes...")
    for ep in tqdm(range(num_episodes)):
        eval_aware = evaluate_episode(agent_aware, val_env_aware)
        eval_naive = evaluate_episode(agent_naive, val_env_naive)
        
        results_aware.append(eval_aware)
        results_naive.append(eval_naive)
    
    # Aggregate stats
    aware_rewards = [r['reward'] for r in results_aware]
    naive_rewards = [r['reward'] for r in results_naive]
    
    aware_sharpe = [r.get('sharpe_ratio', 0) for r in results_aware]
    naive_sharpe = [r.get('sharpe_ratio', 0) for r in results_naive]
    
    return {
        'aware_mean_reward': np.mean(aware_rewards),
        'aware_std_reward': np.std(aware_rewards),
        'naive_mean_reward': np.mean(naive_rewards),
        'naive_std_reward': np.std(naive_rewards),
        'improvement': (np.mean(aware_rewards) - np.mean(naive_rewards)) / (np.abs(np.mean(naive_rewards)) + 1e-8),
        'aware_mean_sharpe': np.mean(aware_sharpe),
        'naive_mean_sharpe': np.mean(naive_sharpe),
        'results_aware': results_aware,
        'results_naive': results_naive,
    }


def test_agents_on_period(agent: Any, env: Any, period_name: str = "Test") -> pd.DataFrame:
    """
    Test agent on a specific period and return detailed results.
    
    Args:
        agent: Trained FinRL DQN agent
        env: Test environment
        period_name: Name of period for logging
        
    Returns:
        DataFrame with weekly performance details
    """
    obs, _ = env.reset()  # gymnasium returns (obs, info)
    records: List[Dict[str, Any]] = []
    
    step_count = 0
    while True:
        action, _ = agent.predict(obs, deterministic=True)
        next_obs, reward, terminated, truncated, info = env.step(action)
        action_id = _as_int_action(action)
        
        records.append({
            'week': step_count,
            'action': info.get('action_name', f'Action {action_id}'),
            'return': info.get('portfolio_return', reward),
            'turnover': info.get('turnover', 0.0),
            'allocation': str(info.get('allocation', 'N/A')),
            'period': period_name,
        })
        
        obs = next_obs
        step_count += 1
        
        if terminated or truncated:
            break
    
    return pd.DataFrame(records)
