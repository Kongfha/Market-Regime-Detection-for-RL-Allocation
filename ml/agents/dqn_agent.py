"""Deep Q-Learning agent with temporal attention."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
from typing import Tuple, Dict, List, Optional
import random

from ml.models.attention_qnetwork import TemporalAttentionQNetwork, DuelingTemporalAttentionQNetwork


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum size of buffer
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state: np.ndarray, action: int, next_state: np.ndarray, 
             reward: float, done: bool):
        """Add experience to buffer."""
        self.buffer.append(Transition(state, action, next_state, reward, done))
    
    def sample(self, batch_size: int) -> List[Transition]:
        """Sample a random batch from buffer."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        """Return buffer size."""
        return len(self.buffer)


class AttentionDQNAgent:
    """
    Deep Q-Network agent with temporal attention for portfolio allocation.
    
    Features:
    - Temporal attention Q-network
    - Double DQN for stability
    - Prioritized experience replay (optional)
    - Epsilon-greedy exploration
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 seq_len: int = 4,
                 learning_rate: float = 1e-4,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: int = 1000,
                 buffer_capacity: int = 10000,
                 batch_size: int = 32,
                 target_update_freq: int = 100,
                 use_dueling: bool = True,
                 device: str = None):
        """
        Initialize DQN agent.
        
        Args:
            state_dim: Dimension of state features
            action_dim: Number of discrete actions
            seq_len: Sequence length for temporal attention
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial epsilon for exploration
            epsilon_end: Final epsilon for exploration
            epsilon_decay: Steps to decay epsilon
            buffer_capacity: Replay buffer capacity
            batch_size: Batch size for training
            target_update_freq: Update target network every N steps
            use_dueling: Use dueling DQN variant
            device: Device for computation (cuda/cpu)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.seq_len = seq_len
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Exploration schedule
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        
        # Device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(self.device)
        
        # Networks
        if use_dueling:
            self.q_network = DuelingTemporalAttentionQNetwork(
                state_dim=state_dim,
                action_dim=action_dim,
                seq_len=seq_len,
                lstm_hidden=64,
                attention_heads=4,
                fc_hidden=128,
                dropout=0.1
            ).to(self.device)
            
            self.target_network = DuelingTemporalAttentionQNetwork(
                state_dim=state_dim,
                action_dim=action_dim,
                seq_len=seq_len,
                lstm_hidden=64,
                attention_heads=4,
                fc_hidden=128,
                dropout=0.1
            ).to(self.device)
        else:
            self.q_network = TemporalAttentionQNetwork(
                state_dim=state_dim,
                action_dim=action_dim,
                seq_len=seq_len,
                lstm_hidden=64,
                attention_heads=4,
                fc_hidden=128,
                dropout=0.1
            ).to(self.device)
            
            self.target_network = TemporalAttentionQNetwork(
                state_dim=state_dim,
                action_dim=action_dim,
                seq_len=seq_len,
                lstm_hidden=64,
                attention_heads=4,
                fc_hidden=128,
                dropout=0.1
            ).to(self.device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        
        # Training stats
        self.steps = 0
        self.episodes = 0
        self.losses = []
        
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state [seq_len, state_dim]
            training: Whether in training mode (affects epsilon)
            
        Returns:
            Action ID (0 to action_dim-1)
        """
        if training and np.random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values, _ = self.q_network(state_tensor)
            action = q_values.max(1)[1].item()
        
        return action
    
    def store_transition(self, state: np.ndarray, action: int, 
                        next_state: np.ndarray, reward: float, done: bool):
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, next_state, reward, done)
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step using Double DQN.
        
        Returns:
            Loss value, or None if buffer not full enough
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        
        # Unpack batch
        states = torch.FloatTensor(np.array([t.state for t in batch])).to(self.device)
        actions = torch.LongTensor(np.array([t.action for t in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([t.reward for t in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([t.next_state for t in batch])).to(self.device)
        dones = torch.FloatTensor(np.array([t.done for t in batch])).to(self.device)
        
        # Compute Q(s, a) from Q-network
        q_values, _ = self.q_network(states)
        q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN: Use Q-network to select best action, target network to evaluate
        with torch.no_grad():
            # Select best action with Q-network
            q_next, _ = self.q_network(next_states)
            best_actions = q_next.argmax(1)
            
            # Evaluate with target network
            q_target_next, _ = self.target_network(next_states)
            q_target_selected = q_target_next.gather(1, best_actions.unsqueeze(1)).squeeze(1)
            
            # Compute target Q-values
            q_target = rewards + self.gamma * q_target_selected * (1 - dones)
        
        # Compute loss (TD error)
        loss = nn.functional.smooth_l1_loss(q_selected, q_target)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        self.steps += 1
        self.losses.append(loss.item())
        
        # Update target network periodically
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                       np.exp(-self.steps / self.epsilon_decay)
        
        return loss.item()
    
    def episode_end(self):
        """Called at end of episode (for tracking)."""
        self.episodes += 1
    
    def get_attention_weights(self, state: np.ndarray) -> np.ndarray:
        """
        Get attention weights for a given state.
        
        Args:
            state: State observation [seq_len, state_dim]
            
        Returns:
            Attention weights from the network
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            _, attention_weights = self.q_network(state_tensor)
        
        return attention_weights.cpu().numpy()
    
    def save_checkpoint(self, filepath: str):
        """Save agent checkpoint."""
        checkpoint = {
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'episodes': self.episodes,
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str, trusted_source: bool = True):
        """Load agent checkpoint.

        Args:
            filepath: Checkpoint path.
            trusted_source: If True, allow full unpickling (required for some
                legacy checkpoints under PyTorch 2.6+).
        """
        try:
            checkpoint = torch.load(
                filepath,
                map_location=self.device,
                weights_only=not trusted_source,
            )
        except TypeError:
            checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.episodes = checkpoint['episodes']
        print(f"Checkpoint loaded from {filepath}")
    
    def get_stats(self) -> Dict:
        """Get agent training statistics."""
        return {
            'steps': self.steps,
            'episodes': self.episodes,
            'epsilon': self.epsilon,
            'avg_loss_100': np.mean(self.losses[-100:]) if self.losses else 0.0,
            'buffer_size': len(self.replay_buffer),
        }
