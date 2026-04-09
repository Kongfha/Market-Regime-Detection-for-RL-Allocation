"""Temporal Attention-based Q-Network for DQN."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism for Q-network.
    
    Uses scaled dot-product attention to focus on relevant timesteps.
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        """
        Initialize temporal attention.
        
        Args:
            hidden_dim: Dimension of hidden state
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Multi-head attention layers
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute multi-head attention.
        
        Args:
            Q: Query tensor, shape (batch_size, seq_len, hidden_dim)
            K: Key tensor, shape (batch_size, seq_len, hidden_dim)
            V: Value tensor, shape (batch_size, seq_len, hidden_dim)
            
        Returns:
            output: Attended tensor, shape (batch_size, seq_len, hidden_dim)
            attention_weights: Attention weights, shape (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size = Q.shape[0]
        seq_len = Q.shape[1]
        
        # Linear transformations
        Q = self.query(Q)
        K = self.key(K)
        V = self.value(V)
        
        # Reshape for multi-head attention: (batch_size, seq_len, hidden_dim) 
        # -> (batch_size, seq_len, num_heads, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose to (batch_size, num_heads, seq_len, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Attention scores: (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Compute attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values: (batch_size, num_heads, seq_len, head_dim)
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads: (batch_size, seq_len, hidden_dim)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.hidden_dim)
        
        # Final linear transformation
        output = self.fc_out(context)
        output = self.dropout(output)
        
        return output, attention_weights


class TemporalAttentionQNetwork(nn.Module):
    """
    Deep Q-Network with temporal attention for portfolio allocation.
    
    Architecture:
    - LSTM for temporal feature processing
    - Temporal attention to focus on relevant timesteps
    - Fully connected layers for Q-value estimation
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 seq_len: int = 4,
                 lstm_hidden: int = 64,
                 attention_heads: int = 4,
                 fc_hidden: int = 128,
                 dropout: float = 0.1):
        """
        Initialize temporal attention Q-network.
        
        Args:
            state_dim: Dimension of state features
            action_dim: Number of discrete actions
            seq_len: Sequence length for temporal attention
            lstm_hidden: LSTM hidden dimension
            attention_heads: Number of attention heads
            fc_hidden: Fully connected hidden dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.seq_len = seq_len
        
        # Feature embedding (optional nonlinear projection)
        self.feature_embed = nn.Linear(state_dim, lstm_hidden)
        
        # LSTM for temporal processing
        self.lstm = nn.LSTM(input_size=lstm_hidden,
                           hidden_size=lstm_hidden,
                           num_layers=1,
                           batch_first=True,
                           dropout=dropout if seq_len > 1 else 0.0)
        
        # Temporal attention
        self.attention = TemporalAttention(hidden_dim=lstm_hidden,
                                          num_heads=attention_heads,
                                          dropout=dropout)
        
        # Q-value head
        self.fc1 = nn.Linear(lstm_hidden, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, fc_hidden)
        self.fc_out = nn.Linear(fc_hidden, action_dim)
        
        # Activation and regularization
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through attention Q-network.
        
        Args:
            state: Input state, shape (batch_size, seq_len, state_dim)
                   or (batch_size, state_dim) if seq_len is handled externally
                   
        Returns:
            q_values: Q-values for each action, shape (batch_size, action_dim)
            attention_weights: Attention weights from last attention head
        """
        batch_size = state.shape[0]
        
        # Handle both (batch_size, seq_len, state_dim) and (batch_size, state_dim) inputs
        if len(state.shape) == 2:
            state = state.unsqueeze(1)  # Add seq_len dimension
            
        # Embed features
        x = self.feature_embed(state)  # (batch_size, seq_len, lstm_hidden)
        x = self.relu(x)
        
        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(x)  # (batch_size, seq_len, lstm_hidden)
        
        # Temporal attention (use LSTM output as Q, K, V)
        attended, attention_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use final timestep after attention
        final_attended = attended[:, -1, :]  # (batch_size, lstm_hidden)
        
        # Q-value estimation
        x = self.relu(self.fc1(final_attended))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        q_values = self.fc_out(x)  # (batch_size, action_dim)
        
        return q_values, attention_weights
    
    def get_q_values(self, state: torch.Tensor) -> torch.Tensor:
        """Convenient wrapper to get Q-values only."""
        q_values, _ = self.forward(state)
        return q_values


class DuelingTemporalAttentionQNetwork(nn.Module):
    """
    Dueling variant: Separates value and advantage streams with attention.
    
    Architecture:
    - Shared LSTM with temporal attention
    - Value stream: estimates state value V(s)
    - Advantage stream: estimates action advantages A(a|s)
    - Q-values: Q(s,a) = V(s) + (A(s,a) - mean_a(A(s,a)))
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 seq_len: int = 4,
                 lstm_hidden: int = 64,
                 attention_heads: int = 4,
                 fc_hidden: int = 128,
                 dropout: float = 0.1):
        """Initialize dueling temporal attention Q-network."""
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.seq_len = seq_len
        
        # Shared layers
        self.feature_embed = nn.Linear(state_dim, lstm_hidden)
        self.lstm = nn.LSTM(input_size=lstm_hidden,
                           hidden_size=lstm_hidden,
                           num_layers=1,
                           batch_first=True,
                           dropout=dropout if seq_len > 1 else 0.0)
        self.attention = TemporalAttention(hidden_dim=lstm_hidden,
                                          num_heads=attention_heads,
                                          dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        # Value stream
        self.value_fc1 = nn.Linear(lstm_hidden, fc_hidden)
        self.value_fc2 = nn.Linear(fc_hidden, 1)
        
        # Advantage stream
        self.advantage_fc1 = nn.Linear(lstm_hidden, fc_hidden)
        self.advantage_fc2 = nn.Linear(fc_hidden, action_dim)
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through dueling attention Q-network.
        
        Args:
            state: Input state, shape (batch_size, seq_len, state_dim)
            
        Returns:
            q_values: Q-values, shape (batch_size, action_dim)
            attention_weights: Attention weights
        """
        if len(state.shape) == 2:
            state = state.unsqueeze(1)
            
        # Shared processing
        x = self.feature_embed(state)
        x = self.relu(x)
        lstm_out, _ = self.lstm(x)
        attended, attention_weights = self.attention(lstm_out, lstm_out, lstm_out)
        final_attended = attended[:, -1, :]
        
        # Value stream
        v = self.relu(self.value_fc1(final_attended))
        v = self.dropout(v)
        v = self.value_fc2(v)  # (batch_size, 1)
        
        # Advantage stream
        a = self.relu(self.advantage_fc1(final_attended))
        a = self.dropout(a)
        a = self.advantage_fc2(a)  # (batch_size, action_dim)
        
        # Combine: Q = V + (A - mean(A))
        a_mean = a.mean(dim=1, keepdim=True)
        q_values = v + (a - a_mean)
        
        return q_values, attention_weights
    
    def get_q_values(self, state: torch.Tensor) -> torch.Tensor:
        """Convenient wrapper to get Q-values only."""
        q_values, _ = self.forward(state)
        return q_values
