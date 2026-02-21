import torch
import torch.nn as nn
import numpy as np
from gymnasium.spaces import Dict, Box
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from typing import Dict as TDict
from gym_enviroment.config.config import config

class NegotiationHeuristicHead(nn.Module):
    """
    Regular Conv1d over time.
    Input:  (B, T, F)
    Output: (B, hidden_dim)
    """
    def __init__(self, input_dim: int, hidden_dim: int = None):
        super().__init__()
        self.device = torch.device(config.core['device'])
        self.hidden_dim = hidden_dim or config.feature_extractor['hidden_size']
        drop = config.feature_extractor['dropout']
        mid = max(64, self.hidden_dim // 2)

        # 1x1 mixes features per time step. 3x1 captures short temporal patterns.
        self.temporal = nn.Sequential(
            nn.Conv1d(input_dim, mid, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(mid, mid, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # -> (B, mid, 1)
        )
        # keep backward-compatible alias
        self.lstm = self.temporal

        self.projection = nn.Sequential(
            nn.Linear(mid, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) -> (B, F, T) for Conv1d
        x = x.to(self.device).permute(0, 2, 1)
        z = self.temporal(x).squeeze(-1)   # (B, mid)
        return self.projection(z)          # (B, hidden_dim)

class CandidateForecastsHead(nn.Module):
    """
    Regular Conv1d over candidate forecast sequences. Slightly wider.
    Input:  (B, max_candidates, forecast_length)
    Output: (B, hidden_dim)
    """
    def __init__(self, max_candidates: int = None, forecast_length: int = None, hidden_dim: int = None):
        super().__init__()
        self.device = torch.device(config.core['device'])
        self.max_candidates = max_candidates or config.core['max_candidates']
        self.forecast_length = forecast_length or config.core['forecast_length']
        self.hidden_dim = hidden_dim or config.feature_extractor['hidden_size']
        drop = config.feature_extractor['dropout']

        mid = max(96, self.hidden_dim)
        half = max(64, self.hidden_dim // 2)

        self.temporal = nn.Sequential(
            nn.Conv1d(1, mid // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(mid // 2, mid, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # -> (B*C, mid, 1)
        )
        # keep old names
        self.lstm = self.temporal

        self.aggregation = nn.Sequential(
            nn.Linear(mid, half),
            nn.ReLU(),
            nn.Dropout(drop)
        )

        self.projection = nn.Sequential(
            nn.Linear(half, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L). Treat each candidate as a sample.
        x = x.to(self.device)
        B, C, L = x.shape
        x = x.view(B * C, 1, L)
        z = self.temporal(x).squeeze(-1)      # (B*C, mid)
        z = self.aggregation(z)               # (B*C, half)
        z = z.view(B, C, -1).mean(dim=1)      # (B, half)
        return self.projection(z)             # (B, hidden_dim)

class StatisticalFeaturesHead(nn.Module):
    """
    Neural network head for processing statistical features.
    
    Processes statistical metrics like awareness correlation and other
    aggregated statistics from the negotiation history.
    """
    
    def __init__(self, input_dim: int = None, hidden_dim: int = None):
        super().__init__()
        self.device = torch.device(config.core['device'])
        self.input_dim = input_dim or config.core['statistical_features']
        self.hidden_dim = hidden_dim or config.feature_extractor['hidden_size']
        
        # Simple MLP for statistical features
        self.projection = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.feature_extractor['dropout']),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.feature_extractor['dropout']),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim)
        )
        
        # Move to device
        self.to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, statistical_features)
        Returns:
            Tensor of shape (batch_size, hidden_dim)
        """
        # Ensure input is on correct device
        x = x.to(self.device)
        
        return self.projection(x)


class FeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for NegoFormer multi-input observations.
    
    Combines negotiation history processing with candidate forecast analysis
    to create a unified representation for policy learning.
    """
    
    def __init__(self, observation_space: Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        self.device = torch.device(config.core['device'])
        
        # Extract dimensions from observation space
        negotiation_shape = observation_space['negotiation_history'].shape
        forecasts_shape = observation_space['candidate_forecasts'].shape
        statistical_shape = observation_space['statistical_features'].shape
        
        self.sequence_length = config.core['sequence_length']
        self.negotiation_features = config.core['negotiation_features']
        self.max_candidates = config.core['max_candidates']
        self.forecast_length = config.core['forecast_length']
        self.statistical_features = statistical_shape[0]  # Number of statistical features
        
        # Initialize processing heads
        self.negotiation_head = NegotiationHeuristicHead(
            input_dim=self.negotiation_features,
            hidden_dim=config.feature_extractor['hidden_size']
        )
        
        self.forecasts_head = CandidateForecastsHead(
            max_candidates=self.max_candidates,
            forecast_length=self.forecast_length,
            hidden_dim=config.feature_extractor['hidden_size']
        )
        
        self.statistical_head = StatisticalFeaturesHead(
            hidden_dim=config.feature_extractor['hidden_size']
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(config.feature_extractor['hidden_size'] * 3, features_dim),  # Combine all three heads
            nn.ReLU(),
            nn.Dropout(config.feature_extractor['dropout']),
            nn.Linear(features_dim, features_dim)
        )
        
        # Move to device
        self.to(self.device)
    
    def forward(self, observations: TDict[str, torch.Tensor]) -> torch.Tensor:
        """
        Process multi-modality observations into unified feature representation.
        
        Args:
            observations: Dictionary containing:
                - 'negotiation_history': (batch, sequence_length, n_features)
                - 'candidate_forecasts': (batch, max_candidates, forecast_length)
                - 'statistical_features': (batch, n_statistical_features)
        
        Returns:
            Unified feature tensor of shape (batch, features_dim)
        """
        # Ensure all observations are on correct device
        observations = {k: v.to(self.device) for k, v in observations.items()}
        
        # Process negotiation history
        negotiation_features = self.negotiation_head(observations['negotiation_history'])
        
        # Process candidate forecasts
        forecasts_features = self.forecasts_head(observations['candidate_forecasts'])
        
        # Process statistical features
        statistical_features = self.statistical_head(observations['statistical_features'])
        
        # Combine representations
        combined = torch.cat([negotiation_features, forecasts_features, statistical_features], dim=1)
        
        return self.fusion(combined)


class MultiInputPolicy(MultiInputActorCriticPolicy):
    """
    Custom MultiInput policy for NegoFormer agent.
    
    Integrates the custom feature extractor with standard actor-critic architecture
    optimized for negotiation tasks.
    """
    
    def __init__(self, *args, **kwargs):
        features_dim = config.feature_extractor.get('features_dim', 256)
        kwargs.setdefault('features_extractor_class', FeaturesExtractor)
        kwargs.setdefault('features_extractor_kwargs', {'features_dim': features_dim})
        
        kwargs.setdefault('net_arch', [features_dim, features_dim])
        kwargs.setdefault('activation_fn', nn.ReLU)
        
        super().__init__(normalize_images=False, *args, **kwargs)
        
        # Ensure the policy is moved to the correct device
        device = torch.device(config.core['device'])
        self.to(device)


def create_observation_space() -> Dict:
    """
    Create the observation space for NegoFormer RL environment.
    
    Returns:
        Gymnasium Dict space for multi-input observations
    """
    return Dict({
        'negotiation_history': Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(config.core['sequence_length'], config.core['negotiation_features']), 
            dtype=np.float32
        ),
        'candidate_forecasts': Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(config.core['max_candidates'], config.core['forecast_length']), 
            dtype=np.float32
        ),
        'statistical_features': Box(
            low=-np.inf,
            high=np.inf,
            shape=(config.core['statistical_features'],),
            dtype=np.float32
        )
    })


def get_default_policy():
    """Convenience function to get the default NegoFormer policy class."""
    return MultiInputPolicy


def get_default_observation_space():
    """Convenience function to get the default observation space."""
    return create_observation_space()
