import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class GatedTacticalExtractor(BaseFeaturesExtractor):
    """
    Custom PyTorch Feature Extractor for PPO implementing a Learned Gating Mask
    (Gated Feature Attention) over our 21D tactical state vector.
    
    This layer dynamically scales, masks, or highlights features—such as zeroing out
    missing StatsBomb 360 data or amplifying fatigue signals late in the match.
    """
    def __init__(self, observation_space, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        input_dim = observation_space.shape[0]  # Expected to be 21
        
        # Gating network: maps input vector to a coefficient vector g in [0, 1]^21
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )
        
        # Dense feature projection layers
        self.fc = nn.Sequential(
            nn.Linear(input_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU()
        )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # 1. Compute the soft learned gating mask coefficients
        gating_mask = self.gate_net(observations)
        
        # 2. Apply element-wise masking (Hadamard product)
        gated_state = observations * gating_mask
        
        # 3. Project to the policy representation space
        return self.fc(gated_state)
