# dual_net.py

import torch
import torch.nn as nn

class DualNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layer_dim=42):
        super(DualNet, self).__init__()
        # Create some layers to encode the input state
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_layer_dim),
            nn.LeakyReLU()
        )
        # Critic output layers to estimate V from the state encoding
        self.critic = nn.Sequential(
            nn.Linear(hidden_layer_dim, 1)
        )
        # Actor output layers to estimate pi from the state encoding
        self.actor = nn.Sequential(
            nn.Linear(hidden_layer_dim, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, s, mode):
        # Get the device of the network parameters
        device = next(self.net.parameters()).device
        
        # Convert input to tensor and move to correct device
        if isinstance(s, torch.Tensor):
            x = s.clone().detach().float().view(1, -1).to(device)
        else:
            x = torch.tensor(s, dtype=torch.float32, device=device).view(1, -1)
        
        # Encode state
        x = self.net(x)
        
        if mode == 'actor':
            # Return probability distribution over actions
            x = self.actor(x)
        else:
            # Return estimate of state value
            x = self.critic(x)
        
        return x.squeeze()
