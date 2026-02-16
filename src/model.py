import torch
import torch.nn as nn

class FitnessPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, dropout=0.1):
        super(FitnessPredictor, self).__init__()
        
        self.net = nn.Sequential(
            # Layer 1: Dynamically sized input
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Layer 2: Non-linear processing
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Layer 3: Output (Scalar fitness score)
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x):
        return self.net(x)

# Update: Now accepts input_dim argument
def get_model(input_dim):
    return FitnessPredictor(input_dim=input_dim)
