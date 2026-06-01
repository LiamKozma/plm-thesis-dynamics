import torch
import torch.nn as nn

class ProteinFamilyPredictor(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=512, dropout=0.1):
        super(ProteinFamilyPredictor, self).__init__()
        
        self.net = nn.Sequential(
            # Layer 1: Dynamically sized input (e.g., 1280 for ESM-2)
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Layer 2: Non-linear processing
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Layer 3: Output (Raw logits for CrossEntropyLoss)
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)

def get_model(input_dim, num_classes, hidden_dim=512, dropout=0.1):
    return ProteinFamilyPredictor(
        input_dim=input_dim, 
        num_classes=num_classes, 
        hidden_dim=hidden_dim, 
        dropout=dropout
    )
