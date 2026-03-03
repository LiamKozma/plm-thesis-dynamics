import torch
import torch.nn as nn
import numpy as np
import argparse
from torch.utils.data import DataLoader, TensorDataset

# Import your custom modules
from model import get_model
from metrics import calculate_rbme, calculate_feature_wasserstein

# Configuration
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(path_X, path_y):
    """
    Load .npy files and convert to Tensor objects.
    (Reused from train.py logic)
    """
    print(f"Loading data from {path_X}...")
    X = np.load(path_X)
    y = np.load(path_y)
    
    # Convert to PyTorch Tensors
    tensor_X = torch.FloatTensor(X)
    tensor_y = torch.FloatTensor(y).view(-1, 1) # Ensure shape is (N, 1)
    
    return tensor_X, tensor_y

if __name__ == "__main__":
    # 1. Parse CLI Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained source_model.pt")
    parser.add_argument("--target_x", type=str, required=True, help="Path to Target X npy")
    parser.add_argument("--target_y", type=str, required=True, help="Path to Target Y npy")
    parser.add_argument("--ref_x", type=str, required=True, help="Path to Source X for distance calculation")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    # 2. Load Data from CLI Arguments
    X, y = load_data(args.target_x, args.target_y)
    ref_X = np.load(args.ref_x) # Load reference data
    
    # Create DataLoader (No shuffling needed for evaluation)
    test_loader = DataLoader(TensorDataset(X, y), batch_size=args.batch_size, shuffle=False)
    
    # 3. Initialize Model & Load Weights
    # Detect input dimension from the target data (must match source data dim)
    input_dim = X.shape[1]
    print(f"Detected input dimension: {input_dim}")

    model = get_model(input_dim=input_dim).to(DEVICE)
    
    # Load the saved state dictionary
    print(f"Loading model weights from {args.model_path}...")
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    model.eval()
    
    criterion = nn.MSELoss()
    
    # 4. Evaluation Loop
    total_loss_accum = 0
    all_preds = []
    all_targets = []
    
    print(f"\nStarting evaluation on {DEVICE}...")
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            
            total_loss_accum += loss.item()
            all_preds.append(preds.cpu())
            all_targets.append(batch_y.cpu())
            
    # 5. Compute Metrics
    avg_loss = total_loss_accum / len(test_loader)
    
    # Concatenate all batches for rBME calculation
    final_targets = torch.cat(all_targets)
    final_preds = torch.cat(all_preds)
    
    val_rbme = calculate_rbme(final_targets, final_preds)
    w_dist = calculate_feature_wasserstein(ref_X, X) # Calculate distance

    # 6. Output Results
    print("-" * 50)
    print(f"{'Metric':<15} | {'Score':<15}")
    print("-" * 50)
    print(f"{'Wasserstein':<15} | {w_dist:<15.6f}") # Add this line!
    print(f"{'MSE Loss':<15} | {avg_loss:<15.6f}")
    print(f"{'rBME':<15} | {val_rbme:<15.6f}")
    print("-" * 50)
