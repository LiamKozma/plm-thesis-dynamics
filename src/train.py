import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import random # Added for complete seeding
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# Import your custom modules
from model import get_model
from metrics import calculate_rbme, calculate_feature_wasserstein

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    """Sets the seed for reproducibility across runs."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data(path_X, path_y):
    """Load .npy files and convert to Tensor objects."""
    print(f"Loading data from {path_X}...")
    X = np.load(path_X)
    y = np.load(path_y)
    
    # Convert to PyTorch Tensors
    tensor_X = torch.FloatTensor(X)
    tensor_y = torch.FloatTensor(y).view(-1, 1) # Ensure shape is (N, 1)
    
    return tensor_X, tensor_y

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_x", type=str, required=True, help="Path to Source X npy")
    parser.add_argument("--source_y", type=str, required=True, help="Path to Source Y npy")
    parser.add_argument("--output_model", type=str, default="baseline_source.pt", help="Output model filename")
    parser.add_argument("--ref_x", type=str, required=True, help="Path to Source X for distance calculation")
    
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    set_seed(42)

    X, y = load_data(args.source_x, args.source_y)
    ref_X = np.load(args.ref_x) 
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=args.batch_size)
    
    input_dim = X.shape[1]
    model = get_model(input_dim=input_dim).to(DEVICE)
    
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    #  Training Loop
    for epoch in range(args.epochs):
        model.train()
        train_loss_accum = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss_accum += loss.item()
            
        avg_train_loss = train_loss_accum / len(train_loader)
        
        # Validation & Metric Calculation
        model.eval()
        val_loss_accum = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                preds = model(batch_X)
                loss = criterion(preds, batch_y)
                
                val_loss_accum += loss.item()
                all_preds.append(preds.cpu())
                all_targets.append(batch_y.cpu())
        
        avg_val_loss = val_loss_accum / len(val_loader)
        
        # Calculate custom rBME metric
        val_rbme = calculate_rbme(
            torch.cat(all_targets), 
            torch.cat(all_preds)
        )
        
        print(f"{epoch+1:<5} | {avg_train_loss:<12.6f} | {avg_val_loss:<12.6f} | {val_rbme:<12.6f}")

    # 7. Save the trained baseline
    torch.save(model.state_dict(), args.output_model)
    print(f"\nModel saved to {args.output_model}")

    # 8. Calculate and Output Wasserstein Distance
    # Calculate the distance between the source (ref_X) and the current training data (X)
    w_dist = calculate_feature_wasserstein(ref_X, X)
    
    print("-" * 50)
    print(f"{'Wasserstein':<15} | {w_dist:<15.6f}")
    print("-" * 50)
