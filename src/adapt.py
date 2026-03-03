import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import os
from torch.utils.data import DataLoader, TensorDataset

# Import your custom modules
from model import get_model
from metrics import calculate_rbme, calculate_feature_wasserstein

# Hardware configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(path_X, path_y):
    """Load .npy files and convert to Tensor objects."""
    X = np.load(path_X)
    y = np.load(path_y)
    tensor_X = torch.FloatTensor(X)
    tensor_y = torch.FloatTensor(y).view(-1, 1)
    return tensor_X, tensor_y

def evaluate_model(model, test_loader, criterion):
    """Evaluates the model on the disjoint test set and calculates MSE and rBME."""
    model.eval()
    test_loss_accum = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            
            test_loss_accum += loss.item()
            all_preds.append(preds.cpu())
            all_targets.append(batch_y.cpu())
            
    avg_test_loss = test_loss_accum / len(test_loader)
    test_rbme = calculate_rbme(torch.cat(all_targets), torch.cat(all_preds))
    
    return avg_test_loss, test_rbme

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True, help="Path to trained source model")
    parser.add_argument("--pool_x", type=str, required=True)
    parser.add_argument("--pool_y", type=str, required=True)
    parser.add_argument("--test_x", type=str, required=True)
    parser.add_argument("--test_y", type=str, required=True)
    parser.add_argument("--ref_x", type=str, required=True, help="Reference source data for Wasserstein")
    parser.add_argument("--output_model", type=str, required=True)
    
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    # 1. Load Data
    pool_X, pool_y = load_data(args.pool_x, args.pool_y)
    test_X, test_y = load_data(args.test_x, args.test_y)
    ref_X = np.load(args.ref_x)

    # 2. Create DataLoaders
    pool_loader = DataLoader(TensorDataset(pool_X, pool_y), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_X, test_y), batch_size=args.batch_size, shuffle=False)

    # 3. Initialize and Load Base Model
    input_dim = pool_X.shape[1]
    model = get_model(input_dim=input_dim).to(DEVICE)
    model.load_state_dict(torch.load(args.base_model, map_location=DEVICE))
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 4. Setup CSV Logging
    batch_log_file = args.output_model.replace('.pt', '_batch_log.csv')
    with open(batch_log_file, 'w') as f:
        f.write("batch_number,samples_seen,train_loss,test_mse,test_rbme\n")

    print(f"\n--- Starting Adaptation ---")
    print(f"Pool size: {len(pool_X)} | Test size: {len(test_X)} | Batch size: {args.batch_size}")
    
    # 5. Evaluate INITIAL state before any adaptation (Batch 0)
    initial_mse, initial_rbme = evaluate_model(model, test_loader, criterion)
    with open(batch_log_file, 'a') as f:
        f.write(f"0,0,0.0,{initial_mse:.6f},{initial_rbme:.6f}\n")
    print(f"Initial State (0 Batches) -> Test MSE: {initial_mse:.4f} | Test rBME: {initial_rbme:.4f}")

    # 6. The Active Learning Loop
    global_batch = 0
    samples_seen = 0

    for batch_X, batch_y in pool_loader:
        global_batch += 1
        samples_seen += len(batch_X)
        
        # A. Train on the Pool batch
        model.train()
        batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
        
        optimizer.zero_grad()
        predictions = model(batch_X)
        train_loss = criterion(predictions, batch_y)
        train_loss.backward()
        optimizer.step()
        
        # B. Immediately evaluate on the entire Test set
        test_mse, test_rbme = evaluate_model(model, test_loader, criterion)
        
        # C. Log the metrics to CSV
        with open(batch_log_file, 'a') as f:
            f.write(f"{global_batch},{samples_seen},{train_loss.item():.6f},{test_mse:.6f},{test_rbme:.6f}\n")

    # 7. Save the final adapted model
    torch.save(model.state_dict(), args.output_model)
    
    # 8. Calculate final Wasserstein and output for Nextflow logging
    w_dist = calculate_feature_wasserstein(ref_X, pool_X.numpy())
    print("-" * 50)
    print(f"Adaptation Complete.")
    print(f"{'Wasserstein':<15} | {w_dist:<15.6f}")
    print(f"{'Final Test MSE':<15} | {test_mse:<15.6f}")
    print(f"{'Final Test rBME':<15} | {test_rbme:<15.6f}")
    print("-" * 50)
