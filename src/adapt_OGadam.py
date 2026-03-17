import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import os
from torch.utils.data import DataLoader, TensorDataset

# Import your custom modules
from model import get_model
from metrics import calculate_macro_f1, calculate_feature_wasserstein

# Hardware configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(path_X, path_y):
    """Load .npy files and convert to Tensor objects for classification."""
    X = np.load(path_X)
    y = np.load(path_y)
    
    tensor_X = torch.FloatTensor(X)
    tensor_y = torch.LongTensor(y) # MUST be LongTensor for CrossEntropyLoss
    
    return tensor_X, tensor_y

def evaluate_model(model, test_loader, criterion):
    """Evaluates the model on the disjoint test set and calculates CE Loss and Macro F1."""
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
            
            # --- THE CRITICAL CLASSIFICATION CHANGE ---
            # Get the index of the max logit (the predicted class)
            preds_classes = torch.argmax(preds, dim=1)
            
            # Append the predicted classes, not the raw logits
            all_preds.append(preds_classes.cpu())
            all_targets.append(batch_y.cpu())
            
    avg_test_loss = test_loss_accum / len(test_loader)
    
    # Calculate Macro F1 instead of rBME
    test_f1 = calculate_macro_f1(torch.cat(all_targets), torch.cat(all_preds))
    
    return avg_test_loss, test_f1

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
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_classes", type=int, default=20, help="Number of protein families")
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
    model = get_model(
        input_dim=input_dim, 
        num_classes=args.num_classes,
        hidden_dim=args.hidden_dim, 
        dropout=args.dropout
    ).to(DEVICE)
    model.load_state_dict(torch.load(args.base_model, map_location=DEVICE))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 4. Setup CSV Logging (Changed headers to test_ce and test_f1)
    batch_log_file = args.output_model.replace('.pt', '_batch_log.csv')
    with open(batch_log_file, 'w') as f:
        f.write("batch_number,samples_seen,train_loss,test_ce,test_f1\n")

    print(f"\n--- Starting Adaptation ---")
    print(f"Pool size: {len(pool_X)} | Test size: {len(test_X)} | Batch size: {args.batch_size}")
    
    # 5. Evaluate INITIAL state before any adaptation (Batch 0)
    initial_ce, initial_f1 = evaluate_model(model, test_loader, criterion)
    with open(batch_log_file, 'a') as f:
        f.write(f"0,0,0.0,{initial_ce:.6f},{initial_f1:.6f}\n")
    print(f"Initial State (0 Batches) -> Test CE: {initial_ce:.4f} | Test Macro F1: {initial_f1:.4f}")

    # 6. The Active Learning Loop
    global_batch = 0
    samples_seen = 0
    eval_every = 500  # <--- Evaluates every 500 batches instead of every 1 batch

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
        
        # B & C. Evaluate and log ONLY every 500 batches or on the final batch
        if global_batch % eval_every == 0 or global_batch == len(pool_loader):
            test_ce, test_f1 = evaluate_model(model, test_loader, criterion)
            
            with open(batch_log_file, 'a') as f:
                f.write(f"{global_batch},{samples_seen},{train_loss.item():.6f},{test_ce:.6f},{test_f1:.6f}\n")
            print(f"Batch {global_batch} | Test CE: {test_ce:.4f} | Test F1: {test_f1:.4f}")

    # 7. Save the final adapted model
    torch.save(model.state_dict(), args.output_model)
    
    # 8. Calculate final Wasserstein and output for Nextflow logging
    w_dist = calculate_feature_wasserstein(ref_X, pool_X.numpy())
    print("-" * 50)
    print(f"Adaptation Complete.")
    print(f"{'Wasserstein':<15} | {w_dist:<15.6f}")
    print(f"{'Final Test CE':<15} | {test_ce:<15.6f}")
    print(f"{'Final Test F1':<15} | {test_f1:<15.6f}")
    print("-" * 50)
