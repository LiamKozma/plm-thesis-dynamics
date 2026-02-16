import numpy as np
import torch

def calculate_rbme(y_true, y_pred, num_bins=10, epsilon=1e-8):
    """
    Calculate Relative Bin-Mean Error (rBME) for protein function prediction.
    
    This metric forces the model to learn the 'rare' high-fitness variants
    by weighting performance in the 'dead' zones and 'active' zones equally.
    
    Args:
        y_true (np.array): Ground truth functional scores.
        y_pred (np.array): Model predicted scores.
        num_bins (int): Number of bins to partition the target range.
        epsilon (float): Stability term.
        
    Returns:
        float: The rBME score (lower is better).
    """
    # Detach from graph if tensors are passed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Flatten arrays
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # Dynamic binning based on the ground truth range
    min_val, max_val = np.min(y_true), np.max(y_true)
    
    # Edge case: minimal variance
    if np.abs(max_val - min_val) < epsilon:
        return np.mean(np.abs(y_true - y_pred))

    bin_edges = np.linspace(min_val, max_val, num_bins + 1)
    
    # Digitize: Assign samples to bins (1 to num_bins)
    bin_indices = np.digitize(y_true, bin_edges)
    
    bin_relative_errors = []
    
    for k in range(1, num_bins + 1):
        mask = (bin_indices == k)
        
        if not np.any(mask):
            continue
            
        y_true_k = y_true[mask]
        y_pred_k = y_pred[mask]
        
        # Mean Absolute Error for this bin
        mae_k = np.mean(np.abs(y_true_k - y_pred_k))
        
        # Normalization Factor (prevent division by zero for dead variants)
        norm_factor = np.mean(np.abs(y_true_k)) + epsilon
        
        bin_relative_errors.append(mae_k / norm_factor)
        
    return np.mean(bin_relative_errors) if bin_relative_errors else 0.0
