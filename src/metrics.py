from scipy.stats import wasserstein_distance
import numpy as np
import torch
from sklearn.metrics import f1_score

def calculate_macro_f1(y_true, y_pred):
    """
    Calculate Macro F1-Score for protein family classification.
    Treats all families equally, regardless of class imbalance.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # zero_division=0 prevents warnings if a model totally collapses and predicts 0 for a class
    return f1_score(y_true, y_pred, average='macro', zero_division=0)

def calculate_feature_wasserstein(source_x, target_x):
    """
    Calculates the average feature-wise 1D Wasserstein distance.
    """
    if hasattr(source_x, 'numpy'): source_x = source_x.cpu().numpy()
    if hasattr(target_x, 'numpy'): target_x = target_x.cpu().numpy()
    
    dist = 0.0
    num_features = source_x.shape[1]
    
    for i in range(num_features):
        dist += wasserstein_distance(source_x[:, i], target_x[:, i])
        
    return dist / num_features
