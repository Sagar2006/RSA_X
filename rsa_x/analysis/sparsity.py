import torch
import numpy as np

def compute_sparsity_metrics(attention_patterns: torch.Tensor, config: dict) -> dict:
    """
    Computes key sparsity metrics from attention tensors.
    
    Metrics:
    - Top-k Mass: Cumulative sum of the top K largest attention weights.
    - Attention Density: Percentage of attention values exceeding a threshold (1/L).
    - Near-Zero Ratio: Percentage of attention values below a low threshold (e.g., 1e-4).
    
    Args:
        attention_patterns (torch.Tensor): Attention weights of shape 
            [batch_size, num_layers, num_heads, seq_len, seq_len].
        config (dict): Configuration dictionary containing thresholds.
        
    Returns:
        dict: A structured dictionary of sparsity metrics.
    """
    batch_size, num_layers, num_heads, seq_len, _ = attention_patterns.shape
    
    # Sort attention weights along key dimension in descending order
    # Shape: [batch_size, num_layers, num_heads, seq_len, seq_len]
    sorted_patterns, _ = torch.sort(attention_patterns, dim=-1, descending=True)
    
    # 1. Compute Top-k Mass
    top_k_values = config["analysis"]["top_k_values"]
    top_k_masses = {}
    
    for k in top_k_values:
        # Clamp k to seq_len to prevent indexing errors
        actual_k = min(k, seq_len)
        # Sum top k elements along key dimension
        k_mass = sorted_patterns[..., :actual_k].sum(dim=-1)
        # Store aggregated per-head metric (mean over batch and sequence query positions)
        top_k_masses[f"top_{k}_mass"] = k_mass.numpy()
        
    # 2. Compute Attention Density
    # Threshold represents uniform attention budget (1/seq_len)
    density_threshold = float(config["analysis"]["density_threshold"])
    density = (attention_patterns > density_threshold).float().mean(dim=-1)
    
    # 3. Compute Sparsity Percentage (fraction of elements below near_zero_threshold)
    near_zero_threshold = float(config["analysis"]["near_zero_threshold"])
    near_zero_mask = (attention_patterns <= near_zero_threshold).float()
    near_zero_ratio = near_zero_mask.mean(dim=-1)
    sparsity_percentage = near_zero_ratio * 100.0
    
    return {
        "top_k_masses": top_k_masses,
        "density": density.numpy(),
        "near_zero_ratio": near_zero_ratio.numpy(),
        "sparsity_percentage": sparsity_percentage.numpy()
    }


class SparsityAnalyzer:
    """
    Sparsity Analysis System for Transformer Attention.
    Computes top-k concentration, attention density, and near-zero ratio distributions.
    """
    def __init__(self, config: dict):
        self.config = config

    def analyze(self, attention_patterns: torch.Tensor) -> dict:
        """
        Runs comprehensive sparsity analysis on a batch of attention patterns.
        
        Args:
            attention_patterns (torch.Tensor): Attention patterns of shape 
                [batch_size, num_layers, num_heads, seq_len, seq_len].
                
        Returns:
            dict: Structured metrics dictionary containing NumPy arrays.
        """
        return compute_sparsity_metrics(attention_patterns, self.config)
