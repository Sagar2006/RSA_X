import torch
import numpy as np

def compute_entropy(attention_matrix: torch.Tensor, epsilon: float = 1e-12) -> torch.Tensor:
    """
    Computes Shannon entropy along the last dimension of the attention tensor.
    Handles zero values exactly using masking to avoid log(0) numerical instability.
    
    Formula: H = -sum(p * log(p))
    
    Args:
        attention_matrix (torch.Tensor): Attention weights of shape [..., seq_len, seq_len].
            Each row along the last dimension must sum to 1.0.
        epsilon (float): Small value for clamping if fallback is needed.
        
    Returns:
        torch.Tensor: Calculated entropy of shape [..., seq_len].
    """
    # Create mask of positive values
    positive_mask = attention_matrix > epsilon
    
    # Pre-allocate output log tensor
    log_p = torch.zeros_like(attention_matrix)
    
    # Calculate natural log only for positive values (0 * log(0) is mathematically 0)
    log_p[positive_mask] = torch.log(attention_matrix[positive_mask])
    
    # Compute Shannon entropy
    entropy = -(attention_matrix * log_p).sum(dim=-1)
    
    return entropy


class EntropyAnalyzer:
    """
    Entropy Analysis System for Transformer Attention.
    Calculates statistical entropy metrics at various granularity levels:
    per-token, per-head, per-layer, and dataset-wide.
    """
    def __init__(self, config: dict):
        self.config = config
        self.epsilon = float(config["analysis"]["epsilon"])

    def analyze(self, attention_patterns: torch.Tensor) -> dict:
        """
        Analyzes attention entropy for a batch of patterns.
        
        Args:
            attention_patterns (torch.Tensor): Tensor of shape 
                [batch_size, num_layers, num_heads, seq_len, seq_len].
                
        Returns:
            dict: Dictionary containing structured entropy tensors at various levels.
        """
        # Shape inputs
        batch_size, num_layers, num_heads, seq_len, _ = attention_patterns.shape
        
        # Calculate per-token (query) entropy
        # Output shape: [batch_size, num_layers, num_heads, seq_len]
        token_entropy = compute_entropy(attention_patterns, self.epsilon)
        
        # Aggregate to per-head-per-sample entropy (averaging across sequence query positions)
        # Output shape: [batch_size, num_layers, num_heads]
        head_entropy = token_entropy.mean(dim=-1)
        
        # Aggregate to per-layer-per-sample entropy (averaging across heads)
        # Output shape: [batch_size, num_layers]
        layer_entropy = head_entropy.mean(dim=-1)
        
        # Aggregate to per-sample entropy (averaging across layers)
        # Output shape: [batch_size]
        sample_entropy = layer_entropy.mean(dim=-1)
        
        # Theoretical maximum entropy for sequence length L: log(L)
        max_entropy = np.log(seq_len)
        
        return {
            "token_entropy": token_entropy.numpy(),
            "head_entropy": head_entropy.numpy(),
            "layer_entropy": layer_entropy.numpy(),
            "sample_entropy": sample_entropy.numpy(),
            "max_entropy": max_entropy,
            "mean_dataset_entropy": head_entropy.mean().item()
        }
