import numpy as np
import pandas as pd
from scipy.stats import linregress

def generate_summary_tables(metrics: dict) -> pd.DataFrame:
    """
    Aggregates metrics across samples, layers, and heads to produce 
    a clean, publication-ready pandas DataFrame table summarizing 
    attention statistics by layer.
    
    Args:
        metrics (dict): Consolidated metrics dictionary containing 
            head_entropy, layer_entropy, and sparsity metrics.
            
    Returns:
        pd.DataFrame: A layer-wise statistical summary table.
    """
    # Extract dimensions
    # head_entropy shape: [num_samples, num_layers, num_heads]
    head_entropy = metrics["head_entropy"]
    num_samples, num_layers, num_heads = head_entropy.shape
    
    # Calculate layer-level averages across samples and heads
    mean_entropy = head_entropy.mean(axis=(0, 2))
    std_entropy = head_entropy.std(axis=(0, 2))
    
    # Sparsity metrics
    sparsity_pct = metrics["sparsity_percentage"].mean(axis=(0, 2, 3)) # [num_layers]
    density = metrics["density"].mean(axis=(0, 2, 3))
    
    # Top-k masses
    top_1 = metrics["top_k_masses"]["top_1_mass"].mean(axis=(0, 2, 3))
    top_5 = metrics["top_k_masses"]["top_5_mass"].mean(axis=(0, 2, 3))
    top_10 = metrics["top_k_masses"]["top_10_mass"].mean(axis=(0, 2, 3))
    top_50 = metrics["top_k_masses"]["top_50_mass"].mean(axis=(0, 2, 3))
    
    # Construct Summary DataFrame
    summary_df = pd.DataFrame({
        "Layer": np.arange(num_layers),
        "Entropy (Mean)": mean_entropy,
        "Entropy (Std)": std_entropy,
        "Sparsity % (Mean)": sparsity_pct,
        "Density (Mean)": density,
        "Top-1 Mass (Mean)": top_1,
        "Top-5 Mass (Mean)": top_5,
        "Top-10 Mass (Mean)": top_10,
        "Top-50 Mass (Mean)": top_50
    })
    
    # Round metrics for presentation
    summary_df = summary_df.round(4)
    
    return summary_df


def fit_power_law(attention_vector: np.ndarray) -> tuple:
    """
    Explores the power-law concentration of attention weights.
    Fits a linear regression model to the log-log ranked weights:
    log(weight) = -alpha * log(rank) + constant
    
    Args:
        attention_vector (np.ndarray): 1D array of attention weights, 
            e.g., from a specific query position to all keys, sorted descending.
            
    Returns:
        tuple: (alpha, r_squared) where alpha is the power law decay exponent,
            and r_squared is the coefficient of determination.
            Returns (None, None) if fit is mathematically impossible.
    """
    # Sort and remove zero/near-zero elements to ensure log safety
    sorted_weights = np.sort(attention_vector)[::-1]
    valid_mask = sorted_weights > 1e-5
    filtered_weights = sorted_weights[valid_mask]
    
    n = len(filtered_weights)
    if n < 5:  # Require minimum samples for statistical validity
        return None, None
        
    ranks = np.arange(1, n + 1)
    
    # Compute log ranks and log weights
    log_ranks = np.log(ranks)
    log_weights = np.log(filtered_weights)
    
    # Perform linear regression
    slope, intercept, r_value, _, _ = linregress(log_ranks, log_weights)
    
    alpha = -slope  # Decaying tail exponent
    r_squared = r_value ** 2
    
    return alpha, r_squared
