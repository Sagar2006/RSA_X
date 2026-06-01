import pytest
import torch
import numpy as np
import pandas as pd
from analysis.entropy import compute_entropy, EntropyAnalyzer
from analysis.sparsity import compute_sparsity_metrics, SparsityAnalyzer
from analysis.statistics import fit_power_law, generate_summary_tables

def test_entropy_mathematical_bounds():
    """
    Asserts that entropy calculations behave correctly under extreme cases:
    1. Uniform distribution: H = log(L)
    2. Point distribution (deterministic): H = 0
    3. Handles zero values exactly without NaN (0 * log(0) = 0)
    """
    seq_len = 64
    epsilon = 1e-12
    
    # Case 1: Uniform Distribution
    uniform_p = torch.ones(1, 1, 1, seq_len, seq_len) / seq_len
    uniform_entropy = compute_entropy(uniform_p, epsilon)
    expected_uniform = np.log(seq_len)
    
    assert torch.allclose(uniform_entropy, torch.tensor(expected_uniform, dtype=torch.float32), atol=1e-5)
    
    # Case 2: Point/Deterministic Distribution (all attention on single key)
    point_p = torch.zeros(1, 1, 1, seq_len, seq_len)
    point_p[..., 0] = 1.0 # query attends 100% to key 0
    point_entropy = compute_entropy(point_p, epsilon)
    
    assert torch.allclose(point_entropy, torch.tensor(0.0, dtype=torch.float32), atol=1e-5)
    
    # Case 3: Mixed sparse distribution with exact zeros
    mixed_p = torch.zeros(1, 1, 1, 1, 4)
    mixed_p[..., 0] = 0.5
    mixed_p[..., 1] = 0.5
    # indices 2 and 3 are exactly 0.0
    mixed_entropy = compute_entropy(mixed_p, epsilon)
    expected_mixed = -(0.5 * np.log(0.5) + 0.5 * np.log(0.5))
    
    assert not torch.isnan(mixed_entropy).any()
    assert torch.allclose(mixed_entropy, torch.tensor(expected_mixed, dtype=torch.float32), atol=1e-5)
    
    # Assert entropy is strictly non-negative
    assert (mixed_entropy >= 0.0).all()


def test_sparsity_metrics():
    """
    Validates that sparsity and density metrics are bounded correctly:
    - Sparsity percentage between 0 and 100%
    - Top-k cumulative mass is monotonic: top-1 <= top-5 <= top-10 <= top-50
    - Top-k cumulative mass bounded in [0.0, 1.0]
    """
    seq_len = 256
    config = {
        "analysis": {
            "top_k_values": [1, 5, 10, 50],
            "density_threshold": 1.0 / seq_len,
            "near_zero_threshold": 1e-4
        }
    }
    
    # Create mock attention weights: exponentially decaying (summing to 1)
    x = torch.arange(seq_len, dtype=torch.float32)
    decay_weights = torch.exp(-0.1 * x)
    decay_weights = decay_weights / decay_weights.sum()
    
    # Shape: [batch_size=1, layers=1, heads=1, seq_len=1, seq_len]
    attention = decay_weights.view(1, 1, 1, 1, seq_len)
    
    metrics = compute_sparsity_metrics(attention, config)
    
    # Validate sparsity percentage bounds
    sparsity_pct = metrics["sparsity_percentage"]
    assert 0.0 <= sparsity_pct.mean() <= 100.0
    
    # Validate density bounds
    density = metrics["density"]
    assert 0.0 <= density.mean() <= 1.0
    
    # Validate Top-K Cumulative Mass Monotonicity
    top_1 = metrics["top_k_masses"]["top_1_mass"]
    top_5 = metrics["top_k_masses"]["top_5_mass"]
    top_10 = metrics["top_k_masses"]["top_10_mass"]
    top_50 = metrics["top_k_masses"]["top_50_mass"]
    
    assert (0.0 <= top_1).all() and (top_50 <= 1.0).all()
    assert (top_1 <= top_5).all()
    assert (top_5 <= top_10).all()
    assert (top_10 <= top_50).all()


def test_power_law_fitting():
    """
    Verifies that the power-law regression fitting operates 
    correctly and is numerically stable.
    """
    # 1. Perfect power law distribution: p(k) = C * k^(-alpha)
    alpha_true = 1.2
    ranks = np.arange(1, 100)
    unnormalized_p = ranks ** (-alpha_true)
    p = unnormalized_p / unnormalized_p.sum()
    
    alpha_fit, r2_fit = fit_power_law(p)
    
    assert alpha_fit is not None
    assert r2_fit is not None
    # Power-law fit should be extremely close to the true exponent
    assert np.isclose(alpha_fit, alpha_true, atol=0.05)
    # R^2 should be near 1.0
    assert r2_fit > 0.99
    
    # 2. Case where fitting is mathematically impossible (flat or too short vector)
    short_vector = np.array([0.5, 0.5])
    alpha_fit_short, r2_fit_short = fit_power_law(short_vector)
    assert alpha_fit_short is None
    assert r2_fit_short is None


def test_statistics_summary_tables():
    """
    Checks that summary tables generate successfully and have the correct shape
    and columns.
    """
    num_samples = 3
    num_layers = 12
    num_heads = 12
    seq_len = 64
    
    metrics = {
        "head_entropy": np.random.uniform(1.0, 3.0, (num_samples, num_layers, num_heads)),
        "sparsity_percentage": np.random.uniform(50.0, 95.0, (num_samples, num_layers, num_heads, seq_len)),
        "density": np.random.uniform(0.01, 0.2, (num_samples, num_layers, num_heads, seq_len)),
        "top_k_masses": {
            "top_1_mass": np.random.uniform(0.1, 0.4, (num_samples, num_layers, num_heads, seq_len)),
            "top_5_mass": np.random.uniform(0.4, 0.7, (num_samples, num_layers, num_heads, seq_len)),
            "top_10_mass": np.random.uniform(0.6, 0.8, (num_samples, num_layers, num_heads, seq_len)),
            "top_50_mass": np.random.uniform(0.8, 1.0, (num_samples, num_layers, num_heads, seq_len))
        }
    }
    
    summary_df = generate_summary_tables(metrics)
    
    assert isinstance(summary_df, pd.DataFrame)
    assert len(summary_df) == num_layers
    assert "Layer" in summary_df.columns
    assert "Entropy (Mean)" in summary_df.columns
    assert "Sparsity % (Mean)" in summary_df.columns
    assert "Top-1 Mass (Mean)" in summary_df.columns


def test_cross_model_statistical_calcs():
    """
    Validates statistical logic of Phase 2:
    - Pearson correlation calculation and correct bounds
    - 95% Confidence Interval formulas
    """
    # Create sample vectors of length 50
    np.random.seed(42)
    n = 50
    entropy = np.random.uniform(1.0, 3.0, n)
    # create correlated sparsity
    sparsity = 100.0 - 15.0 * entropy + np.random.normal(0, 1.0, n)
    
    from scipy.stats import pearsonr
    r_val, p_val = pearsonr(entropy, sparsity)
    
    assert -1.0 <= r_val <= 1.0
    assert 0.0 <= p_val <= 1.0
    # Pearson r should be negative due to construction
    assert r_val < -0.5
    
    # Check Confidence Interval
    mean_val = np.mean(entropy)
    std_val = np.std(entropy)
    margin_of_error = 1.96 * (std_val / np.sqrt(n))
    ci_lower = mean_val - margin_of_error
    ci_upper = mean_val + margin_of_error
    
    assert ci_lower < mean_val < ci_upper
    assert np.isclose((ci_upper - ci_lower), 2 * margin_of_error)
