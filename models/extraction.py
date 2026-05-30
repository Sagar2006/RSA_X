import os
import json
import logging
import torch
import numpy as np
import pandas as pd
from transformer_lens import HookedTransformer

logger = logging.getLogger(__name__)

class AttentionExtractor:
    """
    GPT-2 Attention Extraction Engine.
    Loads a pretrained model via TransformerLens and extracts full attention
    patterns across all layers and heads using cached activations.
    """
    def __init__(self, config: dict):
        self.config = config
        self.model_name = config["model"]["name"]
        
        # Determine device
        device_cfg = config["model"]["device"]
        if device_cfg == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device_cfg
            
        logger.info(f"Loading HookedTransformer '{self.model_name}' on {self.device}...")
        self.model = HookedTransformer.from_pretrained(
            self.model_name,
            device=self.device
        )
        self.model.eval() # Ensure evaluation mode
        
        # Cache dimensions
        self.num_layers = self.model.cfg.n_layers
        self.num_heads = self.model.cfg.n_heads
        
        # Setup paths
        self.results_dir = config["storage"]["results_dir"]
        self.raw_tensors_dir = os.path.join(self.results_dir, config["storage"]["raw_tensors_subdir"])
        os.makedirs(self.raw_tensors_dir, exist_ok=True)

    def extract_batch(self, batch_ids: torch.Tensor) -> torch.Tensor:
        """
        Extracts attention patterns for a batch.
        
        Args:
            batch_ids (torch.Tensor): Tensor of shape [batch_size, seq_len].
            
        Returns:
            torch.Tensor: Attention patterns of shape [batch_size, num_layers, num_heads, seq_len, seq_len].
        """
        batch_ids = batch_ids.to(self.device)
        seq_len = batch_ids.shape[1]
        batch_size = batch_ids.shape[0]
        
        # Hook activation pattern filter (prevents heavy MLP/residual cache bloat)
        pattern_filter = lambda name: name.endswith("hook_pattern")
        
        with torch.no_grad():
            _, cache = self.model.run_with_cache(
                batch_ids,
                names_filter=pattern_filter
            )
            
        # Collect patterns from cache into a single stacked tensor
        # Cache entries are of shape [batch_size, num_heads, seq_len, seq_len]
        batch_patterns = []
        for layer in range(self.num_layers):
            pattern_key = f"blocks.{layer}.attn.hook_pattern"
            if pattern_key in cache:
                batch_patterns.append(cache[pattern_key].cpu())
            else:
                raise KeyError(f"Hook pattern for layer {layer} not found in activation cache.")
                
        # Stack layers: [num_layers, batch_size, num_heads, seq_len, seq_len]
        stacked = torch.stack(batch_patterns, dim=0)
        # Permute to: [batch_size, num_layers, num_heads, seq_len, seq_len]
        stacked = stacked.permute(1, 0, 2, 3, 4)
        
        return stacked

    def save_raw_attention(self, pattern: torch.Tensor, tokens: list, sample_idx: int) -> dict:
        """
        Saves raw attention tensors and tokens for a single sample in a single, 
        highly optimized compressed NumPy format (.npz) to minimize storage footprint.
        
        Args:
            pattern (torch.Tensor): Pattern tensor of shape [num_layers, num_heads, seq_len, seq_len].
            tokens (list): Token string representation list of length seq_len.
            sample_idx (int): Unique identifier of the sample.
            
        Returns:
            dict: Dictionary containing file size details.
        """
        seq_len = pattern.shape[2]
        pattern_np = pattern.numpy().astype(np.float32)
        
        # Base file path
        base_path = os.path.join(self.raw_tensors_dir, f"sample_{sample_idx}")
        npz_file = f"{base_path}.npz"
        
        # Save both attention matrix and token strings in a single compressed NPZ binary file
        import time
        logger.info(f"save_start: Saving raw attention for sample {sample_idx}...")
        start_time = time.perf_counter()
        
        np.savez_compressed(
            npz_file,
            attention=pattern_np,
            tokens=np.array(tokens)
        )
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        logger.info(f"save_end: Saved raw attention for sample {sample_idx}.")
        logger.info(f"save_duration_seconds: {duration:.4f} seconds for sample {sample_idx}.")
        
        return {
            "npz_size_bytes": os.path.getsize(npz_file),
            "total_edges": seq_len * seq_len * self.num_layers * self.num_heads
        }
