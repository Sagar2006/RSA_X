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
                batch_patterns.append(cache[pattern_key])
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
            dict: Dictionary containing file size details and fine-grained timings.
        """
        import time
        import io
        
        logger.info(f"save_start: Saving raw attention for sample {sample_idx}...")
        
        # 1. Measure Tensor Transfer GPU -> CPU
        transfer_start = time.perf_counter()
        pattern_cpu = pattern.cpu()
        transfer_end = time.perf_counter()
        transfer_duration = transfer_end - transfer_start
        
        seq_len = pattern.shape[2]
        pattern_np = pattern_cpu.numpy().astype(np.float32)
        
        # Base file path
        base_path = os.path.join(self.raw_tensors_dir, f"sample_{sample_idx}")
        npz_file = f"{base_path}.npz"
        
        # 2. Measure Compression time (using an in-memory buffer)
        compress_start = time.perf_counter()
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            attention=pattern_np,
            tokens=np.array(tokens)
        )
        buffer.seek(0)
        compressed_data = buffer.read()
        compress_end = time.perf_counter()
        compression_duration = compress_end - compress_start
        
        # 3. Measure Disk Write time
        write_start = time.perf_counter()
        with open(npz_file, "wb") as f:
            f.write(compressed_data)
        write_end = time.perf_counter()
        disk_write_duration = write_end - write_start
        
        duration = transfer_duration + compression_duration + disk_write_duration
        logger.info(f"save_end: Saved raw attention for sample {sample_idx}.")
        logger.info(f"save_duration_seconds: {duration:.4f} seconds for sample {sample_idx}.")
        
        return {
            "npz_size_bytes": os.path.getsize(npz_file),
            "total_edges": seq_len * seq_len * self.num_layers * self.num_heads,
            "timings": {
                "tensor_transfer": transfer_duration,
                "compression": compression_duration,
                "disk_write": disk_write_duration
            }
        }
