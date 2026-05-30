import os
import logging
import numpy as np
import torch
import pandas as pd
from rsa_x.datasets.loader import get_dataset_loader
from rsa_x.models.extraction import AttentionExtractor
from rsa_x.analysis.entropy import EntropyAnalyzer
from rsa_x.analysis.sparsity import SparsityAnalyzer
from rsa_x.analysis.statistics import generate_summary_tables, fit_power_law
from rsa_x.visualization.plots import PublicationVisualizer
from experiments.tracker import ExperimentTracker

logger = logging.getLogger(__name__)

class ExperimentRunner:
    """
    Central Runner for RSA-X Phase 1.
    Orchestrates dataset tokenization, attention extraction, analysis,
    statistical calculations, visualization, and logging for all 3 experiments.
    """
    def __init__(self, config: dict):
        self.config = config
        self.results_dir = config["storage"]["results_dir"]
        self.metrics_dir = os.path.join(self.results_dir, config["storage"]["metrics_subdir"])
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        # Initialize sub-systems
        self.extractor = AttentionExtractor(config)
        self.entropy_analyzer = EntropyAnalyzer(config)
        self.sparsity_analyzer = SparsityAnalyzer(config)
        self.visualizer = PublicationVisualizer(config)
        self.tracker = ExperimentTracker(config)
        
        self.num_layers = self.extractor.num_layers
        self.num_heads = self.extractor.num_heads

    def run_all_experiments(self):
        """
        Runs the full Phase 1 experimental suite.
        - Loads WikiText-103
        - Performs batched attention extraction & analysis
        - Runs Experiment 1 (Sparsity), 2 (Top-k), 3 (Layer-wise)
        - Computes power law distribution statistics
        - Generates 7 publication figures and research tables
        """
        start_time = pd.Timestamp.now()
        logger.info(f"RSA-X Experiment Start Time: {start_time}")
        logger.info("Starting RSA-X Phase 1 Scientific Experiments...")
        
        # 1. Load dataset pipeline
        loader = get_dataset_loader(self.config)
        
        # Accumulators for overall metrics
        all_head_entropies = []
        all_layer_entropies = []
        all_token_entropies = []
        all_sparsity_percentages = []
        all_densities = []
        
        # Initialize Top-K accumulators
        k_values = self.config["analysis"]["top_k_values"]
        all_top_k_masses = {f"top_{k}_mass": [] for k in k_values}
        
        sample_count = 0
        save_raw_limit = self.config["storage"]["save_raw_samples"]
        
        # Cache for specific sample attention maps to visualize
        visualize_sample_pattern = None
        visualize_sample_tokens = None
        visualize_sample_idx = None
        
        # 2. Extraction & Analysis Loop
        logger.info("Running batched model evaluation and attention profiling...")
        for batch_idx, batch in enumerate(loader):
            input_ids = batch["input_ids"]
            batch_size = input_ids.shape[0]
            
            # Extract attention tensors: [batch_size, num_layers, num_heads, seq_len, seq_len]
            patterns = self.extractor.extract_batch(input_ids)
            
            # Analyze each batch
            entropy_metrics = self.entropy_analyzer.analyze(patterns)
            sparsity_metrics = self.sparsity_analyzer.analyze(patterns)
            
            # Accumulate batch metrics
            all_head_entropies.append(entropy_metrics["head_entropy"]) # [batch_size, layers, heads]
            all_layer_entropies.append(entropy_metrics["layer_entropy"]) # [batch_size, layers]
            all_token_entropies.append(entropy_metrics["token_entropy"]) # [batch_size, layers, heads, seq_len]
            all_sparsity_percentages.append(sparsity_metrics["sparsity_percentage"]) # [batch_size, layers, heads]
            all_densities.append(sparsity_metrics["density"]) # [batch_size, layers, heads]
            
            for k in k_values:
                all_top_k_masses[f"top_{k}_mass"].append(sparsity_metrics["top_k_masses"][f"top_{k}_mass"])
                
            # Process individual samples in the batch for saving raw data
            for item_idx in range(batch_size):
                global_idx = sample_count
                
                # Retrieve text tokens using HookedTransformer utility
                token_ids = input_ids[item_idx]
                tokens_str = self.extractor.model.to_str_tokens(token_ids)
                
                # Check if we should save raw attention patterns (Hard storage-efficient limit: max 3 samples)
                if global_idx < min(save_raw_limit, 3):
                    logger.info(f"Saving raw attention weights for sample {global_idx}...")
                    # Extract sample pattern: [num_layers, num_heads, seq_len, seq_len]
                    sample_pattern = patterns[item_idx]
                    self.extractor.save_raw_attention(sample_pattern, tokens_str, global_idx)
                    
                    # Keep one sample for figure 1 visualization (middle layer, middle head)
                    if global_idx == 0:
                        visualize_sample_pattern = sample_pattern.numpy()
                        visualize_sample_tokens = tokens_str
                        visualize_sample_idx = global_idx
                        
                sample_count += 1
                
        # 3. Consolidate and aggregate metrics over the dataset
        logger.info("Consolidating dataset-wide scientific statistics...")
        consolidated = {
            "head_entropy": np.concatenate(all_head_entropies, axis=0),
            "layer_entropy": np.concatenate(all_layer_entropies, axis=0),
            "token_entropy": np.concatenate(all_token_entropies, axis=0),
            "sparsity_percentage": np.concatenate(all_sparsity_percentages, axis=0),
            "density": np.concatenate(all_densities, axis=0),
            "top_k_masses": {
                f"top_{k}_mass": np.concatenate(all_top_k_masses[f"top_{k}_mass"], axis=0)
                for k in k_values
            }
        }
        
        # Save consolidated NumPy metrics for offline figure regeneration
        npz_path = os.path.join(self.metrics_dir, "consolidated_metrics.npz")
        np.savez_compressed(
            npz_path,
            head_entropy=consolidated["head_entropy"],
            layer_entropy=consolidated["layer_entropy"],
            token_entropy=consolidated["token_entropy"],
            sparsity_percentage=consolidated["sparsity_percentage"],
            density=consolidated["density"],
            **{f"top_{k}_mass": consolidated["top_k_masses"][f"top_{k}_mass"] for k in k_values}
        )
        logger.info(f"Saved dataset consolidated metrics as compressed NPZ: {npz_path}")
        
        # 4. Generate Research Tables (Module 5)
        logger.info("Generating scientific layer-wise comparison tables...")
        summary_table = generate_summary_tables(consolidated)
        summary_csv = os.path.join(self.metrics_dir, "layerwise_summary_table.csv")
        summary_table.to_csv(summary_csv, index=False)
        logger.info(f"Research summary table saved to: {summary_csv}")
        print("\n=== LAYER-WISE SPARSITY & ENTROPY RESEARCH SUMMARY ===")
        print(summary_table.to_string(index=False))
        print("========================================================\n")
        
        # 5. Fit Power-Law Distribution (Power-law exploration)
        logger.info("Fitting Power-Law concentration models to attention tails...")
        # Fit on layer 5, head 5 (middle layer representation)
        sample_entropy_token = consolidated["token_entropy"][:, 5, 5, :].mean(axis=0) # [seq_len]
        # We can extract a representative attention weight vector from our sample pattern (Sample 0, Layer 5, Head 5, middle query position)
        if visualize_sample_pattern is not None:
            middle_pos = len(visualize_sample_tokens) // 2
            rep_weights = visualize_sample_pattern[5, 5, middle_pos, :] # [seq_len]
            alpha, r_squared = fit_power_law(rep_weights)
            logger.info(f"Power-law Fit on (L5, H5, query_idx={middle_pos}): Alpha = {alpha:.4f}, R² = {r_squared:.4f}")
            self.tracker.log_metrics({
                "power_law_alpha_L5_H5": alpha if alpha is not None else 0.0,
                "power_law_r2_L5_H5": r_squared if r_squared is not None else 0.0
            })
            
        # 6. Run Experiment-Specific Logging (Modules 6 & 7)
        # Experiment 1: Natural Sparsity Measurement
        mean_sparsity = consolidated["sparsity_percentage"].mean().item()
        mean_entropy = consolidated["head_entropy"].mean().item()
        mean_density = consolidated["density"].mean().item()
        
        self.tracker.log_metrics({
            "dataset_mean_sparsity_pct": mean_sparsity,
            "dataset_mean_entropy_nat": mean_entropy,
            "dataset_mean_density": mean_density
        })
        
        # Experiment 2: Top-k Attention Concentration Analysis
        for k in k_values:
            k_mass_mean = consolidated["top_k_masses"][f"top_{k}_mass"].mean().item()
            self.tracker.log_metrics({f"dataset_mean_top_{k}_mass": k_mass_mean})
            
        # 7. Generate Figures (Module 4)
        logger.info("Generating publication-quality 300 DPI figures...")
        # Figure 1: Attention Heatmap (selected Layer 5, Head 5 of first sample)
        if visualize_sample_pattern is not None:
            # Middle layer 5, head 5 attention map
            self.visualizer.plot_attention_heatmap(
                visualize_sample_pattern[5, 5], 
                visualize_sample_tokens, 
                layer=5, 
                head=5, 
                sample_idx=visualize_sample_idx
            )
            
        # Figure 2: Entropy Histogram
        self.visualizer.plot_entropy_histogram(
            consolidated["token_entropy"], 
            entropy_metrics["max_entropy"]
        )
        
        # Figure 3: Sparsity Histogram
        self.visualizer.plot_sparsity_histogram(consolidated["sparsity_percentage"])
        
        # Figure 4: Top-k Mass Curve
        self.visualizer.plot_top_k_curve(consolidated["top_k_masses"])
        
        # Figure 5: Layer-wise Comparison
        self.visualizer.plot_layerwise_comparison(
            consolidated["head_entropy"], 
            consolidated["sparsity_percentage"]
        )
        
        # Figure 6: Head-wise Grid Heatmaps
        self.visualizer.plot_headwise_comparison(
            consolidated["head_entropy"], 
            consolidated["sparsity_percentage"]
        )
        
        # Figure 7: Attention Density Distribution Plot
        self.visualizer.plot_attention_density(consolidated["density"])
        
        # 8. Log Figures to W&B Tracker
        logger.info("Logging publication plots to experiment dashboard...")
        for file in os.listdir(self.visualizer.figures_dir):
            if file.endswith(self.visualizer.img_format):
                fig_path = os.path.join(self.visualizer.figures_dir, file)
                # Parse figure label name from filename e.g. "fig2_entropy_histogram"
                fig_name = file.split(".")[0]
                self.tracker.log_figure(fig_name, fig_path)
                
        # 9. Clean up and finalize
        self.tracker.finish()
        end_time = pd.Timestamp.now()
        logger.info(f"RSA-X Experiment End Time: {end_time}")
        logger.info(f"Execution Summary: Processed {sample_count} WikiText sample blocks. Model: {self.extractor.model_name}. Mean Sparsity: {mean_sparsity:.4f}%, Mean Entropy: {mean_entropy:.4f} Nat, Mean Density: {mean_density:.4f}.")
        logger.info("RSA-X Phase 1 Scientific Experiments completed successfully.")
        
        return {
            "mean_sparsity": mean_sparsity,
            "mean_entropy": mean_entropy,
            "mean_density": mean_density,
            "summary_table": summary_table
        }
