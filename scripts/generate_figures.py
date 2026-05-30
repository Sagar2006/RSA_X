import os
import argparse
import yaml
import logging
import numpy as np
from rsa_x.visualization.plots import PublicationVisualizer, apply_scientific_style

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("generate_figures")

def load_yaml_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="RSA-X: Standalone Publication Figure Generator"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/default_config.yaml",
        help="Path to YAML configuration file"
    )
    args = parser.parse_args()
    
    logger.info("Loading framework configuration...")
    try:
        config = load_yaml_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return
        
    results_dir = config["storage"]["results_dir"]
    metrics_dir = os.path.join(results_dir, config["storage"]["metrics_subdir"])
    raw_tensors_dir = os.path.join(results_dir, config["storage"]["raw_tensors_subdir"])
    
    npz_path = os.path.join(metrics_dir, "consolidated_metrics.npz")
    
    if not os.path.exists(npz_path):
        logger.error(f"Consolidated metrics file not found: {npz_path}. Please run main.py first.")
        return
        
    logger.info(f"Loading consolidated metrics from: {npz_path}...")
    npz_data = np.load(npz_path)
    
    # Reconstruct consolidated metrics dictionary
    k_values = config["analysis"]["top_k_values"]
    consolidated = {
        "head_entropy": npz_data["head_entropy"],
        "layer_entropy": npz_data["layer_entropy"],
        "token_entropy": npz_data["token_entropy"],
        "sparsity_percentage": npz_data["sparsity_percentage"],
        "density": npz_data["density"],
        "top_k_masses": {
            f"top_{k}_mass": npz_data[f"top_{k}_mass"]
            for k in k_values
        }
    }
    
    # Initialize Visualizer
    logger.info("Initializing PublicationVisualizer...")
    visualizer = PublicationVisualizer(config)
    
    # 1. Recreate Figure 1: Attention Heatmap (from first saved .npy raw file)
    logger.info("Checking for saved raw attention tensors to regenerate Figure 1...")
    if os.path.exists(raw_tensors_dir):
        npy_files = [f for f in os.listdir(raw_tensors_dir) if f.endswith(".npy")]
        if npy_files:
            target_npy = os.path.join(raw_tensors_dir, sorted(npy_files)[0])
            logger.info(f"Loading raw attention tensor from {target_npy} for Figure 1...")
            raw_pattern = np.load(target_npy) # [num_layers, num_heads, seq_len, seq_len]
            
            # Load corresponding tokens from JSON if available
            tokens_json = target_npy.replace(".npy", ".json")
            tokens = None
            if os.path.exists(tokens_json):
                with open(tokens_json, "r") as f:
                    meta = yaml.safe_load(f)
                    tokens = meta["metadata"]["tokens"]
            
            sample_idx = int(npy_files[0].split("_")[1].split(".")[0])
            
            # Plot Middle layer 5, middle head 5
            if tokens is None:
                tokens = [str(i) for i in range(raw_pattern.shape[2])]
                
            visualizer.plot_attention_heatmap(
                raw_pattern[5, 5], 
                tokens, 
                layer=5, 
                head=5, 
                sample_idx=sample_idx
            )
        else:
            logger.warning("No raw .npy tensors found. Skipping Figure 1 regeneration.")
    else:
        logger.warning("Raw tensors directory does not exist. Skipping Figure 1 regeneration.")
        
    # 2. Recreate Figure 2: Entropy Histogram
    logger.info("Regenerating Figure 2 (Entropy Histogram)...")
    seq_len = consolidated["token_entropy"].shape[3]
    max_entropy = np.log(seq_len)
    visualizer.plot_entropy_histogram(consolidated["token_entropy"], max_entropy)
    
    # 3. Recreate Figure 3: Sparsity Histogram
    logger.info("Regenerating Figure 3 (Sparsity Histogram)...")
    visualizer.plot_sparsity_histogram(consolidated["sparsity_percentage"])
    
    # 4. Recreate Figure 4: Top-k Mass Curve
    logger.info("Regenerating Figure 4 (Top-K Cumulative Mass Curve)...")
    visualizer.plot_top_k_curve(consolidated["top_k_masses"])
    
    # 5. Recreate Figure 5: Layer-wise Comparison
    logger.info("Regenerating Figure 5 (Layer-wise Entropy & Sparsity Boxplots)...")
    visualizer.plot_layerwise_comparison(
        consolidated["head_entropy"], 
        consolidated["sparsity_percentage"]
    )
    
    # 6. Recreate Figure 6: Head-wise Comparison
    logger.info("Regenerating Figure 6 (Head-wise 12x12 Heatmap Grids)...")
    visualizer.plot_headwise_comparison(
        consolidated["head_entropy"], 
        consolidated["sparsity_percentage"]
    )
    
    # 7. Recreate Figure 7: Attention Density Plot
    logger.info("Regenerating Figure 7 (Attention Density Curve)...")
    visualizer.plot_attention_density(consolidated["density"])
    
    logger.info(f"All figures successfully regenerated and saved to: {visualizer.figures_dir}")


if __name__ == "__main__":
    main()
