import argparse
import yaml
import logging
import numpy as np
from pathlib import Path
from paths import PathManager
from visualization.plots import PublicationVisualizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("generate_figures")


def load_yaml_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def find_latest_run_dir(results_root: Path) -> Path:
    """Finds the most recently modified run_ subdirectory in results/."""
    if not results_root.exists():
        return None
    subdirs = [
        d for d in results_root.iterdir()
        if d.is_dir() and d.name.startswith("run_")
    ]
    if not subdirs:
        return None
    # Sort subdirectories by modification time, most recent first
    subdirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return subdirs[0]


def main():
    parser = argparse.ArgumentParser(
        description="RSA-X: Standalone Publication Figure Generator (Phase 1.5)"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/default_config.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        help="Target isolated run directory (e.g. results/run_2026_05_30_16_42_54). If omitted, loads the latest run."
    )
    args = parser.parse_args()
    
    logger.info("Loading framework configuration...")
    PROJECT_ROOT = PathManager.get_project_root()
    
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
        
    try:
        config = load_yaml_config(config_path)
    except Exception as e:
        logger.error(f"Failed to load baseline configuration: {e}")
        return
        
    target_run_str = args.run_dir
    results_root = PathManager.get_results_dir()
    
    if not target_run_str:
        logger.info("No --run_dir specified. Scanning for the latest experimental run...")
        target_run = find_latest_run_dir(results_root)
        if not target_run:
            logger.error("No 'results/run_*' subdirectories found. Please execute main.py first.")
            return
        logger.info(f"Automatically identified latest run folder: {target_run}")
    else:
        target_run = Path(target_run_str)
        if not target_run.is_absolute():
            target_run = PROJECT_ROOT / target_run
        if not target_run.exists():
            logger.error(f"Target run directory does not exist: {target_run}")
            return
            
    # Apply dynamic directory overrides in config
    config["storage"]["results_dir"] = str(target_run)
    
    metrics_dir = target_run / config["storage"]["metrics_subdir"]
    raw_tensors_dir = target_run / config["storage"]["raw_tensors_subdir"]
    
    npz_path = metrics_dir / "consolidated_metrics.npz"
    
    if not npz_path.exists():
        logger.error(f"Consolidated metrics file not found: {npz_path}.")
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
    
    # 1. Recreate Figure 1: Attention Heatmap (from first saved .npz raw file)
    logger.info("Checking for saved raw attention tensors to regenerate Figure 1...")
    if raw_tensors_dir.exists():
        npz_files = [f for f in raw_tensors_dir.iterdir() if f.name.endswith(".npz")]
        if npz_files:
            target_npz = sorted(npz_files)[0]
            logger.info(f"Loading raw attention tensor from {target_npz} for Figure 1...")
            sample_data = np.load(target_npz)
            raw_pattern = sample_data["attention"] # [num_layers, num_heads, seq_len, seq_len]
            tokens = list(sample_data["tokens"])
            
            sample_idx = int(target_npz.stem.split("_")[1])
            
            visualizer.plot_attention_heatmap(
                raw_pattern[5, 5], 
                tokens, 
                layer=5, 
                head=5, 
                sample_idx=sample_idx
            )
        else:
            logger.warning("No raw .npz tensors found. Skipping Figure 1 regeneration.")
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
    
    # 6. Recreate Figure 6: Head-wise 12x12 Heatmap Grids (PNG + PDF)
    logger.info("Regenerating Figure 6 (Head-wise 12x12 Heatmap Grids)...")
    visualizer.plot_headwise_comparison(
        consolidated["head_entropy"], 
        consolidated["sparsity_percentage"]
    )
    
    # 7. Recreate Figure 7: Attention Density Plot
    logger.info("Regenerating Figure 7 (Attention Density Curve)...")
    visualizer.plot_attention_density(consolidated["density"])
    
    logger.info(f"All figures successfully regenerated (PNG + PDF format) and saved to: {visualizer.figures_dir}")
    print("\n--- STANDALONE TIMING PROFILE REPORT ---")
    for name, t_dict in visualizer.figure_timings.items():
        print(f"Figure: {name:<25} | Creation: {t_dict['creation_time']:.4f}s | PNG Save: {t_dict['save_png_time']:.4f}s | PDF Save: {t_dict['save_pdf_time']:.4f}s")
    print("------------------------------------------\n")


if __name__ == "__main__":
    main()
