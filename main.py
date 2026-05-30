import os
import argparse
import yaml
import logging
import torch
from experiments.runner import ExperimentRunner

# Define logging format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("rsa_x_execution.log", mode="w", encoding="utf-8")
    ]
)
logger = logging.getLogger("rsa_x")

def load_yaml_config(config_path: str) -> dict:
    """Loads YAML configuration from disk."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="RSA-X: Reinforced Sparse Attention Experimental Framework (Phase 1)"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/default_config.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        help="Override total number of samples to process"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        help="Override token evaluation batch size"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        choices=["cpu", "cuda", "auto"],
        help="Override target execution device (cpu, cuda, auto)"
    )
    parser.add_argument(
        "--wandb_mode", 
        type=str, 
        choices=["online", "offline"],
        help="Override Weights & Biases tracking mode (online, offline)"
    )
    parser.add_argument(
        "--save_raw_samples", 
        type=int, 
        help="Override number of samples to save full raw attention matrices for"
    )
    
    args = parser.parse_args()
    
    logger.info("Loading framework configuration...")
    try:
        config = load_yaml_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return
        
    # Apply CLI overrides
    if args.num_samples is not None:
        config["dataset"]["num_samples"] = args.num_samples
    if args.batch_size is not None:
        config["dataset"]["batch_size"] = args.batch_size
    if args.device is not None:
        config["model"]["device"] = args.device
    if args.wandb_mode is not None:
        config["wandb"]["mode"] = args.wandb_mode
    if args.save_raw_samples is not None:
        config["storage"]["save_raw_samples"] = args.save_raw_samples
        
    # Resolve dynamic device
    dev = config["model"]["device"]
    if dev == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved_device = dev
    logger.info(f"Target system execution device: {resolved_device}")
    
    # Initialize and execute Runner
    try:
        runner = ExperimentRunner(config)
        runner.run_all_experiments()
        logger.info("RSA-X experimental run completed successfully.")
    except Exception as e:
        logger.critical(f"RSA-X encountered a fatal execution error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
