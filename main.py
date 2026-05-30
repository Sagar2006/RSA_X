import os
import argparse
import yaml
import logging
import datetime
import time
import json
import torch
from hardware import get_hardware_diagnostics, print_hardware_summary
from experiments.runner import ExperimentRunner

# We setup a basic logger initially, which we will reconfigure dynamically 
# once the isolated timestamped directories are established.
logger = logging.getLogger("rsa_x")

def load_yaml_config(config_path: str) -> dict:
    """Loads a YAML configuration from disk."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def merge_configs(base: dict, override: dict) -> dict:
    """Recursively merges override dictionary keys into the base dictionary."""
    for key, value in override.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            merge_configs(base[key], value)
        else:
            base[key] = value
    return base


def get_git_commit() -> str:
    """Retrieves the current git commit hash safely. Falls back if git is not initialized."""
    try:
        import subprocess
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        return commit
    except Exception:
        return "No Git repository detected / No commit hash available"


def main():
    parser = argparse.ArgumentParser(
        description="RSA-X: Reinforced Sparse Attention Experimental Framework (Phase 1.5)"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to override YAML configuration file (e.g. configs/local.yaml, configs/kaggle.yaml)"
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
    
    # 1. Configuration Cascading Merge
    default_config_path = os.path.join("configs", "default_config.yaml")
    
    # Load default baseline
    try:
        config = load_yaml_config(default_config_path)
    except Exception as e:
        print(f"CRITICAL: Failed to load baseline default configuration: {e}")
        return
        
    # Apply secondary environment profile configuration
    if args.config is not None:
        try:
            override_config = load_yaml_config(args.config)
            config = merge_configs(config, override_config)
            print(f"Successfully cascade-merged configuration from: {args.config}")
        except Exception as e:
            print(f"WARNING: Failed to load override configuration {args.config} ({e}). Proceeding with default configs.")
            
    # Apply direct CLI argument overrides
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
        
    # 2. Results Directory Timestamping Isolation (Enforce absolute project-root pathing)
    PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    run_folder = f"run_{timestamp}"
    run_dir = os.path.abspath(os.path.join(PROJECT_ROOT, "results", run_folder))
    
    # Create isolated folders
    figures_dir = os.path.join(run_dir, "figures")
    metrics_dir = os.path.join(run_dir, "metrics")
    logs_dir = os.path.join(run_dir, "logs")
    
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # Set run directory dynamically in config
    config["storage"]["results_dir"] = run_dir
    
    # 3. Dynamic Logging System Reconfiguration
    log_file = os.path.join(logs_dir, "experiment.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode="w", encoding="utf-8")
        ],
        force=True  # Force reconfiguration of baseline root logger
    )
    
    # 4. Print System Diagnostics
    logger.info("Initializing environment diagnostics...")
    print_hardware_summary()
    
    # Log starting parameters
    logger.info(f"Isolated run results directory: {run_dir}")
    logger.info(f"Target model: {config['model']['name']}")
    logger.info(f"Sequence length: {config['dataset']['max_seq_len']}")
    logger.info(f"Sample blocks count limit: {config['dataset']['num_samples']}")
    
    # 5. Execute Experiments & Measure Total Duration
    start_perf = time.perf_counter()
    try:
        runner = ExperimentRunner(config)
        runner.run_all_experiments()
        logger.info("RSA-X experimental run completed successfully.")
    except Exception as e:
        logger.critical(f"RSA-X encountered a fatal execution error: {e}", exc_info=True)
    finally:
        end_perf = time.perf_counter()
        exec_duration = end_perf - start_perf
        
        # 6. Save Run Metadata JSON
        logger.info("Generating final run metadata record...")
        metadata = {
            "timestamp": datetime.datetime.now().isoformat(),
            "git_commit_hash": get_git_commit(),
            "config_path": args.config if args.config else "default_config.yaml",
            "model_name": config["model"]["name"],
            "dataset_name": config["dataset"]["name"],
            "hardware_information": get_hardware_diagnostics(),
            "execution_time_seconds": round(exec_duration, 4)
        }
        
        metadata_file = os.path.join(run_dir, "experiment_metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Metadata recorded at: {metadata_file}")
        logger.info(f"Total execution duration: {round(exec_duration, 2)} seconds.")


if __name__ == "__main__":
    main()
