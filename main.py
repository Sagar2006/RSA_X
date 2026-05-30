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


def get_clean_project_root(start_dir: str) -> str:
    """
    Cleans and resolves the absolute project root path by detecting 
    and eliminating repeated nested/duplicate project root folder name occurrences 
    (such as RSA_X/RSA_X/RSA_X) caused by repeated cell executions or git clones.
    """
    drive, tail = os.path.splitdrive(os.path.abspath(start_dir))
    norm_path = tail.replace("\\", "/")
    path_parts = [p for p in norm_path.split("/") if p]
    
    seen_root_name = False
    cleaned_parts = []
    for part in path_parts:
        is_root_name = part.lower() in ("rsa_x", "rsa-x")
        if is_root_name:
            if seen_root_name:
                continue
            seen_root_name = True
        cleaned_parts.append(part)
        
    reconstructed = drive + "/" + "/".join(cleaned_parts) if drive else "/" + "/".join(cleaned_parts)
    return os.path.abspath(reconstructed)


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
    current_file_dir = os.path.abspath(os.path.dirname(__file__))
    PROJECT_ROOT = get_clean_project_root(current_file_dir)
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    run_folder = f"run_{timestamp}"
    run_dir = get_clean_project_root(os.path.abspath(os.path.join(PROJECT_ROOT, "results", run_folder)))
    
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
    
    # 4. Hardware-Aware Execution Engine
    logger.info("Initializing Hardware-Aware Execution Engine...")
    diagnostics = get_hardware_diagnostics()
    cuda_available = diagnostics["cuda_available"]
    ram_gb = diagnostics.get("ram_gb", 16.0)
    gpu_vram_gb = diagnostics.get("gpu_vram_gb", 0.0)
    
    # Check if research mode is explicitly requested via config or arguments
    is_research = config.get("research_mode", False) or config["storage"].get("research_mode", False) or (args.config and "research" in args.config.lower())
    
    # Detect Kaggle automatically
    is_kaggle = "KAGGLE_KERNEL_RUN_TYPE" in os.environ or "KAGGLE_CONTAINER_NAME" in os.environ or os.path.exists("/kaggle")
    
    if is_research:
        selected_mode = "RESEARCH_MODE"
    elif cuda_available and is_kaggle:
        selected_mode = "GPU_MODE"
    elif cuda_available and gpu_vram_gb >= 8.0:
        selected_mode = "GPU_MODE"
    elif ram_gb >= 16.0:
        selected_mode = "STANDARD_MODE"
    else:
        selected_mode = "LOW_RESOURCE_MODE"
        
    logger.info(f"Hardware Engine selected mode: {selected_mode}")
    
    # Apply mode-specific settings overrides
    if selected_mode == "LOW_RESOURCE_MODE":
        config["dataset"]["num_samples"] = 10
        config["dataset"]["max_seq_len"] = 128
        config["dataset"]["batch_size"] = 1
        config["storage"]["save_raw_samples"] = 0
        config["storage"]["save_raw_attention"] = False
        config["storage"]["save_debug_tensors"] = False
        config["analysis"]["density_threshold"] = 1.0 / 128.0
    elif selected_mode == "STANDARD_MODE":
        config["dataset"]["num_samples"] = 50
        config["dataset"]["max_seq_len"] = 256
        config["dataset"]["batch_size"] = 1
        config["storage"]["save_raw_samples"] = 0
        config["storage"]["save_raw_attention"] = False
        config["analysis"]["density_threshold"] = 1.0 / 256.0
    elif selected_mode == "GPU_MODE":
        config["dataset"]["num_samples"] = 500
        config["dataset"]["max_seq_len"] = 512
        config["storage"]["save_raw_samples"] = 0
        config["storage"]["save_raw_attention"] = False
        config["analysis"]["density_threshold"] = 1.0 / 512.0
        
        # Automatic Batch Size Selection
        if config["dataset"].get("batch_size") == "auto" or config["dataset"].get("batch_size") is None or args.batch_size is None:
            if gpu_vram_gb <= 4.0:
                config["dataset"]["batch_size"] = 1
            elif gpu_vram_gb <= 8.0:
                config["dataset"]["batch_size"] = 2
            elif gpu_vram_gb <= 16.0:
                config["dataset"]["batch_size"] = 4
            else:
                config["dataset"]["batch_size"] = 8
    elif selected_mode == "RESEARCH_MODE":
        # Keep config defaults but allow limited raw archival
        config["storage"]["save_raw_attention"] = config["storage"].get("save_raw_attention", True)
        
    # Enforce Automatic Storage Policy & Safety Constraints
    debug_mode = config.get("debug_mode", False) or config["storage"].get("debug_mode", False)
    
    if debug_mode:
        config["storage"]["save_raw_samples"] = 3
    elif selected_mode == "RESEARCH_MODE":
        config["storage"]["save_raw_samples"] = min(config["storage"].get("save_raw_samples", 3), 3)
    else:
        config["storage"]["save_raw_samples"] = 0
        
    # Kaggle-specific override
    if is_kaggle:
        logger.info("Enforcing Kaggle storage optimizations...")
        config["storage"]["save_raw_samples"] = min(config["storage"].get("save_raw_samples", 0), 3)
        if cuda_available:
            config["model"]["device"] = "cuda"
            
    # Force device based on selected mode
    if selected_mode == "GPU_MODE" or (selected_mode == "RESEARCH_MODE" and cuda_available):
        config["model"]["device"] = "cuda"
    elif selected_mode == "LOW_RESOURCE_MODE" or selected_mode == "STANDARD_MODE":
        config["model"]["device"] = "cpu"
        
    # Write hardware_profile.json
    hardware_profile = {
        "cpu_model": diagnostics["cpu"],
        "cpu_core_count": diagnostics["cpu_core_count"],
        "available_ram": diagnostics["ram"],
        "cuda_available": diagnostics["cuda_available"],
        "gpu_name": diagnostics["gpu_name"],
        "gpu_vram": diagnostics["gpu_vram"],
        "available_disk_space": diagnostics["available_disk_space"],
        "execution_mode": selected_mode
    }
    
    profile_path = os.path.join(run_dir, "hardware_profile.json")
    with open(profile_path, "w") as f:
        json.dump(hardware_profile, f, indent=2)
    logger.info(f"Hardware profile generated successfully at: {profile_path}")
    
    print_hardware_summary()
    
    # Log starting parameters
    logger.info(f"Isolated run results directory: {run_dir}")
    logger.info(f"Target model: {config['model']['name']}")
    logger.info(f"Sequence length: {config['dataset']['max_seq_len']}")
    logger.info(f"Sample blocks count limit: {config['dataset']['num_samples']}")
    logger.info(f"Active Device: {config['model']['device']}")
    logger.info(f"Batch Size: {config['dataset']['batch_size']}")
    logger.info(f"Save Raw Samples Limit: {config['storage']['save_raw_samples']}")
    
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
