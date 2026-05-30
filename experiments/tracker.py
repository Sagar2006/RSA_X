import os
import json
import logging
import pandas as pd
import wandb

logger = logging.getLogger(__name__)

class ExperimentTracker:
    """
    Experiment Tracker for RSA-X.
    Wraps Weights & Biases (W&B) logging with robust local fallbacks 
    when running in offline environments or without active accounts.
    """
    def __init__(self, config: dict):
        self.config = config
        self.project = config["wandb"]["project"]
        self.group = config["wandb"]["group"]
        self.mode = config["wandb"]["mode"]
        
        self.results_dir = config["storage"]["results_dir"]
        self.metrics_dir = os.path.join(self.results_dir, config["storage"]["metrics_subdir"])
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        # Local metrics storage path (CSV and JSON)
        self.local_metrics_csv = os.path.join(self.metrics_dir, "local_metrics_log.csv")
        self.local_metrics = []
        
        self.initialized = False
        self.setup_wandb()

    def setup_wandb(self):
        """
        Initializes the W&B run. Gracefully catches exceptions and 
        falls back to offline/local tracking if credentials or networks fail.
        """
        try:
            logger.info(f"Initializing W&B run (Project: {self.project}, Group: {self.group}, Mode: {self.mode})...")
            
            # If API key is not present, configure wandb to run offline to avoid blocking terminal
            if self.mode == "offline":
                os.environ["WANDB_MODE"] = "offline"
                
            wandb.init(
                project=self.project,
                group=self.group,
                config=self.config,
                mode=self.mode
            )
            self.initialized = True
            logger.info("W&B initialized successfully.")
        except Exception as e:
            logger.warning(
                f"W&B initialization failed ({e}). "
                "Running in Pure Local File Logging Mode. No metrics will be uploaded."
            )
            os.environ["WANDB_MODE"] = "disabled"
            self.initialized = False

    def log_metrics(self, metrics: dict, step: int = None):
        """
        Logs a dictionary of scalar metrics.
        
        Args:
            metrics (dict): Key-value dictionary of numeric metrics.
            step (int, optional): The current training/eval step.
        """
        # Save to local tracking list
        log_entry = {"step": step} if step is not None else {}
        log_entry.update(metrics)
        self.local_metrics.append(log_entry)
        
        # Log to W&B
        if self.initialized:
            try:
                wandb.log(metrics, step=step)
            except Exception as e:
                logger.error(f"Error logging to W&B: {e}")
                
        # Proactively dump to local disk
        self._flush_local_metrics()

    def log_figure(self, name: str, figure_path: str):
        """
        Logs an image figure to the experiment tracker.
        
        Args:
            name (str): Label for the figure in the dashboard.
            figure_path (str): File path to the saved figure image.
        """
        if self.initialized and os.path.exists(figure_path):
            try:
                wandb.log({name: wandb.Image(figure_path)})
                logger.info(f"Logged figure '{name}' ({figure_path}) to W&B dashboard.")
            except Exception as e:
                logger.error(f"Error logging image to W&B: {e}")

    def finish(self):
        """
        Closes tracking and saves all local metadata logs.
        """
        self._flush_local_metrics()
        
        # Save config dump
        config_path = os.path.join(self.metrics_dir, "metadata_config.json")
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2)
            
        logger.info(f"Finalized and stored metadata config: {config_path}")
        
        if self.initialized:
            try:
                wandb.finish()
                logger.info("W&B run finished.")
            except Exception as e:
                logger.error(f"Error finishing W&B run: {e}")

    def _flush_local_metrics(self):
        """
        Flushes in-memory metrics to disk as a CSV file.
        """
        try:
            df = pd.DataFrame(self.local_metrics)
            df.to_csv(self.local_metrics_csv, index=False)
        except Exception as e:
            logger.error(f"Error flushing local metrics to CSV: {e}")
