import logging
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

class TextBlockDataset(Dataset):
    """
    A PyTorch Dataset that wraps a pre-tokenized array of token IDs 
    grouped into fixed-size sequence blocks.
    """
    def __init__(self, token_blocks):
        self.input_ids = torch.tensor(token_blocks, dtype=torch.long)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx]}


def get_dataset_loader(config: dict) -> DataLoader:
    """
    Loads and tokenizes the WikiText dataset according to the config.
    Concatenates all text tokens together and splits them into fixed-size blocks of max_seq_len.
    
    Args:
        config (dict): The loaded YAML configuration dictionary.
        
    Returns:
        DataLoader: PyTorch DataLoader yielding batches of shape [batch_size, max_seq_len].
    """
    import time
    dataset_name = config["dataset"]["name"]
    dataset_config = config["dataset"]["config"]
    split = config["dataset"]["split"]
    max_seq_len = config["dataset"]["max_seq_len"]
    batch_size = config["dataset"]["batch_size"]
    num_samples = config["dataset"]["num_samples"]
    model_name = config["model"]["name"]
    
    logger.info(f"Initializing tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    logger.info(f"Loading dataset {dataset_name} ({dataset_config}), split: {split}...")
    # Load dataset. Fallback to wikitext-2 if 103 fails or for offline tests if requested.
    load_start = time.perf_counter()
    try:
        if dataset_name == "ptb_text_only":
            dataset = None
            last_err = None
            
            # Method 1: Download raw text from yoonkim/lstm data source on GitHub (script-free)
            try:
                logger.info("Attempting to download Penn Treebank raw text from Github source...")
                url = "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt"
                import urllib.request
                import ssl
                context = ssl._create_unverified_context()
                local_file = "ptb_test_temp.txt"
                req = urllib.request.Request(
                    url, 
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                )
                with urllib.request.urlopen(req, context=context) as response:
                    with open(local_file, 'wb') as out_file:
                        out_file.write(response.read())
                
                dataset = load_dataset("text", data_files={"test": local_file}, split="test")
                logger.info("Successfully loaded Penn Treebank via raw Github download.")
            except Exception as e:
                logger.warning(f"Raw Github download failed: {e}. Trying HF alternatives...")
                last_err = e
            
            # Method 2: Try HF repositories
            if dataset is None:
                ptb_alternatives = [
                    ("FALcon6/ptb_text_only", "refs/convert/parquet"),
                    ("shenlong7/ptb_text_only", "refs/convert/parquet"),
                    ("FALcon6/ptb_text_only", None),
                    ("shenlong7/ptb_text_only", None),
                    ("ptb_text_only", "refs/convert/parquet"),
                    ("ptb_text_only", None)
                ]
                for ptb_repo, revision in ptb_alternatives:
                    try:
                        if revision:
                            logger.info(f"Attempting to load PTB from HF repository: {ptb_repo} (revision: {revision})...")
                            dataset = load_dataset(ptb_repo, revision=revision, split=split)
                        else:
                            logger.info(f"Attempting to load PTB from HF repository: {ptb_repo} (trust_remote_code=True)...")
                            dataset = load_dataset(ptb_repo, split=split, trust_remote_code=True)
                        logger.info(f"Successfully loaded PTB from {ptb_repo} (revision: {revision})")
                        break
                    except Exception as e:
                        last_err = e
                        try:
                            if not revision:
                                dataset = load_dataset(ptb_repo, split=split)
                                logger.info(f"Successfully loaded PTB from {ptb_repo} (no trust_remote_code)")
                                break
                        except Exception as e2:
                            last_err = e2
            if dataset is None:
                raise last_err if last_err is not None else RuntimeError("All PTB loaders failed.")
        else:
            try:
                dataset = load_dataset(dataset_name, dataset_config, split=split)
            except Exception:
                dataset = load_dataset(dataset_name, dataset_config, trust_remote_code=True, split=split)
    except Exception as e:
        logger.warning(f"Failed to load {dataset_name} ({dataset_config}) due to: {e}. Falling back to wikitext-2-raw-v1...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    load_end = time.perf_counter()
    dataset_load_time = load_end - load_start
        
    logger.info(f"Tokenizing split '{split}'...")
    tokenize_start = time.perf_counter()
    
    # Find the text column name dynamically
    possible_cols = ["text", "sentence"]
    text_column = None
    for col in possible_cols:
        if col in dataset.column_names:
            text_column = col
            break
    if text_column is None:
        text_column = dataset.column_names[0]
        
    # Define tokenization helper
    def tokenize_function(examples):
        return tokenizer(examples[text_column], truncation=False, padding=False)
        
    # Process text column
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=[text_column],
        desc="Tokenizing dataset"
    )
    
    # Flatten all token IDs into a single 1D list
    all_input_ids = []
    for ids in tokenized_dataset["input_ids"]:
        all_input_ids.extend(ids)
        
    total_tokens = len(all_input_ids)
    logger.info(f"Total tokens extracted: {total_tokens}")
    
    # Split into fixed-size chunks
    token_blocks = []
    for i in range(0, total_tokens - max_seq_len + 1, max_seq_len):
        token_blocks.append(all_input_ids[i:i + max_seq_len])
        if len(token_blocks) >= num_samples:
            break
            
    tokenize_end = time.perf_counter()
    tokenization_time = tokenize_end - tokenize_start
    
    # Check if we have enough blocks
    actual_blocks = len(token_blocks)
    logger.info(f"Created {actual_blocks} sequence blocks of length {max_seq_len} (Limit: {num_samples})")
    
    if actual_blocks == 0:
        raise ValueError(
            f"Not enough tokens to form a single block of length {max_seq_len}. "
            f"Total tokens: {total_tokens}"
        )
        
    # Wrap in custom PyTorch Dataset and DataLoader
    block_dataset = TextBlockDataset(token_blocks)
    
    # We do NOT shuffle to preserve sequential reproducibility of baseline experiments
    loader = DataLoader(
        block_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )
    
    # Attach timings as custom attributes
    loader.dataset_load_time = dataset_load_time
    loader.tokenization_time = tokenization_time
    
    return loader
