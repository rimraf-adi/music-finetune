import argparse
import logging
import pickle
import numpy as np
from pathlib import Path
import random

from configs.config import get_config
from data.download import download_atepp, get_midi_paths
from data.tokenizer import CPTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main() -> None:
    """Run the complete data preparation pipeline."""
    parser = argparse.ArgumentParser(description="Prepare data for music generation project.")
    parser.add_argument('--data_dir', type=str, default='data', help="Directory to store datasets")
    args = parser.parse_args()
    
    config = get_config()
    
    # Set seeds for reproducibility
    random.seed(config.seed)
    np.random.seed(config.seed)
    
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Step 1: Downloading ATEPP dataset...")
    atepp_dir = download_atepp(str(data_dir))
    
    logger.info("Step 2: Gathering MIDI files...")
    midi_paths = get_midi_paths(atepp_dir)
    logger.info(f"Found {len(midi_paths)} MIDI files.")
    
    tokenizer = CPTokenizer(config.vocab, config.data)
    
    logger.info("Step 3: Tokenizing MIDI files...")
    str_paths = [str(p) for p in midi_paths]
    
    # To avoid loading everything at once in a real large-scale scenario,
    # one might process in batches or use multiprocessing.
    corpus = tokenizer.encode_corpus(str_paths, max_seq_len=config.data.max_seq_len)
    logger.info(f"Successfully tokenized {len(corpus)} sequences.")
    
    if len(corpus) == 0:
        logger.error("No sequences were tokenized. Please verify the dataset.")
        return
        
    # Shuffle corpus
    random.shuffle(corpus)
    
    logger.info("Step 4: Splitting into pretrain (75%) and GRPO (25%) sets...")
    split_idx = int(len(corpus) * 0.75)
    pretrain_seqs = corpus[:split_idx]
    grpo_seqs = corpus[split_idx:]
    
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Step 5: Saving processed data...")
    with open(processed_dir / "pretrain.pkl", "wb") as f:
        pickle.dump(pretrain_seqs, f)
        
    with open(processed_dir / "grpo.pkl", "wb") as f:
        pickle.dump(grpo_seqs, f)
        
    logger.info("Data preparation complete!")
    logger.info("Statistics:")
    logger.info(f"  Total sequences: {len(corpus)}")
    logger.info(f"  Pretrain sequences: {len(pretrain_seqs)}")
    logger.info(f"  GRPO sequences: {len(grpo_seqs)}")
    
    if len(pretrain_seqs) > 0:
        lengths = [len(seq) for seq in pretrain_seqs]
        logger.info(f"  Mean length (pretrain): {np.mean(lengths):.2f}")
        
    if len(grpo_seqs) > 0:
        lengths = [len(seq) for seq in grpo_seqs]
        logger.info(f"  Mean length (GRPO): {np.mean(lengths):.2f}")

if __name__ == "__main__":
    main()
