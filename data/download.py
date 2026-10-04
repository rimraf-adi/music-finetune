import os
import zipfile
import urllib.request
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

def download_and_extract(url: str, extract_dir: Path) -> None:
    """Download and extract a zip file to the given directory."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    zip_path = extract_dir / "temp.zip"
    
    # Check if directory already has files
    has_files = any(extract_dir.iterdir())
    if not has_files:
        logger.info(f"Downloading from {url} to {zip_path}")
        urllib.request.urlretrieve(url, zip_path)
        logger.info(f"Extracting {zip_path} to {extract_dir}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        zip_path.unlink()
        logger.info("Download and extraction complete.")
    else:
        logger.info(f"Directory {extract_dir} is not empty, skipping download.")

def download_atepp(data_dir: str) -> Path:
    """
    Download the ATEPP dataset.
    
    Args:
        data_dir: Base directory to download data into.
        
    Returns:
        Path to the extracted dataset.
    """
    # Note: Using a placeholder direct URL for ATEPP if direct zip isn't available.
    # In practice, you might need a specific gdown link or a GitHub archive link.
    url = "https://github.com/BetsyTang/ATEPP/archive/refs/heads/main.zip"
    extract_dir = Path(data_dir) / "ATEPP"
    download_and_extract(url, extract_dir)
    return extract_dir

def download_maestro(data_dir: str) -> Path:
    """
    Download the MAESTRO dataset.
    
    Args:
        data_dir: Base directory to download data into.
        
    Returns:
        Path to the extracted dataset.
    """
    url = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
    extract_dir = Path(data_dir) / "MAESTRO"
    download_and_extract(url, extract_dir)
    return extract_dir

def get_midi_paths(data_dir: Path) -> List[Path]:
    """
    Get a list of all MIDI files in a directory and its subdirectories.
    
    Args:
        data_dir: Directory to search.
        
    Returns:
        List of pathlib.Path objects for each MIDI file found.
    """
    return list(data_dir.rglob("*.mid")) + list(data_dir.rglob("*.midi"))
