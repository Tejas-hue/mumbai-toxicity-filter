import logging
from scripts.preprocess import SLANG_DICT, preprocess_text

def load_slang_dict():
    """Return the slang dictionary used for normalization."""
    return SLANG_DICT

def clean_text(text: str) -> str:
    """Clean a single text using the preprocessing pipeline."""
    return preprocess_text(text)

def log_step(message: str):
    """Log a step in the pipeline."""
    logging.basicConfig(level=logging.INFO)
    logging.info(message) 