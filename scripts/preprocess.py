import re
from typing import List

# Slang dictionary for normalization
SLANG_DICT = {
    "bhenc**d": "bhenchod",
    "behen ke lode": "bhen ke lode",
    "noob": "novice",
    "noobde": "novice",
    "lawde": "idiot",
    "chomu": "silly",
    "bakchodi": "nonsense",
    "gadha": "donkey",
    "op": "awesome",
    "gg": "good game",
    "ez": "easy",
    "pro": "professional",
    "hacker": "cheater",
    "carry": "support",
    "camper": "hider",
    "legend": "champion",
    "bakwaas": "nonsense",
    "band baj gayi": "got defeated",
    "lag gayi": "got defeated",
    "mast": "great",
    "mazza": "fun",
    "chill": "relax",
    "masti": "fun"
}

def normalize_slang(text: str, slang_dict: dict = SLANG_DICT) -> str:
    """Replace slang words in text using the provided slang dictionary."""
    for slang, replacement in slang_dict.items():
        # Use word boundaries to avoid partial matches
        text = re.sub(rf'\b{re.escape(slang)}\b', replacement, text)
    return text

def remove_punctuation(text: str) -> str:
    """Remove punctuation from text."""
    return re.sub(r'[^\w\s]', '', text)

def preprocess_text(text: str) -> str:
    """Full preprocessing pipeline: lowercasing, punctuation removal, slang normalization."""
    text = text.lower()
    text = remove_punctuation(text)
    text = normalize_slang(text)
    return text

def preprocess_texts(texts: List[str]) -> List[str]:
    """Preprocess a list of texts."""
    return [preprocess_text(t) for t in texts]

if __name__ == "__main__":
    # Example usage
    sample = [
        "Chal na bhenc**d!",
        "Good game bhai",
        "Kya noob hai tu?"
    ]
    print(preprocess_texts(sample)) 