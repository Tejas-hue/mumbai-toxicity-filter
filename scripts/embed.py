from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

# Load SBERT model once
_model = None
def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

def embed_sentences(texts: List[str]) -> np.ndarray:
    """Embed a list of sentences using SBERT."""
    model = get_model()
    embeddings = model.encode(texts, show_progress_bar=False)
    return np.array(embeddings)

if __name__ == "__main__":
    sample = ["chal na bhenc**d", "good game bhai"]
    print(embed_sentences(sample)) 