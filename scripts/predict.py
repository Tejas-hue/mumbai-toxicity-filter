import joblib
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from scripts.preprocess import preprocess_texts
from scripts.embed import embed_sentences

def predict_toxicity(texts):
    """Predict toxicity for a list of texts."""
    # Load model
    model_path = os.path.join(os.path.dirname(__file__), '../models/toxicity_classifier.pkl')
    clf = joblib.load(model_path)
    # Preprocess
    clean_texts = preprocess_texts(texts)
    # Embed
    X = embed_sentences(clean_texts)
    # Predict
    preds = clf.predict(X)
    return preds

if __name__ == "__main__":
    # Example usage: predict on sample list or user input
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, nargs='*', help='Text(s) to classify')
    args = parser.parse_args()
    if args.text:
        texts = args.text
    else:
        texts = [
            "chal na bhenc**d",
            "good game bhai",
            "abe chomu",
            "teri maa ka"
        ]
    preds = predict_toxicity(texts)
    for t, p in zip(texts, preds):
        print(f'"{t}" -> {"Toxic" if p == 1 else "Non-toxic"}') 