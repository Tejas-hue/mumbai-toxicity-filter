import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from scripts.preprocess import preprocess_texts
from scripts.embed import embed_sentences
from scripts.utils import log_step

def train():
    log_step("Loading dataset...")
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/dataset.csv'))
    X_raw = df['text'].tolist()
    y = df['label'].values

    log_step("Preprocessing texts...")
    X_clean = preprocess_texts(X_raw)

    log_step("Embedding texts...")
    X_emb = embed_sentences(X_clean)

    log_step("Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(X_emb, y, test_size=0.2, random_state=42, stratify=y)

    log_step("Training XGBoost classifier...")
    clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=50, max_depth=3)
    clf.fit(X_train, y_train)

    log_step("Evaluating model...")
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    log_step("Saving model...")
    os.makedirs(os.path.join(os.path.dirname(__file__), '../models'), exist_ok=True)
    joblib.dump(clf, os.path.join(os.path.dirname(__file__), '../models/toxicity_classifier.pkl'))
    log_step("Model saved to models/toxicity_classifier.pkl")

if __name__ == "__main__":
    train() 