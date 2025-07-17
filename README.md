# Mumbai Toxicity Filter v1

A lightweight Hinglish (Hindi-English) chat toxicity classifier for gaming and social platforms.

## Features
- Classifies Hinglish chat as toxic or non-toxic
- Uses Sentence-BERT (SBERT) for embeddings
- XGBoost for fast, lightweight classification
- Easily upgradable for context/severity in future

## Folder Structure
```
mumbai-toxicity-filter/
  data/
    dataset.csv
  models/
    toxicity_classifier.pkl
  scripts/
    train_model.py
    embed.py
    predict.py
    preprocess.py
    utils.py
  notebooks/
    analysis.ipynb
  README.md
  requirements.txt
```

## Setup
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. (Optional) Activate your virtual environment.

## Training
To train the model on the synthetic dataset:
```bash
python scripts/train_model.py
```
Model will be saved to `models/toxicity_classifier.pkl`.

## Prediction
To predict toxicity for new messages:
```bash
python scripts/predict.py --text "chal na bhenc**d" "good game bhai"
```
Or run without arguments to use sample messages.

## Data
- `data/dataset.csv` contains synthetic Hinglish chat samples with labels (1 = toxic, 0 = non-toxic).

## Analysis
- See `notebooks/analysis.ipynb` for data exploration and confusion matrix plotting.

## Upgrades
- The codebase is structured for easy upgrades (context, severity, new models, etc.)

---
MIT License 