# 🧠 Mumbai Toxicity Filter - v1.0  
> _Detecting region-specific slurs and toxic Hinglish chat in real-time._

---

## ⚡ What is this project?

> A targeted solution to **detect and analyze toxic chat patterns** on Indian multiplayer game servers — specifically **Valorant Mumbai**.  
This project aims to address online abuse in region-specific forms: code-switched Hinglish slurs, cultural insults, and behaviorally toxic messages.

---

## 🎯 Goals

> **Long-term mission:**  
Build a highly accurate, explainable, and modular filter that can **flag**, **analyze**, and **visualize** toxicity trends across Indian game servers, beginning with Mumbai.

> **Version 1 (this release):**  
🚧 Initial prototype — rule-based + ML hybrid  
✅ Code-switched text preprocessing (basic)  
✅ Hinglish slur normalization using slang dictionary  
✅ Basic ML classifier trained on mock data  
✅ Real-time prediction script (command-line)  
✅ Project modularized with preprocessing, training, and prediction scripts  
✅ Initial trend analysis notebook (`notebooks/analysis.ipynb`)

---

## 🛠️ Structure

```bash
mumbai-toxicity-filter/
├── data/                   # Dataset files (e.g., raw chat data)
├── models/                 # Saved models
├── notebooks/              # Jupyter notebooks for analysis
├── scripts/                # All Python scripts (train, preprocess, predict)
├── requirements.txt        # Dependencies
└── README.md               # This file
