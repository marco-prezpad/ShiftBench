#!/usr/bin/env python3
"""
extract_newsgroups_embeddings.py

Extract TF-IDF embeddings from 20 Newsgroups (two categories).

Saves embeddings/labels to embeddings/newsgroups/.

Author: Marco Pérez Padilla
Date:   19-08-2026
"""

from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

OUT_DIR = Path("embeddings/newsgroups")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES = ["rec.sport.baseball", "sci.med"]

print("Downloading 20 Newsgroups subset...")
news_train = fetch_20newsgroups(
    subset="train", categories=CATEGORIES, shuffle=True, random_state=42
)
news_test = fetch_20newsgroups(
    subset="test", categories=CATEGORIES, shuffle=True, random_state=42
)

print("Vectorizing with TF-IDF (500 features)...")
vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
X_train = vectorizer.fit_transform(news_train.data)
X_test = vectorizer.transform(news_test.data)

X_train_dense = X_train.toarray().astype(np.float32)
X_test_dense = X_test.toarray().astype(np.float32)
y_train = np.array(news_train.target)
y_test = np.array(news_test.target)

np.save(OUT_DIR / "train_embeddings.npy", X_train_dense)
np.save(OUT_DIR / "train_labels.npy", y_train)
np.save(OUT_DIR / "test_embeddings.npy", X_test_dense)
np.save(OUT_DIR / "test_labels.npy", y_test)

print(f"Train embeddings: {X_train_dense.shape}")
print(f"Test embeddings: {X_test_dense.shape}")
print("Saved to", OUT_DIR)