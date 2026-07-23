"""
sentiment_word2vec_nb.py
========================
Trains a Word2Vec embedding on movie review text, averages word vectors
per review, and fits a Gaussian Naive Bayes classifier on top to predict
sentiment. Fixed/modernized version of the original script.

Usage:
    python sentiment_word2vec_nb.py \
        --num-features 300 --min-word-count 40 --num-workers 4 --context 10

Expects data/labeledTrainData.csv and data/testData.csv relative to the
project root (BASE_DIR / "data"), each a tab-separated file with a
"review" column (train also needs "sentiment", test needs "id").
"""

import argparse
import datetime
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from gensim.models import word2vec
from nltk.corpus import stopwords
import nltk.data
from sklearn.naive_bayes import GaussianNB

warnings.filterwarnings("ignore", category=UserWarning, module="bs4")

# --- Paths -------------------------------------------------------------
# Adjust parents[N] to match where this script actually lives relative
# to your project root (parents[0] = same folder as this file's parent,
# parents[1] = one level up from that, etc.).
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
TRAIN_FILE = DATA_DIR / "labeledTrainData.csv"
TEST_FILE = DATA_DIR / "testData.csv"
OUTPUT_FILE = BASE_DIR / "results" / "output.csv"


# --- Text preprocessing -------------------------------------------------

def review_wordlist(review, remove_stopwords=False):
    """Clean a single review into a list of lowercase alphabetic words."""
    review_text = BeautifulSoup(review, "lxml").get_text()
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    words = review_text.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if w not in stops]
    return words


def review_sentences(review, tokenizer, remove_stopwords=False):
    """Split a review into a list of cleaned, tokenized sentences."""
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(review_wordlist(raw_sentence, remove_stopwords))
    return sentences


# --- Word2Vec ------------------------------------------------------------

def train_word2vec(sentences, num_features, min_word_count, num_workers, context):
    """Train a Word2Vec model on tokenized sentences and save it to disk."""
    downsampling = 1e-3  # Downsample setting for frequent words

    model = word2vec.Word2Vec(
        sentences,
        workers=num_workers,
        vector_size=num_features,   # renamed from `size` in gensim 4.x
        min_count=min_word_count,
        window=context,
        sample=downsampling,
    )
    # `.init_sims(replace=True)` is deprecated/unnecessary since gensim 4.0
    # (the memory optimization it triggered now happens automatically).

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    # Colons aren't valid in filenames on Windows -> use a safe timestamp format.
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model.save(str(MODEL_DIR / f"word2vec_{timestamp}.model"))
    return model


def feature_vector(words, model, num_features):
    """Average the Word2Vec vectors of a review's in-vocabulary words."""
    feature_vec = np.zeros(num_features, dtype="float32")
    n_words = 0

    # gensim 4.x: vocabulary lives on model.wv, not model directly.
    index2word_set = set(model.wv.index_to_key)

    for word in words:
        if word in index2word_set:
            n_words += 1
            feature_vec = np.add(feature_vec, model.wv[word])

    if n_words == 0:
        # No in-vocabulary words -> avoid a ZeroDivisionError / NaN vector.
        return feature_vec
    return np.divide(feature_vec, n_words)


def get_avg_feature_vecs(reviews, model, num_features):
    """Compute an average feature vector for every review in ``reviews``."""
    feature_vecs = np.zeros((len(reviews), num_features), dtype="float32")
    for i, review in enumerate(reviews):
        if i % 1000 == 0:
            print(f"Review {i} of {len(reviews)}")
        feature_vecs[i] = feature_vector(review, model, num_features)
    return feature_vecs


# --- Pipeline -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Word2Vec + Naive Bayes sentiment classifier.")
    parser.add_argument("--num-features", type=int, default=300)
    parser.add_argument("--min-word-count", type=int, default=40)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--context", type=int, default=10)
    args = parser.parse_args()

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Training file not found: {TRAIN_FILE}")
    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Test file not found: {TEST_FILE}")

    train = pd.read_csv(TRAIN_FILE, header=0, delimiter="\t", quoting=3)
    test = pd.read_csv(TEST_FILE, header=0, delimiter="\t", quoting=3)

    tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

    print("Parsing sentences from training set...")
    sentences = []
    for review in train["review"]:
        sentences += review_sentences(review, tokenizer)

    print("Training word2vec model...")
    model = train_word2vec(
        sentences,
        num_features=args.num_features,
        min_word_count=args.min_word_count,
        num_workers=args.num_workers,
        context=args.context,
    )

    print("Building average feature vectors for training set...")
    clean_train_reviews = [
        review_wordlist(review, remove_stopwords=True) for review in train["review"]
    ]
    train_data_vecs = get_avg_feature_vecs(clean_train_reviews, model, args.num_features)

    print("Building average feature vectors for test set...")
    clean_test_reviews = [
        review_wordlist(review, remove_stopwords=True) for review in test["review"]
    ]
    test_data_vecs = get_avg_feature_vecs(clean_test_reviews, model, args.num_features)

    print("Fitting Naive Bayes classifier to training data...")
    nb_model = GaussianNB()
    nb_model.fit(train_data_vecs, train["sentiment"])

    result = nb_model.predict(test_data_vecs)
    output = pd.DataFrame(data={"id": test["id"], "sentiment": result})

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_FILE, index=False, quoting=3)
    print(f"Predictions written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
