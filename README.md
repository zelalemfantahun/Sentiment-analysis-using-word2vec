# Sentiment Analysis using Word2Vec

> An end-to-end Natural Language Processing (NLP) project that uses **Word2Vec** embeddings and machine learning techniques to classify movie reviews as **positive** or **negative**.

---

## Project Overview

This project demonstrates how distributed word representations (Word2Vec) can improve sentiment analysis by transforming text into dense numerical vectors that capture semantic relationships between words.

The workflow includes:

- Text preprocessing
- Tokenization
- Training a Word2Vec model
- Sentence vector generation
- Machine Learning classification
- Model evaluation

---

## Dataset

This project uses the **IMDb Movie Review Dataset**.

- **25,000** labeled training reviews
- Binary sentiment classification
- Positive and Negative labels

Dataset source:

https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

---

## Project Workflow

```text
IMDb Reviews
      │
      ▼
Text Cleaning
      │
      ▼
Tokenization
      │
      ▼
Word2Vec Training
      │
      ▼
Sentence Embeddings
      │
      ▼
Feature Extraction
      │
      ▼
Machine Learning Classifier
      │
      ▼
Sentiment Prediction
      │
      ▼
Model Evaluation
```

---

## Features

- Word2Vec word embeddings
- Text preprocessing pipeline
- Feature engineering
- Sentiment classification
- Configurable Word2Vec parameters
- End-to-end NLP workflow

---

## Technologies

- Python
- Gensim
- NLTK
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

---

## Repository Structure

```text
Sentiment-analysis-using-word2vec/

├── data/
├── models/
├── notebooks/
├── results/
├── src/
├── images/
├── tests/
│
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/zelalemfantahun/Sentiment-analysis-using-word2vec.git
```

Navigate into the project

```bash
cd Sentiment-analysis-using-word2vec
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

```bash
python src/main.py
```

or run the original script

```bash
python Sentiment_Analysis.py
```

---

## Word2Vec Parameters

| Parameter | Description |
|-----------|-------------|
| vector_size | Dimensionality of the word vectors |
| window | Context window size |
| min_count | Minimum frequency required for a word |
| workers | Number of CPU threads |
| sample | Down-sampling threshold for frequent words |

Example configuration

```python
vector_size = 300
min_count = 40
workers = 4
window = 10
sample = 1e-3
```

---

## Future Improvements

- Compare Word2Vec with FastText
- Compare Word2Vec with GloVe
- Implement Doc2Vec
- Add LSTM classifier
- Add Transformer (BERT) baseline
- Hyperparameter tuning
- Cross-validation
- Docker support

---

## Skills Demonstrated

- Natural Language Processing (NLP)
- Text Preprocessing
- Feature Engineering
- Word Embeddings
- Machine Learning
- Data Analysis
- Python Development

---

## Author

**Zelalem Abate**

Toronto, Ontario, Canada

GitHub:
https://github.com/zelalemfantahun

---

## License

This project is licensed under the MIT License.
