import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
import nltk.data
from gensim.models import word2vec
from bs4 import BeautifulSoup
import datetime
from sklearn.naive_bayes import GaussianNB
NB_model = GaussianNB()
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')



class sentiment_analyzer:

    def __init__(self):
        self.train = pd.read_csv("/home/zelalem/Downloads/input/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
        self.test = pd.read_csv("/home/zelalem/Downloads/input/testData.tsv", header=0, delimiter="\t", quoting=3)
        self.tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    def review_wordlist(review, remove_stopwords=False):
        review_text = (BeautifulSoup(review, 'lxml').get_text())
        review_text = re.sub("[^a-zA-Z]", " ", review_text)
        words = review_text.lower().split()
        if remove_stopwords:
            stops = set(stopwords.words("english"))
            words = [w for w in words if not w in stops]

        return (words)

    def review_sentences(review, tokenizer, remove_stopwords=False):
        raw_sentences = tokenizer.tokenize(review.strip())
        sentences = []

        for raw_sentence in raw_sentences:
            if len(raw_sentence) > 0:

                sentences.append(review_wordlist(raw_sentence, remove_stopwords))

        return sentences

    sentences = []
    print("Parsing sentences from training set")
    for review in train["review"]:
        sentences += review_sentences(review, tokenizer)






if __name__ == '__main__':


    a = sentiment_analyzer()
