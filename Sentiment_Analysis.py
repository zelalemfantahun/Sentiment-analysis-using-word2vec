import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
import nltk.data
from gensim.models import word2vec
from bs4 import BeautifulSoup
import datetime
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

NB_model = GaussianNB()

train = pd.read_csv("/home/zelalem/Downloads/input/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("/home/zelalem/Downloads/input/testData.tsv",header=0, delimiter="\t", quoting=3)
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


def review_wordlist(review, remove_stopwords=False):

    review_text = (BeautifulSoup(review,'lxml').get_text())
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
        if len(raw_sentence)>0:
           sentences.append(review_wordlist(raw_sentence, remove_stopwords))
    return sentences
sentences = []

print("Parsing sentences from training set")
for review in train["review"]:
    sentences += review_sentences(review, tokenizer)

print("Training word2vec model....")




def define_model():

    global model, num_features
    num_features = 300  # Word vector dimensionality
    min_word_count = 40 # Ignores all words with total frequency lower than this.
    num_workers = 4     # Number of parallel threads
    context = 10        # Maximum distance between the current and predicted word within a sentence.
    downsampling = 1e-3 # (0.001) Downsample setting for frequent words
    model = word2vec.Word2Vec(sentences, workers=num_workers, size=num_features, min_count=min_word_count, window=context, sample=downsampling)
    model.init_sims(replace=True)
    currentDT = datetime.datetime.now()
    model_name =  (currentDT.strftime("%Y-%m-%d"+"_"+"%H:%M:%S"))
    model.save(model_name)


define_model()

def featureVecMethod(words, model, num_features):

    # Pre-initialising empty numpy array for speed
    featureVec = np.zeros(num_features, dtype="float32")
    nwords = 0

    # Converting Index2Word which is a list to a set for better speed in the execution.
    index2word_set = set(model.wv.index2word)

    for word in words:
        if word in index2word_set:
            nwords = nwords + 1
            featureVec = np.add(featureVec, model[word])

    # Dividing the result by number of words to get average
    featureVec = np.divide(featureVec, nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")

    for review in reviews:
        # Printing a status message every 1000th review
        if counter % 1000 == 0:
            print("Review %d of %d" % (counter, len(reviews)))

        reviewFeatureVecs[counter] = featureVecMethod(review, model, num_features)
        counter = counter + 1

    return reviewFeatureVecs

clean_train_reviews = []
for review in train['review']:
    clean_train_reviews.append(review_wordlist(review, remove_stopwords=True))

trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model, num_features)

# Calculating average feature vactors for test set
clean_test_reviews = []
for review in test["review"]:
    clean_test_reviews.append(review_wordlist(review, remove_stopwords=True))

testDataVecs = getAvgFeatureVecs(clean_test_reviews, model, num_features)


print("Fitting Naive Bayes Classifier to training data....")
NB_model = NB_model.fit(trainDataVecs, train["sentiment"])

# Predicting the sentiment values for test data and saving the results in a csv file
result = NB_model.predict(testDataVecs)
print (result)
output = pd.DataFrame(data={"id":test["id"], "sentiment":result})
output.to_csv( "output.csv", index=False, quoting=3 )

    # Predicting the sentiment values for test data and saving the results in a csv file
result = NB_model.predict(testDataVecs)
print (result)
output = pd.DataFrame(data={"id":test["id"], "sentiment":result})
output.to_csv( "output.csv", index=False, quoting=3 )

