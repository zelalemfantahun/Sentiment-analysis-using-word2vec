# Sentiment-analysis-using-word2vec

Dataset

The labeled training set. The file is tab-delimited and has a header row followed by 25,000 rows containing an id, sentiment, and text for each review. 

word2vec. The data can be found here https://www.kaggle.com/varun08/imdb-dataset/downloads/imdb-dataset.zip/1

**sentences** : iterable of iterables, optional
            The sentences iterable can be simply a list of lists of tokens
            
workers : int, optional
            Use these many worker threads to train the model (=faster training with multicore machines).
            
size : int, optional
            Dimensionality of the word vectors.
            
min_count : int, optional
            Ignores all words with total frequency lower than this.
            
window : int, optional
            Maximum distance between the current and predicted word within a sentence.
            
sample : float, optional
            The threshold for configuring which higher-frequency words are randomly downsampled,
            useful range is (0, 1e-5).