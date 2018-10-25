# Sentiment-analysis-using-word2vec

Dataset
=======
The labeled training set. The file is tab-delimited and has a header row followed by 25,000 rows containing an id, sentiment, and text for each review. The data can be found here  https://www.kaggle.com/varun08/imdb-dataset/downloads/imdb-dataset.zip/1

word2vec Parameters:

sentences : iterable of iterables, optional
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
            
Usage


python3.5 "Sentiment_Analysis.py" "file_path_to/testData.csv" "size" "mine_count" "workers" "window"

Sample setting values for the various parameters
================================================
num_features = 300  # Word vector dimensionality
min_word_count = 40 # Minimum word count
num_workers = 4     # Number of parallel threads
context = 10        # Context window size
downsampling = 1e-3 # (0.001) Downsample setting for frequent words