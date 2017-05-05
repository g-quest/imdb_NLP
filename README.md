# imdb_NLP

### Overview
This is a Natural Language Processing (NLP) program that uses Stochastic Gradient Descent as a classifier to evaluate the polarity of text from the "Large Movie Review Dataset" as either positive or negative.

### Dataset
The dataset is compiled from a collection of 50,000 reviews from the Internet Movie Database (IMDb). It is split evenly into 25k reviews for the training set and 25k for the test set. The distribution of positive and negative labels is also split evenly, with 25k positive and 25k negative. In the entire collection, no more than 30 reviews are allowed for any given movie since reviews for the same movie tend to have correlated ratings. Each review is written as a separate plaintext file and is separated either in the `pos` directory for positive reviews or `neg` directory for negative reviews.

The  entire dataset can be downloaded at http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz. The compressed file includes more information on how the data was collected and structured.

*Dataset Credit: Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). Learning Word Vectors for Sentiment Analysis. The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).*

### Program

The program executes the following:

`preprocessing`: The two separate sets are loaded, cleaned, and outputted into separate csv files that combine its positive and negative reviews. It's cleansed by making the data all lowercase, using regular expression to remove punctuation, digits, and other unwanted characters, and removing stopwords from the separate `stopwords.en.txt` file.

`unigram`: Creates a Document-Term Matrix (DTM) where each review is encoded as a discrete vector that counts occurrences of each word in the vocabulary it contains. It then trains the data using the Stochastic Gradient Classifier from the scikit-learn package.

`bigram`: Executes the same as `unigram` but uses a bigram representation instead.

`unigram_tfidf`: Executes the same as `unigram` but uses term frequency - inverse document frequency (tf-idf) to alleviate insignifance of high word counts

`bigram_tfidf`: Executes the same as `unigram_tfidf` using a bigram representation.

### Execution

To execute, download the dataset and extract the compressed file into the same directory as `imdb_nlp.py` and `stopwords.en.txt`. Run with Python version 3.

A confusion matrix and accuracy score of the four methods is outputted to the console upon completion of each.
