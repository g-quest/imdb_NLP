
import csv
import os
import re
import pandas as pd
from random import shuffle
from nltk import word_tokenize
from nltk.util import ngrams
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
    
trainPath = './aclImdb/train/' 
testPath  = './aclImdb/test/'

trainFile = 'imdb_train.csv'
testFile  = 'imdb_test.csv'

def main():
    
    preprocess(trainPath, trainFile)
    preprocess(testPath, testFile)
    
    trainDF = pd.read_csv('./' + trainFile, encoding = "ISO-8859-1")     
    testDF  = pd.read_csv('./' + testFile, encoding = 'ISO-8859-1') 
  
    unigram(trainDF, testDF)
    bigram(trainDF, testDF)
    unigram_tfidf(trainDF, testDF)
    bigram_tfidf(trainDF, testDF)

    print("\nProgram complete.\n")

def preprocess(inPath, outFile):
       
    posReviews = load_reviews(inPath + 'pos/')
    clean_reviews(posReviews)
    posReviewsFiltered = remove_stopwords(posReviews)

    negReviews = load_reviews(inPath + 'neg/')
    clean_reviews(negReviews)
    negReviewsFiltered = remove_stopwords(negReviews)

    i = 0
    outputList = []
    for review in posReviewsFiltered:
        outputList.append([i, review, 1])
        i += 1
    for review in negReviewsFiltered:
        outputList.append([i, review, 0])
        i += 1 

    shuffle(outputList)
    
    combinedFile = open('./' + outFile, 'w')
    writer = csv.writer(combinedFile)
    writer.writerow(['review_number', 'text', 'polarity'])
    for review in outputList:
        writer.writerow(review)
    combinedFile.close()   
    
    print('\n{0} file created'.format(outFile))
    
    return None

def load_reviews(directory):
    
    files = [file for file in os.listdir(directory)] 

    reviews = []
    for file in files:
        with open(os.path.join(directory,file), encoding = "ISO-8859-1") as plaintext:
            lines = plaintext.read().replace('\n', '')
            reviews.append(lines)

    return reviews

def clean_reviews(reviews_to_clean):

    revCount = len(reviews_to_clean)
    for review in range(0, revCount):
        reviews_to_clean[review] = reviews_to_clean[review].lower()
        reviews_to_clean[review] = re.sub('<br />',' ', reviews_to_clean[review])
        reviews_to_clean[review] = re.sub(r'[-\[\]]', ' ', reviews_to_clean[review])
        reviews_to_clean[review] = re.sub(r'[^a-zA-Z0-9 ]', '', reviews_to_clean[review])
        reviews_to_clean[review] = re.sub(r'\d+', '', reviews_to_clean[review])
    
    return reviews_to_clean
        
def remove_stopwords(reviews):

    # get stopwords from file
    with open('./stopwords.en.txt', encoding = "ISO-8859-1") as file:
        stopWords = file.read().splitlines()
    
    #filter words in each review
    filtered_words = []
    for review in reviews:
        words = []
        for word in review.split():
            if word not in stopWords:
                words.append(word)
        filtered_words.append(words)
    
    #combine back into single string
    filtered_reviews = []
    for review in filtered_words:
        filtered_reviews.append(" ".join(review)) 
        
    return filtered_reviews
    
def unigram(trainDF, testDF):

    print('\n*** Unigram SGD Classifier ***\n')
    
    # training Document Term Matrix (DTM) from training set
    cv = CountVectorizer()
    X_train = pd.DataFrame(cv.fit_transform(trainDF['text']).toarray(), columns = cv.get_feature_names())
    y_train = trainDF.iloc[:, -1].values

    # get test DTM for test set fitted from training DTM
    X_test = cv.transform(testDF['text'])
    y_test = testDF.iloc[:, -1].values

    # use SGD classifier to fit training data
    cfr = SGDClassifier(loss = "hinge", penalty = "l1", shuffle = True)
    cfr.fit(X_train, y_train)

    # predict using fitted data
    predictions = cfr.predict(X_test)   

    # examine performance
    cm = confusion_matrix(y_test, predictions)
    print('Unigram SGD Confusion Matrix: \n', cm) 
    testScore = accuracy_score(y_test, predictions)
    print('\nUnigram SGD Classifier Training Set Test Score: ', testScore)

    print('\nUnigram complete.\n')
    
def bigram(trainDF, testDF):

    print('\n*** Bigram SGD Classifier ***\n')
    
    # get tokens in training set
    tokenListTrain = []
    for review in trainDF['text']:
        token = word_tokenize(review)
        tokenListTrain.append(token)
    
    # get training bigram representation from tokens
    bigramListTrain = []
    for tokens in tokenListTrain:
        bigrams = list(ngrams(tokens, 2))
        bigramListTrain.append(bigrams)

    # turns sequences into scipy.sparse matrices
    fh = FeatureHasher(input_type='string')
    X_train = fh.transform(((' '.join(x) for x in bigrams) for bigrams in bigramListTrain))
    y_train = trainDF.iloc[:, -1].values   

    # get test bigram representation for test set fitted from training sparse matrices
    tokenListTest = []
    for review in testDF['text']:
        token = word_tokenize(review)
        tokenListTest.append(token)
    
    bigramListTest = []
    for tokens in tokenListTest:
        bigrams = list(ngrams(tokens, 2))
        bigramListTest.append(bigrams)
        
    X_test = fh.transform(((' '.join(x) for x in bigrams) for bigrams in bigramListTest))
    y_test = testDF.iloc[:, -1].values

    # use SGD classifier to fit training data
    cfr = SGDClassifier(loss = "hinge", penalty = "l1", shuffle = True)
    cfr.fit(X_train, y_train)

    # predict using fitted data
    predictions = cfr.predict(X_test)   

    # examine performance
    cm = confusion_matrix(y_test, predictions)
    print('Bigram SGD Confusion Matrix: \n', cm)
    testScore = accuracy_score(y_test, predictions)
    print('\nBigram SGD Classifier Training Set Test Score: ', testScore)

    print('\nBigram complete.\n')

def unigram_tfidf(trainDF, testDF):
    
    print('\n*** Unigram TF-IDF SGD Classifier ***\n')
    
    # use tfidf on unigram representation
    tv = TfidfVectorizer()
    X_train = pd.DataFrame(tv.fit_transform(trainDF['text']).toarray(), columns = tv.get_feature_names())
    y_train = trainDF.iloc[:, -1].values   

    X_test = tv.transform(testDF['text']) 
    y_test = testDF.iloc[:, -1].values

    # use SGD classifier to fit training data
    cfr = SGDClassifier(loss = "hinge", penalty = "l1", shuffle = True)
    cfr.fit(X_train, y_train)

    # predict using fitted data
    predictions = cfr.predict(X_test)   

    # examine performance
    cm = confusion_matrix(y_test, predictions)
    print('Unigram TF-IDF SGD Confusion Matrix: \n', cm)
    testScore = accuracy_score(y_test, predictions)
    print('\nUnigram TF-IDF SGD Classifier Training Set Test Score: ', testScore)

    print('\nUnigram TF-IDF complete.\n')
 
def bigram_tfidf(trainDF, testDF):
    
    print('\n*** Bigram TF-IDF SGD Classifier ***\n')
    
    tokenListTrain = []
    for review in trainDF['text']:
        token = word_tokenize(review)
        tokenListTrain.append(token)
    
    bigramListTrain = []
    for tokens in tokenListTrain:
        bigrams = list(ngrams(tokens, 2))
        bigramListTrain.append(bigrams)

    fh = FeatureHasher(input_type='string')
    fht = fh.transform(((' '.join(x) for x in bigrams) for bigrams in bigramListTrain))
    tf = TfidfTransformer()
    X_train = tf.fit_transform(fht)   
    y_train = trainDF.iloc[:, -1].values   

    tokenListTest = []
    for review in testDF['text']:
        token = word_tokenize(review)
        tokenListTest.append(token)
    
    bigramListTest = []
    for tokens in tokenListTest:
        bigrams = list(ngrams(tokens, 2))
        bigramListTest.append(bigrams)

    X_test = fh.transform(((' '.join(x) for x in bigrams) for bigrams in bigramListTest))
    y_test = testDF.iloc[:, -1].values      
    
    # use SGD classifier to fit training data
    cfr = SGDClassifier(loss = "hinge", penalty = "l1", shuffle = True)
    cfr.fit(X_train, y_train)

    # predict using fitted data
    predictions = cfr.predict(X_test)  

    # examine performance 
    cm = confusion_matrix(y_test, predictions)
    print('Bigram TF-IDF SGD Confusion Matrix: \n', cm)
    testScore = accuracy_score(y_test, predictions)
    print('\nBigram TF-IDF SGD Classifier Training Set Test Score: ', testScore)

    print('\nBigram TF-IDF complete.\n')
     
if __name__ == "__main__":

     main()