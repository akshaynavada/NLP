import nltk
import pickle
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import wordnet
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn import preprocessing
from nltk.tokenize import word_tokenize

##creates a set of features that consists of {word:bool}
def find_features(document,word_features):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

##Load Classifier from pickle file
def load_classifier(name):
    try:
        classifier_f = open(name, "rb")
        classifier = pickle.load(classifier_f)
        classifier_f.close()
    except:
        return 0
    return classifier
##saves the classifier in a pickle
def save_classifier(name, classifier):
    sc = open(name,"wb")
    pickle.dump(classifier, sc)
    sc.close()

##gets data from files, tokenizes it, and returns it as a list of tuples.
##First column of tuple is a set of features which each have a set of words and a boolean indicating if the word exists or not
##Second column of tuple is a string saying if set corresponds to pos or neg
def obtaindata(pos_file,neg_file):
    ##read the input files
    pos_contents = open(pos_file, "r").read()
    neg_contents = open(neg_file, "r").read()


    documents = []  # documents is gonna be a list of tuples that have a line of review and a class (pos or neg)

    for r in pos_contents.split('\n'):
        documents.append((r, "pos"))
    for r in neg_contents.split('\n'):
        documents.append((r, "neg"))


    print("tokenizing all reviews")
    all_words = []  # gonna contain all the words in both corpuses combined (nonunique)
    lemmatizer = WordNetLemmatizer()
    pos_words = word_tokenize(pos_contents)
    neg_words = word_tokenize(neg_contents)

    print("adding pos tags")
    tagged_pos_words = nltk.pos_tag(pos_words)
    tagged_neg_words = nltk.pos_tag(neg_words)

    print("lemmatizing words")
    #if word is not a noun adjective adverb or verb ignore it
    for (w,p) in tagged_pos_words:
        if(p[:2] == "NN"):
            out = lemmatizer.lemmatize(w,pos = wordnet.NOUN)
            all_words.append((out.lower(),wordnet.NOUN))
        elif(p[:2] == "JJ"):
            out = lemmatizer.lemmatize(w,pos = wordnet.ADJ)
            all_words.append((out.lower(),wordnet.ADJ))
        elif(p[:2] == "VB"):
            out = lemmatizer.lemmatize(w,pos = wordnet.VERB)
            all_words.append((out.lower(),wordnet.VERB))
        elif(p[:2] == "RB"):
            out = lemmatizer.lemmatize(w,pos = wordnet.ADV)
            all_words.append((out.lower(),wordnet.ADV))

    for (w,p) in tagged_neg_words:
        if(p[:2] == "NN"):
            out = lemmatizer.lemmatize(w,pos = wordnet.NOUN)
            all_words.append((out.lower(),wordnet.NOUN))
        elif(p[:2] == "JJ"):
            out = lemmatizer.lemmatize(w,pos = wordnet.ADJ)
            all_words.append((out.lower(),wordnet.ADJ))
        elif(p[:2] == "VB"):
            out = lemmatizer.lemmatize(w,pos = wordnet.VERB)
            all_words.append((out.lower(),wordnet.VERB))
        elif(p[:2] == "RB"):
            out = lemmatizer.lemmatize(w,pos = wordnet.ADV)
            all_words.append((out.lower(),wordnet.ADV))

    # print("creating bigrams")
    # all_words = list(nltk.bigrams(all_words))
    # pos_words = list(nltk.bigrams(posTemp))
    # neg_words = list(nltk.bigrams(negTemp))

    # for w in pos_words:
    #     if w[0].isalnum():
    #         all_words.append((w[0].lower(),w[1]))
    # for w in neg_words:
    #     if w[0].isalnum():
    #         all_words.append((w[0].lower(),w[1]))


    print("selecting most used overall words to be used as features")

    all_words = nltk.FreqDist(all_words)
    word_features = list(all_words.keys())[:5000]#gets the top 5000 most common words to use as features
    featuresets = [(find_features(review,word_features), category) for (review, category) in documents]
    # random.shuffle(featuresets)
    return featuresets


def train(pos_file,neg_file):
    training_set = np.array(obtaindata(pos_file,neg_file))
    print("Training Support vector machine")
    print("scaling support vector machine")


    SVC_classifier = SklearnClassifier(SVC())
    SVC_classifier.train(training_set)
    save_classifier("SVC.pickle",SVC_classifier)


def test(pos_file,neg_file):
    testing_set = obtaindata(pos_file,neg_file)
    #Support vector machine
    classifier = load_classifier("SVC.pickle")

    if(classifier == 0):
        return 0
    print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)
    return classifier





##Main Function


try:
    testing = eval(input("Press 1 to test. Press any other key to train"))
except:
    testing = 2
if(testing == 1):
    print("Testing Selected")
else:
    print("Training Selected")
pos_file = input("Positive File Name: ")
neg_file = input("Negative File Name: ")
print("Positive file selected: " + pos_file +"\nNegative file name: "+neg_file)

if(testing == 1):
    print("Testing:...")
    if(test(pos_file,neg_file) == 0):
        print("Classifier Not Trained")
else:
    print("Training:...")
    train(pos_file,neg_file)

print("done")

