import nltk
import random
import sys
import pickle
import numpy as np
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode

from nltk.tokenize import word_tokenize

#custom classifier that takes a bunch of classifier values and maxvotes them to try to get hte best results
class MaxClassifier(ClassifierI):
    def __init__(self, *classifiers):  # arbitrary number of classifiers
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

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
    short_pos = open(pos_file, "r").read()
    short_neg = open(neg_file, "r").read()

    documents = []  # documents is gonna be a list of tuples that have a line of review and a class (pos or neg)

    for r in short_pos.split('\n'):
        documents.append((r, "pos"))
    for r in short_neg.split('\n'):
        documents.append((r, "neg"))

    all_words = []  # gonna contain all the words in both corpuses combined (nonunique)

    short_pos_words = word_tokenize(short_pos)
    short_neg_words = word_tokenize(short_neg)

    for w in short_pos_words:
        all_words.append(w.lower())
    for w in short_neg_words:
        all_words.append(w.lower())

    all_words = nltk.FreqDist(all_words)
    word_features = list(all_words.keys())[:5000]#gets the top 5000 most common words to use as features
    featuresets = [(find_features(rev,word_features), category) for (rev, category) in documents]
    random.shuffle(featuresets)
    return featuresets


def train(pos_file,neg_file,classifier_num):
    training_set = np.array(obtaindata(pos_file,neg_file))

    if(classifier_num == 2):
        print("Training multinomial naive bayes")
        MNB_classifier = SklearnClassifier(MultinomialNB())
        MNB_classifier.train(training_set)
        save_classifier("Mnaivebayes.pickle",MNB_classifier)
    elif(classifier_num == 3):
        print("Training Bernoulli Naive Bayes")
        BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
        BernoulliNB_classifier.train(training_set)
        save_classifier("BernoulliNB.pickle",BernoulliNB_classifier)
    elif(classifier_num == 4):
        print("Training Logistic Regression")
        LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
        LogisticRegression_classifier.train(training_set)
        save_classifier("LR.pickle",LogisticRegression_classifier)
    elif(classifier_num == 5):
        print("Training Stochastic Gradient Descent")
        SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
        SGDClassifier_classifier.train(training_set)
        save_classifier("SGDC.pickle",SGDClassifier_classifier)    
    elif(classifier_num == 6):
        print("Training Support vector machine")
        SVC_classifier = SklearnClassifier(SVC())
        SVC_classifier.train(training_set)
        save_classifier("SVC.pickle",SVC_classifier)     
    elif(classifier_num == 6):
        print("Training Linear Support vector machine")
        LinearSVC_classifier = SklearnClassifier(LinearSVC())
        LinearSVC_classifier.train(training_set)
        save_classifier("LSVC.pickle",LinearSVC_classifier)
    elif(classifier_num == 7):
        print("Training Nu Support vector machine")
        NuSVC_classifier = SklearnClassifier(NuSVC())
        NuSVC_classifier.train(training_set)
        save_classifier("NuSVC.pickle",NuSVC_classifier)
    else:
        print("Training naive bayes")
        classifier = nltk.NaiveBayesClassifier.train(training_set)
        save_classifier("naivebayes.pickle",classifier)
        
def test(pos_file,neg_file,classifier_num):
    testing_set = obtaindata(pos_file,neg_file)

    if(classifier_num == 2):
        #multinomial naive bayes
        classifier = load_classifier("Mnaivebayes.pickle")
        if(classifier == 0):
            return 0

        print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(classifier, testing_set)) * 100)
    elif(classifier_num == 3):
        #Bernoulli Naive Bayes
        classifier = load_classifier("BernoulliNB.pickle")
        if(classifier == 0):
            return 0

        print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(classifier, testing_set)) * 100 ) 
    elif(classifier_num == 4):
        #Logistic Regression
        classifier = load_classifier("LR.pickle")
        if(classifier == 0):
            return 0

        print("LogisticRegression_classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set)) * 100)
    elif(classifier_num == 5):
        #Stochastic Gradient Descent
        classifier = load_classifier("SGDC.pickle")
        if(classifier == 0):
            return 0

        print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(classifier, testing_set)) * 100)
    elif(classifier_num == 6):
        #Support vector machine
        classifier = load_classifier("SVC.pickle")
        if(classifier == 0):
            return 0

        print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)
    elif(classifier_num == 6):
        #Linear Support vector machine
        classifier = load_classifier("LSVC.pickle")
        if(classifier == 0):
            return 0

        print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(classifier, testing_set)) * 100)
    elif(classifier_num == 7):
        #Nu Support vector machine
        classifier = load_classifier("NuSVC.pickle")
        if(classifier == 0):
            return 0
        print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(classifier, testing_set)) * 100)
    else:
        #naive bayes
        classifier = load_classifier("naivebayes.pickle")
        if(classifier == 0):
            return 0
        classifier.show_most_informative_features(15)
        print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set)) * 100)
    
    return classifier

##Main Function
try:
    testing = eval(input("Press 1 to test. Press any other key to train. Press 2 to train all Classifiers: "))
except:
    testing = 2
if(testing == 1):
    print("Testing Selected")
elif(testing == 2):
    print("Train All Classifiers Selected")
else:
    print("Training Selected")
pos_file = input("Positive File Name: ")
neg_file = input("Negative File Name: ")
print("Positive file selected: " + pos_file +"\nNegative file name: "+neg_file)

if(testing == 1):
    print("Testing:...")
    try:
        classifier_num = eval(input("Enter the number next to which Classifier you want to use:"+
                               "\n1-\tNaive Bayes"+
                               "\n2-\tMultinomial Naive Bayes"+
                               "\n3-\tBernoulli Naive Bayes"+
                               "\n4-\tLogistic Regression"+
                               "\n5-\tStochastic Gradient Descent"+
                               "\n6-\tLinear Support vector machine"+
                               "\n7-\tNu Support vector machine"+
                               "\n8-\tMax Classifier (Only Works if all other classifiers have been trained already)"+
                               "\nDefault-  Naive Bayes\n"))
    except:
        classifier_num = 1

    if(classifier_num != 8):
        if(test(pos_file,neg_file,classifier_num) == 0):
            print("Classifier Not Trained")
    else:
        classifiersArr = []
        i = 1
        while(i < 8):
            temp = test(pos_file, neg_file, i)
            classifiersArr.append(temp)
            if(temp == 0):
                print("One of the classifiers have not been trained")
                sys.exit(1)
            i+=1      

        voted_classifier = MaxClassifier(classifiersArr[0],classifiersArr[1],classifiersArr[2],classifiersArr[3],classifiersArr[4],classifiersArr[5],classifiersArr[6])
        testing_set = obtaindata(pos_file,neg_file)
        print("max_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set)) * 100)
else:
    print("Training:...")
    if(testing == 2):
        j = 1
        while(j<8):
            train(pos_file,neg_file,j)
            j += 1
    else:
        try:
            classifier_num = eval(input("Enter the number next to which Classifier you want to use:"+
                                   "\n1-\tNaive Bayes"+
                                   "\n2-\tMultinomial Naive Bayes"+
                                   "\n3-\tBernoulli Naive Bayes"+
                                   "\n4-\tLogistic Regression"+
                                   "\n5-\tStochastic Gradient Descent"+
                                   "\n6-\tLinear Support vector machine"+
                                   "\n7-\tNu Support vector machine"+
                                   "\nDefault-  Naive Bayes\n"))
        except:
            classifier_num = 1
        train(pos_file,neg_file,classifier_num)
    
print("done")
