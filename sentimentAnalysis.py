import nltk
import pickle
import numpy as np
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import wordnet
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.tokenize import word_tokenize


##creates a set of features that consists of {word:bool}
def find_features(document, word_features):
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
    sc = open(name, "wb")
    pickle.dump(classifier, sc)
    sc.close()


def processDocument(doc):
    lemmatizer = WordNetLemmatizer()
    word = nltk.word_tokenize(doc)
    tagged = nltk.pos_tag(word)
    tagged_word = ""
    for w, p in tagged:
        if (p[:2] == "NN"):
            out = lemmatizer.lemmatize(w, pos=wordnet.NOUN)
            tagged_word += out.lower() + "_" + p + " "
        elif (p[:2] == "JJ"):
            out = lemmatizer.lemmatize(w, pos=wordnet.ADJ)
            tagged_word += out.lower() + "_" + p + " "
        elif (p[:2] == "VB"):
            out = lemmatizer.lemmatize(w, pos=wordnet.VERB)
            tagged_word += out.lower() + "_" + p + " "
        elif (p[:2] == "RB"):
            out = lemmatizer.lemmatize(w, pos=wordnet.ADV)
            tagged_word += out.lower() + "_" + p + " "
    return tagged_word


def obtaindata(pos_file, neg_file, vectorizer=0, tfidf_transformer=0, input_data=0):
    text = []  # each element of text is the processed words of a document
    labels = []  # each element is the corresponding label

    print("POS Tagging and lemmatizing")
    if ((pos_file == 0 or neg_file == 0) and input_data != 0):  # if there is raw input data provided process that, else process files
        text.append(processDocument(input_data))
        labels.append("poop")  # not needed cuz u most likely know the sentiment of ur own input
    else:
        pos_contents = open(pos_file, "r").read()
        neg_contents = open(neg_file, "r").read()
        for doc in pos_contents.split('\n'):
            tagged_word = processDocument(doc)
            text.append(tagged_word)
            labels.append("pos")
        for doc in neg_contents.split('\n'):
            tagged_word = processDocument(doc)
            text.append(tagged_word)
            labels.append("neg")
    n = 4
    print("creating " + str(n) + "-grams")
    if (vectorizer == 0):
        vectorizer = CountVectorizer(ngram_range=(1, n))
        X = vectorizer.fit_transform(text)
        save_classifier("vectorizer.pickle", vectorizer)
    else:
        X = vectorizer.transform(text)  # testing

    if (tfidf_transformer == 0):
        tfidf_transformer = TfidfTransformer()
        X_train_tf = tfidf_transformer.fit_transform(X)
        save_classifier("tfidf_transformer.pickle", tfidf_transformer)
    else:
        X_train_tf = tfidf_transformer.transform(X)  # testing

    return (X_train_tf, labels)


def train(pos_file, neg_file):
    training_set, label_set = obtaindata(pos_file, neg_file)
    print("Training Support vector machine")
    SVC_classifier = LinearSVC().fit(training_set, label_set)
    save_classifier("SVC.pickle", SVC_classifier)


def test(pos_file, neg_file, input_text=0):
    classifier = load_classifier("SVC.pickle")
    vectorizer = load_classifier("vectorizer.pickle")
    tfidf_transformer = load_classifier("tfidf_transformer.pickle")
    if (classifier == 0 or tfidf_transformer == 0 or vectorizer == 0):
        return 0

    if input_text == 0:
        testing_data, actual_labels = obtaindata(pos_file, neg_file, vectorizer, tfidf_transformer, input_text)
        predicted_labels = classifier.predict(testing_data)
        print("accuracy: "+str(np.mean(predicted_labels == actual_labels)*100)+"%")
    else:
        testing_data, actual_labels = obtaindata(0, 0, vectorizer, tfidf_transformer, input_text)
        predicted_labels = classifier.predict(testing_data)
        print(predicted_labels)


##Main Function


try:
    testing = eval(input("Press 1 to test files. Press 2 to test custom input.\nPress any other key to train\n"))
except:
    testing = 4
if (testing == 1):
    print("Testing Files Selected")
elif (testing == 2):
    print("Testing custom input selected")
else:
    print("Training Selected")
if (testing != 2):
    pos_file = input("Positive File Name: ")
    neg_file = input("Negative File Name: ")
    print("Positive file selected: " + pos_file + "\nNegative file name: " + neg_file)

if (testing == 1):
    print("Testing:...")
    if (test(pos_file, neg_file) == 0):
        print("Classifier Not Trained")
elif (testing == 2):
    input_text = input("Enter Text to categorize: ")
    if (test(0, 0, input_text) == 0):
        print("Classifier Not Trained")
else:
    print("Training:...")
    train(pos_file, neg_file)

print("done")
