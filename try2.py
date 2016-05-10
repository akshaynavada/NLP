__author__ = 'HetianWu'

import nltk
import pickle
import numpy as np
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import wordnet
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.tokenize import word_tokenize


import nltk
import os
import glob
from nltk.tokenize import word_tokenize
import codecs
import string
import numpy as np
from nltk.stem import SnowballStemmer
snowball_stemmer = SnowballStemmer('english')

class NB_Classifier:
# check if an input string is a valid word
# if it's a punctuation mark or purely digits
    def __init__ (self):
        self.word_counter = 0

        # the key is class' name and value is its index in the class list
        self.class_name_dict = {}

        # list for store all classes' names
        self.class_list = []

        # the key is the name of the file and value is its correct label
        self.doc_label = {}

        # use to count the cardnality of the output class set
        self.class_counter = 1

        # the key is class's name and value is the number of files in the class
        self.class_count_dict = {}

        # the key is class's name and value is the number of words in the class
        self.class_word_count = []

        # the number of files in the training class, together with the number
        # of files in a particular class, they can be used to compute the prior probability
        self.total_class_count = 0

        # the dictionay for storing all words in the training set
        # the length of this dictionary will be used for smoothing
        self.train_dict = {}

        # this is the total number of words in the training set
        # also the length of train_dict
        self.train_word_count = 0

        #self.test_file_label = []

        # number of correctly predicted labels
        self.correct_test_case = 0

        # total amount of instances being tested
        self.total_test_case = 0

        #self.train_path = ""

        # a list of dictionary, each dictionary has
        # all words in one class as the key, and number
        # of occurence as the value
        self.word_count_dict = []

    # determine if an input string is an valid word
    def valid_word(self,str):
        if(str in string.punctuation):
            return False
        if(str.isdigit()):
            return False
        return True

    # lower case and stem the input word
    def stem_lower(self,input_word):
        word_lower= string.lower(input_word)
        return snowball_stemmer.stem(word_lower)

    # open a file based on its relative path and
    # return the tokenized strings
    def open_read_tokenize(self,rel_path):
        path = os.getcwd() + "/"+rel_path
        f = open(path,'r')
        print f.read()
        f = open(path,'r')
        lines = f.readlines()
        sentences = []
        for line in lines:
            if line:
                sentences.append(line)
                sentences.append("   ")
        print sentences

        #return word_tokenize(f.read())
    # for a given class_dictionary and a given input file
    # we will update the class_dictionary based on the words
    # appeared in the input file
    def count_word_per_file(self,rel_path,class_dict):
        str = self.open_read_tokenize(rel_path)
        word_count = 0
        for word_ori in str:
            word = self.stem_lower(word_ori)
            if(self.valid_word(word)):
                word_count +=1
                if(class_dict.has_key(word)):
                    class_dict[word] = class_dict[word]+1
                else:
                    class_dict[word] = 1
        return word_count
    # in this function, we will update the dictionary for
    # the entire training dataset.
    def count_word(self,rel_path):
        str = self.open_read_tokenize(rel_path)
        for word_ori in str:
            word = self.stem_lower(word_ori)
            if(self.valid_word(word)):
                if(self.train_dict.has_key(word)):
                    self.train_dict[word] +=1
                else:
                    self.train_dict[word] = 1
    # Given a test document, this function will
    # return a predicted label
    def label_test_set(self,test_doc_token):
        prob_list = []
        for num in range(0,len(self.class_list)):
            prob = 0
            denominator = float(self.class_word_count[num]+self.train_word_count)
            for word_ori in test_doc_token:
                word = self.stem_lower(word_ori)
                if(self.valid_word(word)):
                    numerator = float(1)
                    if (self.word_count_dict[num].has_key(word)):
                        numerator = float(self.word_count_dict[num][word])
                    prob += np.log(float(numerator)/float(denominator))
            prior = np.log(float(self.class_count_dict[self.class_list[num]])/float(self.total_class_count))
            prob_list.append(prob+prior)
        max_index = prob_list.index(max(prob_list))
        return self.class_list[max_index]

    # this function will read all pairs of training files and their labels
    # if start and end are valid indices, the start-th file to the end-th file
    # in the training set will be excluded in the training process. Instead, they
    # will be reserved for cross-validation.
    # if start is -1, then there is no cross-validation.
    def read_and_train(self,label_path,start,end):
        print "training..."
        counter = 0
        for filename in label_path:
            if(not(counter <= end and counter >= start) or (start == -1) ):
                docname = filename.split()[0]
                classname = filename.split()[1]
                if(not self.class_name_dict.has_key(classname)):
                    self.class_name_dict[classname] = self.class_counter
                    self.class_counter +=1
                    self.class_count_dict[classname] = 1
                    self.class_list.append(classname)
                else:
                    self.class_count_dict[classname] += 1
                    self.doc_label[docname] = classname
                self.total_class_count += 1
            counter += 1
        for num in range(1,self.class_counter):
            empty_dict = {}
            self.word_count_dict.append(empty_dict)
            self.class_word_count.append(0)
        for key in self.doc_label:
            num_words = self.count_word_per_file(key,self.word_count_dict[self.class_name_dict[self.doc_label[key]]-1])
            self.class_word_count[self.class_name_dict[self.doc_label[key]]-1] += num_words
            self.count_word(key)
        self.train_word_count = len(self.train_dict)
        #print "word counter is ",self.train_word_count,"start is", start, "end is",end, "training number is",self.total_class_count

    def label_test_files(self,corpus,write_output):
        print "predicting labels for test files..."
        self.correct_test_case = 0
        self.total_test_case = 0
        test_label_path = os.getcwd()+"/"+corpus+"_test.labels"
        ftest = open(test_label_path,'r')
        if(write_output == True):
            output = open(corpus+"_predictions.labels",'w')
        for filename in ftest:
            docname = filename.split()[0]
            classname = filename.split()[1]
            data = self.open_read_tokenize(docname)
            label = self.label_test_set(data)
            if(write_output == True):
                output.write(docname+' '+label+'\n')
            if (label == classname):
                self.correct_test_case += 1
            self.total_test_case +=1
    # print out the accuracy
    """
    def print_accuracy(self):
        print "number of correct labels:",self.correct_test_case
        print "number of total test documetns:",self.total_test_case
        c = float(self.correct_test_case)
        t = float(self.total_test_case)
        print "Accuracy is :",c/t
    """
instance = NB_Classifier()
print instance.open_read_tokenize("ptrain.txt")