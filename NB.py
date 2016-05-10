__author__ = 'HetianWu'

import nltk
import os
import glob
from nltk.tokenize import word_tokenize
import codecs
import string
import numpy as np
from nltk.util import ngrams
from nltk.stem import SnowballStemmer
snowball_stemmer = SnowballStemmer('english')


import nltk
import pickle
import numpy as np
import string
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import wordnet
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import BernoulliNB, MultinomialNB


class SA_NB_Classifier:
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

    # determine if an input
    #  string is an valid word
    def valid_word(self,str):
        return True
        """
        if(str in string.punctuation):
            return False
        if(str.isdigit()):
            return False
        return True
        """
    def is_ascii(self,s):
        return all(ord(c)<128 for c in s)

    # remove non-ascii character
    def remove_non_ascii(self,text):
        return ''.join(i for i in text if ord(i)<128)
    # lower case and stem the input word
    def stem_lower(self,input_word):
        if(self.is_ascii(input_word)):
            word_lower= string.lower(input_word)
            return snowball_stemmer.stem(word_lower)
        else:
            return ""


    def negator(self,wordVec):
        negation = False
        negated_doc = []
        lemmatizer = WordNetLemmatizer()
        for w,p in wordVec:
            w_out = ""
            if (p[:2] == "NN"):
                w_out = lemmatizer.lemmatize(w.lower(), pos=wordnet.NOUN)
            elif (p[:2] == "JJ"):
                w_out = lemmatizer.lemmatize(w.lower(), pos=wordnet.ADJ)
            elif (p[:2] == "VB"):
                w_out = lemmatizer.lemmatize(w.lower(), pos=wordnet.VERB)
            elif (p[:2] == "RB"):
                w_out = lemmatizer.lemmatize(w.lower(), pos=wordnet.ADV)
            if(w_out == "not" or w_out == "n't" ):
                #print "blah"
                negation = not negation
                #rint negation
            elif(w_out in string.punctuation and w_out != ''):

                negation = False
            elif(negation):
                #print negation
                w_out = "NOT_"+w_out
            negated_doc.append((w_out,p))
        #print negated_doc
        return negated_doc

    def processDocument(self,doc,add_tag):
        lemmatizer = WordNetLemmatizer()
        word = nltk.word_tokenize(self.remove_non_ascii(doc))
        tagged = nltk.pos_tag(word)
        negated_words = self.negator(tagged)

        tagged_word = ""
        for w, p in negated_words:
            if (p[:2] == "NN" or p[:2] == "JJ" or p[:2] == "VB" or p[:2] == "RB"):
                if(add_tag):
                    tagged_word += w + "_"+p+" "
                else:
                    tagged_word += w + " "
        return tagged_word

    def doc_set_generator(self,rel_path,add_tag = False):
        path = os.getcwd() + "/"+rel_path
        f = open(path,'r')
        doc_set = []
        for doc in f.read().split('\n'):
            doc_set.append(self.processDocument(doc,add_tag))
        return doc_set
    #return word_tokenize(f.read())
    # for a given class_dictionary and a given input file
    # we will update the class_dictionary based on the words
    # appeared in the input file
    def count_word_per_file(self,doc,class_dict,unigram = True,bigram = False ,binary =False):
        str = word_tokenize(self.remove_non_ascii(doc))
        word_list = []
        word_count = 0
        if(unigram):
            uni = ngrams(str,1)
            word_count += self.count_word_per_file_sub(class_dict,uni,binary)

        if(bigram):
            bi = ngrams(str,2)
            word_count += self.count_word_per_file_sub(class_dict,bi,binary)

        return word_count

    def count_word_per_file_sub(self,class_dict,grams,binary):
        word_count = 0
        doc_voc = {}
        for word_ori in grams:
            word = word_ori
            #word = self.stem_lower(word_ori)
            if(doc_voc.has_key(word) and binary == False):
                if(self.valid_word(word)):
                    word_count +=1
                    if(class_dict.has_key(word)):
                        class_dict[word] = class_dict[word]+1
                    else:
                        class_dict[word] = 1
            else:
                doc_voc[word] = 1

                if(self.valid_word(word)):
                    word_count +=1
                    if(class_dict.has_key(word)):
                        class_dict[word] = class_dict[word]+1
                    else:
                        class_dict[word] = 1

        return word_count

    # in this function, we will update the dictionary for
    # the entire training dataset.


    def count_word(self,doc,unigram = True,bigram = False,binary = False):
        str = word_tokenize(self.remove_non_ascii(doc))
        doc_voc = {}
        if(unigram):
            uni = ngrams(str,1)
            self.count_word_sub(doc_voc,uni,binary)

        if(bigram):
            bi = ngrams(str,2)
            self.count_word_sub(doc_voc,bi,binary)



    def count_word_sub(self,doc_voc,gram,binary):
        for word in gram:
            if(doc_voc.has_key(word)):
                if(binary == False):
                    if(self.valid_word(word)):
                        if(self.train_dict.has_key(word)):
                            self.train_dict[word] +=1
                        else:
                            self.train_dict[word] = 1
            else:
                doc_voc[word] = 1

                if(self.valid_word(word)):
                    if(self.train_dict.has_key(word)):
                        self.train_dict[word] +=1
                    else:
                        self.train_dict[word] = 1

    def uni_bi_gram(self,doc,unigram,bigram):
        ret_list = []
        if(unigram):
            uni = ngrams(doc,1)
            for gram in uni:
                ret_list.append(gram)
        if(bigram):
            bi = ngrams(doc,2)
            for gram in bi:
                ret_list.append(gram)
        return ret_list



    # Given a test document, this function will
    # return a predicted label
    def label_test_set(self,doc,unigram,bigram,binary,alpha):

        test_doc = word_tokenize(self.remove_non_ascii(doc))
        test_doc_token = self.uni_bi_gram(test_doc,unigram,bigram)
        prob_list = []
        for num in range(0,len(self.class_list)):
            prob = 0
            denominator = float(self.class_word_count[num])+ float(self.train_word_count)*alpha
            test_doc_word = {}
            test_doc_token_new = []
            if(binary == True):
                for word  in test_doc_token:
                    if(not test_doc_word.has_key(word)):
                        test_doc_word[word] = 1
                        test_doc_token_new.append(word)


            else:
                test_doc_token_new = test_doc_token

            for word_ori in test_doc_token_new:
                word = word_ori



                if(self.valid_word(word)):
                    numerator = float(alpha)
                    if (self.word_count_dict[num].has_key(word)):
                        numerator = float(self.word_count_dict[num][word])
                    prob += np.log(float(numerator)/float(denominator))
            prior = np.log(float(self.class_count_dict[self.class_list[num]])/float(self.total_class_count))
            prob_list.append(prob+prior)
        max_index = prob_list.index(max(prob_list))
        return self.class_list[max_index]



    def read_and_train(self,doc_set,sentiment,unigram,bigram,binary):
        print "adding files..."
        counter = 0
        classname = sentiment
        for doc in doc_set:
            if(not self.class_name_dict.has_key(classname)):
                self.class_name_dict[classname] = self.class_counter
                self.class_counter +=1
                self.class_count_dict[classname] = 1
                self.class_list.append(classname)
            else:
                self.class_count_dict[classname] += 1
                self.total_class_count += 1
            counter += 1

        empty_dict = {}
        self.word_count_dict.append(empty_dict)
        self.class_word_count.append(0)

        for doc in doc_set:
            num_words = self.count_word_per_file(doc,self.word_count_dict[len(self.word_count_dict)-1],unigram,bigram,binary)
            self.class_word_count[len(self.class_word_count)-1] += num_words
            self.count_word(doc,unigram,bigram,binary)

        self.train_word_count = len(self.train_dict)
        #print "word counter is ",self.train_word_count,"start is", start, "end is",end, "training number is",self.total_class_count

    def label_test_files(self,doc_set,sentiment,unigram,bigram,binary,alpha =1):
        print "predicting labels for test files..."
        self.correct_test_case = 0
        self.total_test_case = 0
        for doc in doc_set:
            #print doc
            label = self.label_test_set(doc,unigram,bigram,binary,alpha)
            if (label == sentiment):
                self.correct_test_case += 1
            self.total_test_case +=1
    # print out the accuracy

    def print_accuracy(self):
        print "number of correct labels:",self.correct_test_case
        print "number of total test documetns:",self.total_test_case
        c = float(self.correct_test_case)
        t = float(self.total_test_case)
        print "Accuracy is :",c/t
        return c/t








