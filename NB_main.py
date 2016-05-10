__author__ = 'HetianWu'
import NB



alpha = 0.65

def saveOutput(message,accuracy,outfile):
    outf = open(outfile,"a")
    str2 = message + "," + str(accuracy)+"\n"
    outf.write(str2)
    outf.close()



def grid_search(unigram,bigram,alpha,add_tag):
    instance = NB.SA_NB_Classifier()
    train_p_doc_set = instance.doc_set_generator("positive.train",add_tag)
    train_n_doc_set = instance.doc_set_generator("negative.train",add_tag)

    instance.read_and_train(train_p_doc_set,"pos",unigram,bigram,binary)
    instance.read_and_train(train_n_doc_set,"neg",unigram,bigram,binary)


    test_p_doc_set = instance.doc_set_generator("positive.test",add_tag)
    test_n_doc_set = instance.doc_set_generator("negative.test",add_tag)

    instance.label_test_files(test_p_doc_set,"pos",unigram,bigram,binary,alpha)
    instance.label_test_files(test_n_doc_set,"neg",unigram,bigram,binary,alpha)


    outfile = "result_nb_new4"
    accuracy = instance.print_accuracy()
    msg = "Unigram: "+str(unigram)+";Bigram: "+ str(bigram) + "; Binary: "+ str(binary) + ";Alpha: "+ str(alpha) + ";add_tag "+str(add_tag)
    saveOutput(msg,accuracy,outfile)


for unig_num in range(0,2):
    for big_num in range(0,2):
        for bin_num in range(0,2):
            for add_tag_num in range(0,2):
                if(unig_num == 0):
                    unigram = False
                else:
                    unigram = True
                if(big_num == 0):
                    bigram = False
                else:
                    bigram = True
                if(bin_num == 0):
                    binary = False
                else:
                    binary = True

                if(add_tag_num == 1):
                    add_tag = False
                else:
                    add_tag = True

                msg = "Unigram: "+str(unigram)+";Bigram: "+ str(bigram) + "; Binary: "+ str(binary) + ";Alpha: "+ str(alpha) + ";add_tag "+str(add_tag)
                if(unigram or bigram):
                    grid_search(unigram,bigram,alpha,add_tag)