import argparse
import numpy as np
import pandas as pd
import os
import random
import string
import sys
from collections import Counter
from glob import glob
from sklearn.decomposition import PCA

CLASS_COLUMN = "classname"
FILE_COLUMN = "filename"

def __build_class_model(files, class_name, corpus_list):
    for filename in files:
        document_dict = {CLASS_COLUMN : class_name, FILE_COLUMN : filename}
        filestring = ""
        with open(filename, "r") as thefile:
            for line in thefile:
                filestring += "\n" + line
        text = filestring.lower().split() # lowercase so the words are not treated differently if they appear at the beginning of a sentence.
        text = [word for word in text if word not in string.punctuation] # removes punctuation marks
        document_dict.update(Counter(text))
        corpus_list.append(document_dict)
    return corpus_list

def reduce_dim(X,n=10):
    pca = PCA(n_components=n)
    return pca.fit_transform(X)

def shuffle_split(X,testsize=20):
    indexes_list = list(range(X.shape[0])) # Creates a list of indexes that can be shuffled without losing the parallelism between X and y
    random.shuffle(indexes_list)
    train_perc = (100 - testsize) / 100
    last_training_index = int(train_perc * X.shape[0]) # Calculates the last index to separate the data into two parts with ratio 80/20
    indexes_test = indexes_list[last_training_index:] # Chooses which indexes of the shuffled list will become part of the training or test set
    train_or_test_col_X = np.zeros((X.shape[0],1))
    train_or_test_col_X[indexes_test] = 1
    X = np.concatenate((X,train_or_test_col_X), axis=1)
    return X


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert directories into table.")
    parser.add_argument("inputdir", type=str, help="The root of the author directories.")
    parser.add_argument("outputfile", type=str, help="The name of the output file containing the table of instances.")
    parser.add_argument("dims", type=int, help="The output feature dimensions.")
    parser.add_argument("--test", "-T", dest="testsize", type=int, default="20", help="The percentage (integer) of instances to label as test.")

    args = parser.parse_args()

    print("Reading {}...".format(args.inputdir))
    subdirs = glob(args.inputdir + "/*")
    corpus_list = []
    for subdir in subdirs:
        allfiles = glob("{}/*".format(subdir))
        corpus_list = __build_class_model(allfiles, subdir, corpus_list)

    print("Constructing table with {} feature dimensions and {}% test instances...".format(args.dims, args.testsize))
    corpus_df = pd.DataFrame(corpus_list)
    corpus_df = corpus_df.fillna(0)
    y = corpus_df[CLASS_COLUMN]
    filenames = corpus_df[FILE_COLUMN]
    corpus = corpus_df.drop([CLASS_COLUMN, FILE_COLUMN], axis = 1)
    X = corpus.to_numpy()
    X = reduce_dim(X, args.dims)
    X = shuffle_split(X, args.testsize)
    
    print("Writing to {}...".format(args.outputfile))
    with open(args.outputfile, "a") as f:
        for index_row in range(X.shape[0]):
            line = [filenames[index_row]]
            line.extend([str(val) for val in X[index_row]])
            line.append(str(y[index_row]))
            line.append("\n")
            f.write(",".join(line))

    print("Done!")
    
