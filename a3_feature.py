import os
import string
import sys
import argparse
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import string
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy

punctuation_without_colon=list(string.punctuation)
#we want to keep ":" in the text, since they are used to strip headlines/signatures
punctuation_without_colon.remove(":")

data=[]

#Step 1: organise the data per author, without headlines and signatures

def format_data(directory:str)->list:
    '''This function pre-processes the data by cutting out the headers and signatures.
    Returns: a list of tuples, where tuple[0] is the author name, and tuple [1] is one entire email.'''
    path=directory #/scratch/lt2222-v23/enron_sample 
    for folder in os.listdir(path):
        for file in os.listdir(path+"/"+folder+"/"):
            email=[]
            with open(path+"/"+folder+"/"+file) as f:
                for line in f:
                    line = line.replace("\n", "") #cutting out the line jumps
                    for punct in punctuation_without_colon:
                        line = line.replace(punct, "").lower() #removing punctuation and capital letters
                    tokenized_sentences=word_tokenize(line) #tekenizing the sentence
                    if len(tokenized_sentences) > 2: #removing any signature such as "Chris" or "Chris Evans"
                         if tokenized_sentences[1] != ":": #removing headers such as eg. "Topic:"
                            email.append(line)
            data.append((folder," ".join(email)))
    return data



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert directories into table.")
    parser.add_argument("inputdir", type=str, help="The root of the author directories.")
    parser.add_argument("outputfile", type=str, help="The name of the output file containing the table of instances.")
    parser.add_argument("outputfiletest", type=str, help="The name of the output file containing the table of instances for the test set.")
    parser.add_argument("dims", type=int, help="The output feature dimensions.")
    parser.add_argument("--test", "-T", dest="testsize", type=int, default="20",
                        help="The percentage (integer) of instances to label as test.")

    args = parser.parse_args()

    print("Reading {}... It might take some time, but bear with me.".format(args.inputdir))
    
    #Pre-processing
    formatted_data = format_data(args.inputdir)
    table = pd.DataFrame(data=formatted_data)
    print("Here is a table with the non-vectorized data:\n",table)

    print("Constructing table with {} feature dimensions and {}% test instances...".format(args.dims, args.testsize))
    # Build the table here.

    #Defining y
    y=table[0]
    labelencoder=LabelEncoder()
    y=labelencoder.fit_transform(y)

    #Defining x
    x=table[1]
    vectorizer = CountVectorizer(max_features=int(args.dims))
    X = vectorizer.fit_transform(x)

    #Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.testsize/100,random_state=42)

    # Creating the feature table for the train set
    feature_table = pd.DataFrame(X_train.toarray(), columns=vectorizer.get_feature_names_out(), index=y_train)
    print("Here is the training table with the featurized data:",feature_table)
    
    print("Writing the training data to {}...".format(args.outputfile))

    # Printing the resulting table in a file
    feature_table_file = open(args.outputfile, "x")
    dfAsString = feature_table.to_string()
    feature_table_file.write(dfAsString)
    feature_table_file.close()

    print("Done! Let's do the test table now...")
    
    #Creating the feature table for the test set
    feature_table_test = pd.DataFrame(X_test.toarray(), columns=vectorizer.get_feature_names_out(), index=y_test)
        
    print("Writing the test data to {}...".format(args.outputfiletest))

    # Printing the resulting table in a file
    feature_table_file_test = open(args.outputfiletest, "x")
    dfAsString = feature_table_test.to_string()
    feature_table_file_test.write(dfAsString)
    feature_table_file_test.close()
    
    print("Done! You have two files, one with the training set, one with the test set.")