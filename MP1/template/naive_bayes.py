# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Last Modified 8/23/2023


"""
This is the main code for this MP.
You only need (and should) modify code within this file.
Original staff versions of all other files will be used by the autograder
so be careful to not modify anything else.
"""


import reader
import math
from tqdm import tqdm
from collections import Counter, defaultdict


'''
util for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

"""
load_data loads the input data by calling the provided utility.
You can adjust default values for stemming and lowercase, when we haven't passed in specific values,
to potentially improve performance.
"""
def load_data(trainingdir, testdir, stemming=True, lowercase=True, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


"""
Main function for training and predicting with naive bayes.
    You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def naiveBayes(dev_set, train_set, train_labels, laplace=10.0, pos_prior=0.8, silently=False):
    print_values(laplace,pos_prior)

    # a dictionary to store every count of word type in both negative and positive sets
    word_type_dict = defaultdict(lambda: {0: 0, 1: 0})
    
    # print("training...")
    for doc, label in tqdm(zip(train_set, train_labels), disable=silently):
        for word in doc:
            word_type_dict[word][label] += 1
    
    word_types_pos_count = sum(1 for value in word_type_dict.values() if value[1] > 0)
    word_types_neg_count = sum(1 for value in word_type_dict.values() if value[0] > 0)
    words_pos_count = sum(value[1] for value in word_type_dict.values() if value[1] > 0)
    words_neg_count = sum(value[0] for value in word_type_dict.values() if value[0] > 0)
    print("pos_count:", words_pos_count)
    print("neg_count:", words_neg_count)
    print("pos_type_count:", word_types_pos_count)
    print("neg_type_count:", word_types_neg_count)

    yhats = []
    denominator_pos = words_pos_count + (word_types_pos_count + 1) * laplace
    denominator_neg = words_neg_count + (word_types_neg_count + 1) * laplace
    # print("testing...")
    for doc in tqdm(dev_set, disable=silently):
        pos_value = math.log(pos_prior)
        neg_value = math.log(1 - pos_prior)
        for word in doc:
            # for unseen word, give an alpha mass
            if word not in word_type_dict:
                pos_value += math.log(laplace) - math.log(denominator_pos)
                neg_value += math.log(laplace) - math.log(denominator_neg)
            else:
                pos_value += math.log(word_type_dict[word][1] + laplace) - math.log(denominator_pos)
                neg_value += math.log(word_type_dict[word][0] + laplace) - math.log(denominator_neg)
        yhats.append(1 if pos_value > neg_value else 0)

    return yhats

if __name__ == "__main__":
    train_set, train_labels, dev_set, dev_labels=load_data("data/movie_reviews/train", "data/movie_reviews/dev")
    # print("train_set length:", len(train_set))
    # print("train_label length:", len(train_labels))
    # print("train_set[1]:", train_set[0])
    # print("train_label[1]:", train_labels[0])
    print(naiveBayes(dev_set=dev_set, train_set=train_set, train_labels=train_labels))