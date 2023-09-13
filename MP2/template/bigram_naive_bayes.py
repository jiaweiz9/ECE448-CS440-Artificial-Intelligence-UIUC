# bigram_naive_bayes.py
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
from collections import Counter


'''
utils for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

def print_values_bigram(unigram_laplace, bigram_laplace, bigram_lambda, pos_prior):
    print(f"Unigram Laplace: {unigram_laplace}")
    print(f"Bigram Laplace: {bigram_laplace}")
    print(f"Bigram Lambda: {bigram_lambda}")
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
Main function for training and predicting with the bigram mixture model.
    You can modify the default values for the Laplace smoothing parameters, model-mixture lambda parameter, and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def bigramBayes(dev_set, train_set, train_labels, unigram_laplace=10.0, bigram_laplace=1.0, bigram_lambda=0.9, pos_prior=0.8, silently=False):
    print_values_bigram(unigram_laplace, bigram_laplace, bigram_lambda, pos_prior)
    pos_bigram_counter = Counter()
    neg_bigram_counter = Counter()
    for doc, label in tqdm(zip(train_set, train_labels), disable=silently):
        for fist_word, second_word in zip(doc[0:-2], doc[1:-1]):
            bigram = fist_word + ' ' + second_word
            if label == 1:
                pos_bigram_counter[bigram] = pos_bigram_counter.get(bigram, 0) + 1
            else:
                neg_bigram_counter[bigram] = neg_bigram_counter.get(bigram, 0) + 1
    pos_bigram_total = sum(pos_bigram_counter.values())
    neg_bigram_total = sum(neg_bigram_counter.values())
    pos_bigram_types = len(pos_bigram_counter)
    neg_bigram_types = len(neg_bigram_counter)
    # print("pos_bigram_total:", pos_bigram_total)
    # print("neg_bigram_total:", neg_bigram_total)
    # print("pos_bigram_types:", pos_bigram_types)
    # print("neg_bigram_types:", neg_bigram_types)

    pos_unigram_counter = Counter()
    neg_unigram_counter = Counter()
    for doc, label in tqdm(zip(train_set, train_labels), disable=silently):
        for word in doc:
            if label == 1:
                pos_unigram_counter[word] = pos_unigram_counter.get(word, 0) + 1
            else:
                neg_unigram_counter[word] = neg_unigram_counter.get(word, 0) + 1
    pos_unigram_total = sum(pos_unigram_counter.values())
    neg_unigram_total = sum(neg_unigram_counter.values())
    pos_unigram_types = len(pos_unigram_counter)
    neg_unigram_types = len(neg_unigram_counter)
    # print("pos_unigram_total:", pos_unigram_total)
    # print("neg_unigram_total:", neg_unigram_total)
    # print("pos_unigram_types:", pos_unigram_types)
    # print("neg_unigram_types:", neg_unigram_types)

    pos_unigram_denominator = pos_unigram_total + (pos_unigram_types + 1) * unigram_laplace
    neg_unigram_denominator = neg_unigram_total + (neg_unigram_types + 1) * unigram_laplace
    pos_bigram_denominator = pos_bigram_total + (pos_bigram_types + 1) * bigram_laplace
    neg_bigram_denominator = neg_bigram_total + (neg_bigram_types + 1) * bigram_laplace

    yhats = []
    for doc in tqdm(dev_set, disable=silently):
        pos_unigram_value = math.log(pos_prior)
        neg_unigram_value = math.log(1 - pos_prior)
        pos_bigram_value = math.log(pos_prior)
        neg_bigram_value = math.log(1 - pos_prior)
        # tests for unigram:
        for word in doc:
            if word not in pos_unigram_counter:
                pos_unigram_value += math.log(unigram_laplace) - math.log(pos_unigram_denominator)
            else:
                pos_unigram_value += math.log(pos_unigram_counter[word]) - math.log(pos_unigram_denominator)
            if word not in neg_unigram_counter:
                neg_unigram_value += math.log(unigram_laplace) - math.log(neg_unigram_denominator)
            else:
                neg_unigram_value += math.log(neg_unigram_counter[word]) - math.log(neg_unigram_denominator)
        # tests for bigram:
        for first_word, second_word in zip(doc[0:-2], doc[1:-1]):
            bigram = first_word + ' ' + second_word
            if bigram not in pos_bigram_counter:
                pos_bigram_value += math.log(bigram_laplace) - math.log(pos_bigram_denominator)
            else:
                pos_bigram_value += math.log(pos_bigram_counter[bigram]) - math.log(pos_bigram_denominator)
            if bigram not in neg_bigram_counter:
                neg_bigram_value += math.log(bigram_laplace) - math.log(pos_bigram_denominator)
            else:
                neg_bigram_value += math.log(neg_bigram_counter[bigram]) - math.log(neg_bigram_denominator)
        pos_posibility = (1 - bigram_lambda) * pos_unigram_value + bigram_lambda * pos_bigram_value
        neg_posibility = (1 - bigram_lambda) * neg_unigram_value + bigram_lambda * neg_bigram_value
        yhats.append(1 if pos_posibility > neg_posibility else 0)
    
    return yhats



