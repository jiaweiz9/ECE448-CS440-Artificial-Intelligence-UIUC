"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""

import math
from collections import defaultdict, Counter
from math import log

# Note: remember to use these two elements when you find a probability is 0 in the training data.
epsilon_for_pt = 1e-5
alpha = 0.001   # exact setting seems to have little or no effect


def training(sentences):
    """
    Computes initial tags, emission words and transition tag-to-tag probabilities
    :param sentences:
    :return: intitial tag probs, emission words given tag probs, transition of tags to tags probs
    """
    init_prob = defaultdict(lambda: 0) # {init tag: #}
    emit_prob = defaultdict(lambda: defaultdict(lambda: 0)) # {tag: {word: # }}
    trans_prob = defaultdict(lambda: defaultdict(lambda: 0)) # {tag0:{tag1: # }}
    
    # TODO: (I)
    # Input the training set, output the formatted probabilities according to data statistics.
    #word_count = 0
    #print("length of sentences", len(sentences))
    for sentence in sentences:
        # print("sentence", sentence)
        for word_pre, word_post in zip(sentence[0:-1], sentence[1:]):
            emit_prob[word_pre[1]][word_pre[0]] += 1
            trans_prob[word_pre[1]][word_post[1]] += 1
            #word_count += 1
        emit_prob[sentence[-1][1]][sentence[-1][0]] += 1
    

    for tag in emit_prob.keys():
        emit_prob[tag]['UNSEEN'] += alpha
        sum_of_words_for_this_tag = sum(emit_prob[tag].values())
        for word in emit_prob[tag]:
            emit_prob[tag][word] /= sum_of_words_for_this_tag

    for tag_pre in trans_prob.keys():
        sum_of_tags_for_this_tag = sum(trans_prob[tag_pre].values())
        for tag_post in trans_prob[tag_pre]:
            trans_prob[tag_pre][tag_post] /= sum_of_tags_for_this_tag
    
    # print('emit_prob', emit_prob['DET'])
    # print('trans_prob', trans_prob)
    init_prob['START'] = 1

    return init_prob, emit_prob, trans_prob

def viterbi_stepforward(i, word, prev_prob, prev_predict_tag_seq, emit_prob, trans_prob):
    """
    Does one step of the viterbi function
    :param i: The i'th column of the lattice/MDP (0-indexing)
    :param word: The i'th observed word
    :param prev_prob: A dictionary of tags to probs representing the max probability of getting to each tag at in the
    previous column of the lattice
    :param prev_predict_tag_seq: A dictionary representing the predicted tag sequences leading up to the previous column
    of the lattice for each tag in the previous column
    :param emit_prob: Emission probabilities
    :param trans_prob: Transition probabilities
    :return: Current best log probs leading to the i'th column for each tag, and the respective predicted tag sequences
    """
    log_prob = {} # This should store the log_prob for all the tags at current column (i)
    predict_tag_seq = {} # This should store the tag sequence to reach each tag at column (i)

    # TODO: (II)
    # implement one step of trellis computation at column (i)
    # You should pay attention to the i=0 special case.
    tag_list = emit_prob.keys()
    
    def bayes_estimate(prev_prob, trans_prob, emit_prob):
        #print()
        if trans_prob == 0:
            trans_prob = epsilon_for_pt
        
        return prev_prob + log(trans_prob) + log(emit_prob)

    for tag in tag_list:
        if i == 0:
            predict_tag_seq[tag] = [] + [(word, tag)]
            log_prob[tag] = prev_prob[tag]
        else:
            emit_prob_cur = emit_prob[tag][word] if word in emit_prob[tag].keys() else emit_prob[tag]['UNSEEN']
            
            #emit_prob_cur = emit_prob[tag][word] if word in emit_prob[tag].keys() else emit_prob[tag]['UNSEEN']
            max_tag_prev_key = max(prev_prob.keys(), key=lambda key: bayes_estimate(prev_prob[key], trans_prob[key][tag], 
                                                                                    emit_prob_cur))
            log_prob[tag] = bayes_estimate(prev_prob[max_tag_prev_key], trans_prob[max_tag_prev_key][tag], 
                                        emit_prob_cur)

            predict_tag_seq[tag] = prev_predict_tag_seq[max_tag_prev_key]+ [(word, tag)]
   
    return log_prob, predict_tag_seq

def viterbi_1(train, test, get_probs=training):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    init_prob, emit_prob, trans_prob = get_probs(train)
    
    predicts = []
    for sen in range(len(test)):
        sentence=test[sen]
        length = len(sentence)
        log_prob = {}
        predict_tag_seq = {}
        # init log prob
        for t in emit_prob:
            if t in init_prob:
                log_prob[t] = log(init_prob[t])
            else:
                log_prob[t] = log(epsilon_for_pt)
            predict_tag_seq[t] = []

        # forward steps to calculate log probs for sentence
        #print(sentence)
        for i in range(length):
            log_prob, predict_tag_seq = viterbi_stepforward(i, sentence[i], log_prob, predict_tag_seq, emit_prob, trans_prob)
            
        # TODO:(III) 
        # according to the storage of probabilities and sequences, get the final prediction.
        max_key = max(log_prob, key = log_prob.get)
        predicts.append(predict_tag_seq[max_key])
        #print(predicts)
    return predicts




