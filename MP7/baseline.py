"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""
from collections import defaultdict, Counter

def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    word_tag = defaultdict(lambda: defaultdict(lambda: 0))
    list_tag = [word[1] for sentence in train for word in sentence]
    tag_count = Counter(list_tag)
    for sentence in train:
        for word in sentence:
            word_tag[word[0]][word[1]] += 1
    print(tag_count)
    output = []
    for sentence in test:
        tag_sentence = []
        for word in sentence:
            if word in word_tag.keys():
                tag_sentence.append((word, max(word_tag[word], key=word_tag[word].get)))
            else:
                tag_sentence.append((word, max(tag_count, key=tag_count.get)))
        output.append(tag_sentence)
    return output