"""
Part 4: Here should be your best version of viterbi, 
with enhancements such as dealing with suffixes/prefixes separately
"""
import math
import re
from collections import defaultdict, Counter
from math import log

# Note: remember to use these two elements when you find a probability is 0 in the training data.
epsilon_for_pt = 1e-5
alpha = 0.0001   # exact setting seems to have little or no effect

def pattern_match(pattern, word): # patten is ly, ing
	match = re.search(pattern, word)
	return match

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
	hapax_word = {}
	for sentence in sentences:
		# print("sentence", sentence)
		for word_pre, word_post in zip(sentence[0:-1], sentence[1:]):
			emit_prob[word_pre[1]][word_pre[0]] += 1
			trans_prob[word_pre[1]][word_post[1]] += 1
			#word_count += 1
		emit_prob[sentence[-1][1]][sentence[-1][0]] += 1

		for word, tag in sentence:
			if word in hapax_word:
				hapax_word[word] = None
			else:
				hapax_word[word] = tag
	print(emit_prob.keys())
	#word_counter = Counter([word for sentence in sentences for word, tag in sentence])
	#hapax_word = [word for word in word_counter.keys() if word_counter[word] == 1]
	#numOfHapax = len(hapax_word)
	hapax_word_tag = [(word, tag) for word, tag in hapax_word.items() if tag is not None]
	numOfHapax = len(hapax_word_tag)
	print(len(hapax_word_tag))
	#print(hapax_word_tag)

	hapax_un = [(word, tag) for word, tag in hapax_word_tag if pattern_match('^un', word)]
	hapax_re = [(word, tag) for word, tag in hapax_word_tag if pattern_match('^re', word)]
	hapax_ing = [(word, tag) for word, tag in hapax_word_tag if pattern_match('ing$', word)]
	hapax_ly = [(word, tag) for word, tag in hapax_word_tag if pattern_match('ly$', word)]
	hapax_s = [(word, tag) for word, tag in hapax_word_tag if pattern_match('s$', word)]
	hapax_ed = [(word, tag) for word, tag in hapax_word_tag if pattern_match('ed$', word)]
	hapax_ion = [(word, tag) for word, tag in hapax_word_tag if pattern_match('ion$', word)]
	hapax_num = [(word, tag) for word, tag in hapax_word_tag if pattern_match('\d$', word)]

	hapax_other = [(word, tag) for word, tag in hapax_word_tag 
					 		if (word, tag) not in hapax_ing and (word, tag) not in hapax_ly and (word, tag) not in hapax_s 
							and (word, tag) not in hapax_ed and (word, tag) not in hapax_re and (word, tag) not in hapax_un
							and (word, tag) not in hapax_ion and (word, tag) not in hapax_num]
	# numOfING = len(hapax_ing)
	# numOfLY = len(hapax_ly)
	# numOfS = len(hapax_s)
	# numOfED = len(hapax_ed)
	# numOfOTHER = len(hapax_other)
	hapax_un_dict = defaultdict(lambda: 0)
	hapax_re_dict = defaultdict(lambda: 0)
	hapax_ing_dict = defaultdict(lambda: 0)
	hapax_ly_dict = defaultdict(lambda: 0)
	hapax_s_dict = defaultdict(lambda: 0)
	hapax_ed_dict = defaultdict(lambda: 0)
	hapax_ion_dict = defaultdict(lambda: 0)
	hapax_num_dict = defaultdict(lambda: 0)
	hapax_other_dict = defaultdict(lambda: 0)

	for word, tag in hapax_un:
		hapax_un_dict[tag] += 1
	for word, tag in hapax_re:
		hapax_re_dict[tag] += 1
	for word, tag in hapax_ing:
		hapax_ing_dict[tag] += 1
	for word, tag in hapax_ly:
		hapax_ly_dict[tag] += 1
	for word, tag in hapax_s:
		hapax_s_dict[tag] += 1
	for word, tag in hapax_ed:
		hapax_ed_dict[tag] += 1
	for word, tag in hapax_ion:
		hapax_ion_dict[tag] += 1
	for word, tag in hapax_num:
		hapax_num_dict[tag] += 1
	for word, tag in hapax_other:
		hapax_other_dict[tag] += 1

	print("un tags", hapax_un_dict)
	print("re tags", hapax_re_dict)
	print("ing tags",hapax_ing_dict)
	print("ly tags", hapax_ly_dict)
	print("s tags", hapax_s_dict)
	print("ed tags", hapax_ed_dict)
	print("ion tags", hapax_ion_dict)
	print("num tags", hapax_num_dict)
	print("other tags", hapax_other_dict)
	
	for tag in emit_prob.keys():
		if tag in hapax_un_dict.keys():
			emit_prob[tag]['UNSEEN-UN'] = alpha * hapax_un_dict[tag] / numOfHapax
		else:
			emit_prob[tag]['UNSEEN-UN'] = alpha * 1 / numOfHapax

		if tag in hapax_re_dict.keys():
			emit_prob[tag]['UNSEEN-RE'] = alpha * hapax_re_dict[tag] / numOfHapax
		else:
			emit_prob[tag]['UNSEEN-RE'] = alpha * 1 / numOfHapax

		if tag in hapax_ing_dict.keys():
			emit_prob[tag]['UNSEEN-ING'] = alpha * hapax_ing_dict[tag] / numOfHapax
		else:
			emit_prob[tag]['UNSEEN-ING'] = alpha * 1 / numOfHapax

		if tag in hapax_ly_dict.keys():
			emit_prob[tag]['UNSEEN-LY'] = alpha * hapax_ly_dict[tag] / numOfHapax
		else:
			emit_prob[tag]['UNSEEN-LY'] = alpha * 1 / numOfHapax

		if tag in hapax_s_dict.keys():
			emit_prob[tag]['UNSEEN-S'] = alpha * hapax_s_dict[tag] / numOfHapax
		else:
			emit_prob[tag]['UNSEEN-S'] = alpha * 1 / numOfHapax

		if tag in hapax_ed_dict.keys():
			emit_prob[tag]['UNSEEN-ED'] = alpha * hapax_ed_dict[tag] / numOfHapax
		else:
			emit_prob[tag]['UNSEEN-ED'] = alpha * 1 / numOfHapax

		if tag in hapax_ion_dict.keys():
			emit_prob[tag]['UNSEEN-ION'] = alpha * hapax_ion_dict[tag] / numOfHapax
		else:
			emit_prob[tag]['UNSEEN-ION'] = alpha * 1 / numOfHapax

		if tag in hapax_num_dict.keys():
			emit_prob[tag]['UNSEEN-NUM'] = alpha * hapax_num_dict[tag] / numOfHapax
		else:
			emit_prob[tag]['UNSEEN-NUM'] = alpha * 1 / numOfHapax

		if tag in hapax_other_dict.keys():
			emit_prob[tag]['UNSEEN-OTHER'] = alpha * hapax_other_dict[tag] / numOfHapax
		else:
			emit_prob[tag]['UNSEEN-OTHER'] = alpha * 1 / numOfHapax
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
			#emit_prob_cur = emit_prob[tag][word] if word in emit_prob[tag].keys() else emit_prob[tag]['UNSEEN']
			if word in emit_prob[tag].keys():
				emit_prob_cur = emit_prob[tag][word]
			elif pattern_match('^un', word):
				emit_prob_cur = emit_prob[tag]['UNSEEN-UN']
			elif pattern_match('^re', word):
				emit_prob_cur = emit_prob[tag]['UNSEEN-RE']
			elif pattern_match('ing$', word):
				emit_prob_cur = emit_prob[tag]['UNSEEN-ING']
			elif pattern_match('ly$', word):
				emit_prob_cur = emit_prob[tag]['UNSEEN-LY']
			elif pattern_match('s$', word):
				emit_prob_cur = emit_prob[tag]['UNSEEN-S']
			elif pattern_match('ed$', word):
				emit_prob_cur = emit_prob[tag]['UNSEEN-ED']
			elif pattern_match('ion$', word):
				emit_prob_cur = emit_prob[tag]['UNSEEN-ION']
			elif pattern_match('\d$', word):
				emit_prob_cur = emit_prob[tag]['UNSEEN-NUM']
			else:
				emit_prob_cur = emit_prob[tag]['UNSEEN-OTHER']
			
			beam_search_dict = dict(sorted(prev_prob.items(), key=lambda item: item[1], reverse=True)[0:3])
			#emit_prob_cur = emit_prob[tag][word] if word in emit_prob[tag].keys() else emit_prob[tag]['UNSEEN']
			max_tag_prev_key = max(beam_search_dict.keys(), key=lambda key: bayes_estimate(beam_search_dict[key], trans_prob[key][tag], 
																					emit_prob_cur))
			log_prob[tag] = bayes_estimate(prev_prob[max_tag_prev_key], trans_prob[max_tag_prev_key][tag], 
										emit_prob_cur)

			predict_tag_seq[tag] = prev_predict_tag_seq[max_tag_prev_key]+ [(word, tag)]
   
	return log_prob, predict_tag_seq

def viterbi_3(train, test):
	'''
	input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
			test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
	output: list of sentences, each sentence is a list of (word,tag) pairs.
			E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
	'''
	init_prob, emit_prob, trans_prob = training(train)
	
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