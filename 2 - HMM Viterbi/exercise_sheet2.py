################################################################################
## SNLP exercise sheet 2
################################################################################

from collections import defaultdict
from collections import Counter
import random
import numpy as np
from math import log, inf
from functools import reduce

'''
This function can be used for importing the corpus.
Parameters: path_to_file: string; path to the file containing the corpus
Returns: list of list; the second layer list contains tuples (token,label);
'''
def import_corpus(path_to_file):
    sentences = []
    sentence = []
    f = open(path_to_file)
    
    while True:
        line = f.readline()
        if not line: break
            
        line = line.strip()
        if len(line) == 0:
            sentences.append(sentence)    
            sentence = []
            continue
                
        parts = line.split(' ')
        sentence.append((parts[0], parts[-1]))
        
    f.close()        
    return sentences
    
UNKNOWN_T = '<unknown>'

def preprocess(corpus, token_frequencies):

    for sentence in corpus:
        i = 0
        while i < len(sentence):
            token, tag = sentence[i]
            assert token_frequencies[token] != 0

            if token_frequencies[token] == 1:
                sentence[i] = (UNKNOWN_T, tag)
            i += 1


# Exercise 1 ###################################################################
'''
Implement the probability distribution of the initial states.
Parameters:
    state: string
    internal_representation: dictionary containing the probabilities that a state is the initial one;
        this data structure is returned by the function estimate_initial_state_probabilities
Returns: float; initial probability of the given state
'''
def initial_state_probabilities(state, internal_representation):
    return internal_representation[state]


'''
Implement the matrix of transition probabilities.
Parameters:
    from_state: string;
    to_state: string;
    internal_representation: dictionary of dictionaries containing the conditional probability
        of the appearence of a state, given the appearence its predecessor.
        this data structure is returned by the function estimate_transition_probabilities
Returns: float; probability of transition from_state -> to_state
'''
def transition_probabilities(from_state, to_state, internal_representation):
    internal_representation[from_state][to_state]

    
'''
Implement the matrix of emmision probabilities.
Parameters:
    state: string;
    emission_symbol: string;
    internal_representation: dictionary of dictionaries containing the conditional probability
        of the appearence of an observation, given that we know its state.
        this data structure is returned by the function estimate_emission_probabilities
Returns: float; emission probability of the symbol emission_symbol if the current state is state
'''
def emission_probabilities(state, emission_symbol, internal_representation):
    internal_representation[state][emission_symbol]


'''
Implement a function for estimating the parameters of the probability distribution of the initial states.
Parameters: corpus: list returned by the function import_corpus
Returns: dictionary containing the parameters of the probability distribution of the initial states;
            use this data structure for the argument internal_representation of the function initial_state_probabilities
'''
def estimate_initial_state_probabilities(corpus):
    # probabilities that a state appears as initial, default to 0
    initial_probabilities = defaultdict(int)

    for sentence in corpus:
        initial_state = sentence[0][1]
        initial_probabilities[initial_state] += 1

    for state, frequency in initial_probabilities.items():
        initial_probabilities[state] = frequency / len(corpus)

    return initial_probabilities

'''
Implement a function for estimating the parameters of the matrix of transition probabilities
Parameters: corpus: list returned by the function import_corpus
Returns: data structure containing the parameters of the matrix of transition probabilities;
            use this data structure for the argument internal_representation of the function transition_probabilities
'''
def estimate_transition_probabilities(corpus):
    tag_frequencies = defaultdict(int)
    pair_tag_frequencies = defaultdict(int)

    for sentence in corpus:
        for (_, tag1), (_, tag2) in zip(sentence[:-1], sentence[1:]):
            tag_frequencies[tag1] += 1
            pair_tag_frequencies[(tag1,tag2)] += 1

        # last element of the sentence
        tag_frequencies[sentence[-1][1]] += 1

    # transition_probabilities[from_state][to_state]
    transition_probabilities = defaultdict(lambda: defaultdict(float))

    for (tag1, tag2) in pair_tag_frequencies:
        transition_probabilities[tag1][tag2] = pair_tag_frequencies[(tag1, tag2)] / tag_frequencies[tag1]

    return transition_probabilities


'''
Implement a function for estimating the parameters of the matrix of emission probabilities
Parameters: corpus: list returned by the function import_corpus
Returns: data structure containing the parameters of the matrix of emission probabilities;
            use this data structure for the argument internal_representation of the function emission_probabilities
'''
def estimate_emission_probabilities(corpus):
    tag_frequencies = defaultdict(int)
    pair_frequencies = defaultdict(int)

    for sentence in corpus:
        for observation, tag in sentence:
            tag_frequencies[tag] += 1
            pair_frequencies[(observation,tag)] += 1

    # emission_probabilities[state][observation]
    emission_probabilities = defaultdict(lambda: defaultdict(float))

    for (observation, tag) in pair_frequencies:
        emission_probabilities[tag][observation] = pair_frequencies[(observation, tag)] / tag_frequencies[tag]

    return emission_probabilities


# Exercise 2 ###################################################################

def full_log(x):
    if x == 0:
        return -inf
    else:
        return log(x)

full_log()

''''
Implement the Viterbi algorithm for computing the most likely state sequence given a sequence of observed symbols.
Parameters: observed_smbols: list of strings; the sequence of observed symbols
            initial_state_probabilities_parameters: data structure containing the parameters of the probability distribution of the initial states, returned by estimate_initial_state_probabilities
            transition_probabilities_parameters: data structure containing the parameters of the matrix of transition probabilities, returned by estimate_transition_probabilities
            emission_probabilities_parameters: data structure containing the parameters of the matrix of emission probabilities, returned by estimate_emission_probabilities
Returns: list of strings; the most likely state sequence
'''
def most_likely_state_sequence(observed_simbols,
                               initial_state_probabilities_parameters,
                               transition_probabilities_parameters,
                               emission_probabilities_parameters):
    # get tag set
    tags = emission_probabilities_parameters.keys()

    T = len(observed_simbols) - 1

    # (tag, observation_index) --> log probability
    trellis = defaultdict(lambda: -inf)

    # list to keep track of the values
    psi_list = []

    # tag that gave us the maximum value
    current_psi = ""
    # maximum logarithmic probability
    max_probability = -inf

    for tag, initial_probability in initial_state_probabilities_parameters.items():
        trellis[tag, 0] = \
            full_log(initial_probability) + \
            full_log(emission_probabilities_parameters[tag][observed_simbols[0]])

        # print(observed_simbols[0])
        # print(emission_probabilities_parameters[tag][observed_simbols[0]])

        if trellis[tag, 0] > max_probability:
            max_probability = trellis[tag, 0]
            current_psi = tag

    psi_list.append(current_psi)

    t = 1
    while t <= T:
        max_probability = -inf

        # print("-t: ", t)

        for tag_j in tags:
            delta_j = -inf
            for tag_i in tags:
                delta_j_candidate = trellis[tag_i,t-1] + \
                    full_log(transition_probabilities_parameters[tag_i][tag_j]) + \
                    full_log(emission_probabilities_parameters[tag_j][observed_simbols[t-1]])

                # consider the maximum
                if delta_j_candidate > delta_j:
                    delta_j = delta_j_candidate
                    # print("tag_j: ", tag_j)
                    current_psi = tag_j

            trellis[tag_j, t] = delta_j

        psi_list.append(current_psi)
        t += 1

    # print(trellis)
    return psi_list

def main():

    corpus = import_corpus('corpus_ner.txt')

    # split the corpus into training and test set
    random.shuffle(corpus)
    training_set = corpus[:int(len(corpus) * 0.98)]
    test_set = corpus[int(len(corpus) * 0.98):]

    # preprocess training set
    token_frequencies = Counter()
    for sentence in training_set:
        for token, _ in sentence:
            token_frequencies[token] += 1

    preprocess(training_set, token_frequencies)

    # print(training_set)
    # print(test_set)

    # train
    initial_state_probabilities = estimate_initial_state_probabilities(training_set)
    transition_probabilities = estimate_transition_probabilities(training_set)
    emission_probabilities = estimate_emission_probabilities(training_set)

    # ignore tag information
    test_sentences = \
        list(map(lambda sentence: list(map(lambda pair: pair[0], sentence)), test_set))

    test_tag_sequences = \
        list(map(lambda sentence: list(map(lambda pair: pair[1], sentence)), test_set))

    # preprocess test sentences
    for sentence in test_sentences:
        for i, token in enumerate(sentence):
            # recognize unknown tokens
            if token_frequencies[token] == 0:
                sentence[i] = UNKNOWN_T

    total_tokens = 0
    hits = 0

    for test_sentence, test_tag_sequence in zip(test_sentences, test_tag_sequences):
        prediction = most_likely_state_sequence(test_sentence,
                                                    initial_state_probabilities,
                                                    transition_probabilities,
                                                    emission_probabilities)

        # print("----")
        # print(prediction)
        # print(test_tag_sequence)

        for predicted_tag, actual_tag in zip(prediction, test_tag_sequence):
            total_tokens += 1
            if predicted_tag == actual_tag:
                hits += 1

    print("Accuracy: ", hits/total_tokens)

main()
