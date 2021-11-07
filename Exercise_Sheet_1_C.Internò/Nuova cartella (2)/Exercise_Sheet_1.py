import numpy as np
import random, collections, bisect
from collections import defaultdict
import random, collections, bisect
from itertools import islice, chain
import seaborn as sns

CORPUS_FILENAME = "./corpus.txt"
# Definition of a function for load the corpus file
# Whit this we can divide the text in sentence

def load(file : str = CORPUS_FILENAME):
    with open(file, "rt") as corpus:
        for l in corpus:
            l = l.rstrip()
            if l != "":
                yield l

#'Processing_file' returns a sequence of w2,delete specific chars
def customize_file(text, no_chars: str = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~\t\r\n"):
   
    #remove the whitespace,split text on spaces and replace the characters we dont want with empty spaces
    text = text.strip()
    toks = text.split(" ")
    text = "".join([character if character not in no_chars else " " for character in text])
    #delete empty toks 
    toks = list(filter(None, toks))
    # "Introduce special w2 to model the beginning and the end of a sentence!"
    # [START]:Start of sentence
    # [END]:End of sentence
    toks = ["[START]"] + toks + ["[END]"]
    return toks

#-------------------------------------------------------------------------------------------------------------------------------------
#list of all the word
def lst(fileName):
    data = [] #Dict for the word
    for s_ind, s_text in enumerate(load(fileName)):
        for tok in customize_file(s_text, "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~\t\r\n"):
            data.append(tok)
    return data

# ------------------------------Unigram P(w) ----------------------------------------------------------

#Frequency for each unique word
#probabilities: each term's (frequency) / n. total toks
def unigram (file):
    freq_tokens = defaultdict(int) 
    for sentence in load(file):
        for tok in customize_file(sentence, "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~\t\r\n"):
            if tok=="[START]" or tok=="[END]":
                tok=''
            else:
                freq_tokens[tok] += 1
    total_tok_count = sum(freq_tokens.values())
    tok_probabilities = {} #dict of the probabilities
    for tok, count in freq_tokens.items():
        tok_probabilities[tok] = count / total_tok_count
        
    return tok_probabilities, freq_tokens


# ------------------------------Bigram P(wi|wi−1)------------------------------------------------------------

def bigram_model(w2):
    
    tuples = zip(*[w2[i:] for i in range(2)])#create the tuples for the bigrams
    dic_bigrams = {} #Dict for the frequency of the bigrams
    for bigram in tuples:
        if bigram not in dic_bigrams:
            dic_bigrams[bigram] = 0
        dic_bigrams[bigram] += 1
    
    count_uni = {}
    
    for i in range(len(w2)):
        if w2[i] in count_uni:
            count_uni[w2[i]] += 1
        else:
            count_uni[w2[i]] = 1
    
    return dic_bigrams, list(tuples), count_uni

def bigram_probability(count_uni, bigramCounts):
    #Probability of the bigram: bigram frequency/frequency of the previous two w2
    Prob_list_bigrams = {}
    for bigram in bigramCounts.keys():
        w1 = bigram[0]
        w2 = bigram[1]
        Prob_list_bigrams[bigram] = (bigramCounts[bigram]/(count_uni.get(w1)))
    return Prob_list_bigrams

# ------------------------------Trigram P(wi|wi−1, wi−2) --------------------------------

def trigram_generate(data):
    trigrams_tuple = zip(*[data[i:] for i in range(3)]) #tuples for trigrams
    trigrams_count = {} #Dict for trigrams
    
    for trigram in trigrams_tuple:
        if trigram not in trigrams_count:
            trigrams_count[trigram] = 0
        trigrams_count[trigram] += 1
    
    return trigrams_count, list(trigrams_tuple)

def trigram_prob(trigramCounts, bigramCounts):
    
    #Probability of the trigram: frequency of the trigram /frequency of the previous two w2.
    listOfProb = {} #dict for the probability
    for trigram in trigramCounts.keys():
        w1 = trigram[0]
        w2 = trigram[1]
        listOfProb[trigram] = trigramCounts[trigram]/(bigramCounts[(w1,w2)])

    return listOfProb
#-------------------------------------------------------------------------------------------------------------------------------------
#For the Exercise 2, I use this function for bigram and trigram
def func_for_ese2(w2, a):
    map = {}
    a.extend(islice(w2, 0, len(a)))
    for word in w2:
        entry = map.setdefault(tuple(a), {})
        entry[word] = entry[word] + 1 if word in entry else 1
        a.append(word)
    for prefix in map.keys():
        total = float(sum(map[prefix].values()))
        cum_prob = 0.
        suffixes = []
        for w, c in map[prefix].items():
            cum_prob += c / total
            suffixes.append((cum_prob, w))
        map[prefix] = tuple(suffixes)
    return map

# ------------------------------ Drawing Samples of Unigram --------------------------------
#2)a
def drawing_samples_unigram(prob_values: dict, size: int):
    
    sample = np.random.choice(len(list(prob_values.values())), size, p=list(prob_values.values()))

    return sample
#-------------------------------------------------------------------------------------------
class Vocabulary:
    
    def __init__(self, arity=1):
        self._frozen = False
        self.id2token = {}
        self.token2id = {}
        self._arity = arity
        self._token_delimiter = "_"
    
    @property
    def token_delimiter(self):
        return self._token_delimiter
    
    @token_delimiter.setter
    def set_token_delimiter(self, value):
        self._token_delimiter = value
    
    @property
    def arity(self):
        return self._arity
    
    @arity.setter
    def set_arity(self, value):
        self._arity = value
        
    @property
    def frozen(self):
        return self._frozen

    @frozen.setter
    def set_frozen(self, value):
        self._frozen = value
        
    def size(self):
        return len(self.token2id)
        
    def __repr__(self):
        return {self.size()}
    
    def _split_token(self, token):
        return token.split(self._token_delimiter, self.arity)
    
    def _convert_token(self, token):
        if isinstance(token, list):
            if len(token) != self.arity:
                raise Exception(+ str(self.arity)                                  
                                + str(len(token)))
            token = self._token_delimiter.join(token)
        return token

    def add(self, token):
        if self.frozen:
            raise Exception()
            
        token = self._convert_token(token)
        
        if token in self.token2id:
            return self.token2id[token]
        
        newid = len(self.token2id)
        self.token2id[token] = newid
        self.id2token[newid] = token
        return newid

    def get(self, token_or_index):
        if isinstance(token_or_index, (int, np.integer)):
            index = token_or_index
            if index in self.id2token:
                return self._split_token(self.id2token[index])
            return None
        token = self._convert_token(token_or_index)
        if token in self.token2id:
            return self.token2id[token]
        return None

#-------------------------------------------------------------------------------------------------------------------------------------
#helper function that generates n-gram sequences:
def ngrams_from_corpus(n: int):
    if n < 1:
        raise Exception()
    
    for sentence_index, sentence_text in enumerate(load()):
        sentence_tokens = customize_file(sentence_text)
        
        for current_index in range(n-1, len(sentence_tokens)):
            yield sentence_tokens[current_index-n+1:current_index+1]
            
for seq_id, tokens in enumerate(ngrams_from_corpus(n=2)):
    if seq_id > 18:
        break
        
#-------------------------------------------------------------------------------------------------------------------------------------
counts = {}

for i in range(10, 20):
    elem = i // 2
    if elem not in counts:
        counts[elem] = 0
    counts[elem] += 1

#-------------------------------------------------------------------------------------------------------------------------------------
from collections import defaultdict

counts = defaultdict(int)

for i in range(10, 20):
    elem = i // 2
    counts[elem] += 1
#-------------------------------------------------------------------------------------------------------------------------------------
vocab = Vocabulary(arity=1)
ngram_counts = defaultdict(int)

for ngram in ngrams_from_corpus(n=1):
    if ngram[0] == "[AND]":
        continue
    ngram_counts[vocab.add(ngram)] += 1
    
#-------------------------------------------------------------------------------------------------------------------------------------
[(i, vocab.get(i), ngram_counts[i]) for i in range(vocab.size())][:10]
#-------------------------------------------------------------------------------------------------------------------------------------
total_ngram_count = sum(ngram_counts.values())
#-------------------------------------------------------------------------------------------------------------------------------------
unigram_probabilities = np.array([ngram_counts[i] / total_ngram_count for i in range(vocab.size())],
                                dtype=np.double)
#-------------------------------------------------------------------------------------------------------------------------------------

def sample_unigram_sentence(unigram_vocab, unigram_probabilities, max_length=20):
    sentence_sequence = []
    
    while len(sentence_sequence) < max_length and "[START]" not in sentence_sequence:
        nexttoken = drawing_samples_unigram(unigram_vocab, 20)
        nexttoken = vocab.get(nexttoken[0])[0]
        sentence_sequence.append(nexttoken)
                
    return sentence_sequence
#-------------------------------------------------------------------------------------------------------------------------------------
def generate_ngram_sentence(ngrams, tup):
    
    #This function generates consecutive words that were in the provided ngrams index using for choosing the suffixes the frequency information in the index.
    
    while tuple(tup) in ngrams:       
                chosen = ngrams[tuple(tup)]
                _, word = chosen[bisect.bisect_left(chosen, (random.random(), ''))]
                tup.append(word)
                yield word
#-------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------MAIN---------------------------------------------------------------------------------
if __name__ == "__main__":
    fileName = './corpus.txt'
    data = lst(fileName)
    print(f'1.a{"_"*60}')
    print(f'e.g. of a sentence:')
    for s_ind, s_text in enumerate(load(fileName)):
        if s_ind < 1:
            print( customize_file(s_text, "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~\t\r\n"))
#----------------------------------------------------------------------------------------------------------------
    #1)B Unigram 
    tok_probabilities, freq_tokens=unigram(fileName)
    #show 20
    print(f'1.b.1{"_"*60}')
    print(f'Unigrams with the highest probabilities:')
    for k, v in sorted(tok_probabilities.items(), key=lambda item: item[1], reverse=True)[:20]:
        print(str(k) + ' = '+ str(v))
        
    #1)B Bigram 
    bigramCounts, Prob_list_bigrams, count_uni = bigram_model(data)
    bigramProb = bigram_probability(count_uni, bigramCounts)
    
    print(f'1.b.2{"_"*60}')
    print(f'Bigrams with the highest probabilities:')
    for i,bigrams in enumerate(bigramCounts.keys()):
        if i<20:
            print(str(bigrams) + ' = ' + str(bigramProb[bigrams]))

    #1)B Trigram 
    trigramCounts, listOfTrigrams=trigram_generate(data)
    trigramProb = trigram_prob(trigramCounts, bigramCounts)
    
    print(f'1.b.3{"_"*60}')
    print(f'Trigrams with the highest probabilities:')    
    for i,trigrams in enumerate(trigramCounts.keys()):
        if i<20:
            print(str(trigrams) + ' = ' + str(trigramProb[trigrams])) 
            
    #1.c How does the number of parameters of these distributions scale with the number of different w2 in the corpus? 
    print(f'1.c{"_"*60}')
    print(f"The number of parameters increases exponentially:e.g. for the unigram is equal to n = {len(count_uni.keys())}; For the bigrams, we have to store every tuple of two w2(a,b)")
    print(f"freq(a,a),freq(a,b),freq(b,a),freq(b,b), so X numbers of parameters is equal to n^2 = {len(count_uni.keys())*len(count_uni.keys())} and for the trigram model it is equalt to n^3 =")    
    print(f"{len(count_uni.keys())*len(count_uni.keys())*len(count_uni.keys())}")
    
    #2.a and 2.b Python functions for drawing samples from the distributions P(w), P(wi|wi−1) and P(wi|wi−1, wi−2) and generating words using the probability distirbutions.
    
    #Unigram P(w)
    print(f'2.a {"_"*60}')
    #print(f'A random sample with index of the words generated by the unigram distribution is:')
    #print(drawing_samples_unigram(tok_probabilities,20))
    
    print(f'{"_"*60}\nTSentence generated from the sampling of unigram distribution:')
    print(" ".join(sample_unigram_sentence(tok_probabilities, 20, 20)))
    
    #Bigram P(wi|wi−1)
    print(f'{"_"*60}\nSentence generated from the sampling of bigram distribution:')
    n = 2
    input = data
    bi = func_for_ese2(input, collections.deque(maxlen= n-1))
    start = collections.deque(random.choice(list(bi.keys())), n - 1)
    seq=[]
    for word in start:
        seq.append(word)
    for token in generate_ngram_sentence(bi, start):
        seq.append(token)
        if len(seq)>60:
            break
    print(' '.join(seq))
    
    #Trigram P(wi|wi−1, wi−2) a
    print(f'2.b {"_"*60}\nSentence generated from the sampling of trigram distribution:')
    n = 3
    input = data
    tri = func_for_ese2(input, collections.deque(maxlen= n-1))
    start = collections.deque(random.choice(list(tri.keys())), n - 1)
    seq=[]
    for word in start:
        seq.append(word)
    for token in generate_ngram_sentence(tri, start):
        seq.append(token)
        if len(seq)>60:
            break
    print(' '.join(seq))
    
    #2.c 
    print(f'2.c{"_"*60}')
    print('The sentence generated using Unigram makes no logical sense, as there is no conditional probability, but a simple random sampling.Using Bigrams and Trigrams you can see how the sentence start to makes more  sense. The generated text improves as n increases, I suppose as the size of the N-gram model increases, the model starts copying the original text.')