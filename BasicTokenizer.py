from copy import deepcopy
from collections import Counter
from typing import List
import sys
from InvalidVocabSizeException import InvalidVocabSizeException
class BasicTokenizer:
    def train(self, text : str, vocab_size : int, isFile : bool = False, verbose : bool = False):
        if not isFile:
            self.text = text
        
        else :
            self.text = self.toText(text)
        
        raw_tokens = list(self.text.encode('utf-8'))

        try:
            if vocab_size < 256:
                raise InvalidVocabSizeException
            else:
                self.vocab_size = vocab_size
        
        except InvalidVocabSizeException:
            print("Vocab Size should be greater than 256.")
            sys.exit()

        
        self.merges = {}

        num_merges = self.vocab_size - 256
        replace_tokens = deepcopy(raw_tokens)
        for i in range(num_merges):
            swap_idx = 256 + i
            freq = Counter(list(zip(replace_tokens, replace_tokens[1:])))
            tup = max(freq, key = freq.get)
            replace_tokens = self.merge(replace_tokens, tup, swap_idx)
            self.merges[tup] = swap_idx
        
        if verbose:
            print(f'The Compression ratio is {len(raw_tokens) / len(replace_tokens) : .2f}x')
        
        return replace_tokens
    
    def toText(self, link : str):
        return open(link).read()
    
    def merge(self, fill : List[int], tup : tuple, swap_idx : int):
        i = 0
        ret = []
        while i < len(fill):
            if i < len(fill) -1 and fill[i] == tup[0] and fill[i+1] == tup[1]:
                ret.append(swap_idx)
                i+=2
            else:
                ret.append(fill[i])
                i += 1
        
        return ret
    
    def create_vocab(self):
        self.vocab = {idx : bytes([idx]) for idx in range(256)}
        for (p0, p1), i in self.merges.items():
            self.vocab[i] = self.vocab[p0] + self.vocab[p1]
    
    def encode(self, text : str):
        tokens = list(text.encode('utf-8'))

        while True:
            stats = list(Counter(zip(tokens, tokens[1:])))
            tup = min(stats, key = lambda p : self.merges.get(p, float('inf')))
            if tup not in self.merges:
                break #nothing more to merge

            tokens = self.merge(tokens, tup, self.merges[tup])
        
        return tokens
    
    def decode(self, tokens : List[int]):
        self.create_vocab()
        ids = b''.join(self.vocab[idx] for idx in tokens)
        return ids.decode('utf-8', errors='replace')

        


    


        
tokenizer = BasicTokenizer()
path = '/Users/debdeepsanyal/Downloads/TaylorSwiftWiki.txt' #change this to source 
tokens = tokenizer.train(path, 2256, True)
test = 'this is just like the most random text. a jocust rendezvous of words tangled in the strings of my imaginations!'
print(tokenizer.decode(tokenizer.encode(test)))
