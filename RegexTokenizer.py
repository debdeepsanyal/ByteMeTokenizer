import regex as re 
from BasicTokenizer import BasicTokenizer
from copy import deepcopy
from collections import Counter

instance = BasicTokenizer()
merge = instance.merge
toText = instance.toText

#this has been taken from the tiktoken library
#find the same pattern at https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py in cl100k_base
gpt4_pattern = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer():
    def __init__(self, pattern = None):
        if pattern == None:
            self.pattern = gpt4_pattern
        else:
            self.pattern = pattern
        
        self.compiled_pattern = re.compile(self.pattern)
    
    def train(self, text : str, vocab_size : int, isFile : bool = False, verbose : bool = False):
        assert vocab_size >= 256
        if isFile:
            self.text = toText(text)
        else:
            self.text = text
        
        chunk_texts = re.findall(self.pattern, self.text)
        self.merges = {}
        self.vocab = {idx : bytes([idx]) for idx in range(256)}
        num_merges = vocab_size - 256

        raw_tokens = [list(chunk.encode('utf-8')) for chunk in chunk_texts] #this is a matrix 
        new_tokens = deepcopy(raw_tokens)
        for i in range(num_merges):
            swap_idx = 256 + i
            freq = dict()
            for token in new_tokens:
                a = Counter(zip(token, token[1:]))
                freq.update(a)
            
            tup = max(freq, key = freq.get)
            new_tokens = merge(new_tokens, tup, swap_idx)
            self.vocab[swap_idx] = self.vocab[tup[0]] + self.vocab[tup[1]]
            self.merges[tup] = swap_idx
        
        if verbose:
            print(f'Compression ratio {len(raw_tokens) / len(new_tokens) : .2f}')
        
    



        
        

