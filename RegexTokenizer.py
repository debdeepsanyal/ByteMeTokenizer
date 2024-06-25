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
        
        chunk_texts = re.findall(self.compiled_pattern, self.text)
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
        
    
    def _encode_chunk(self, text_byte):
        raw_tokens = list(text_byte)
        while len(raw_tokens) >= 2:
            freq = Counter(zip(raw_tokens, raw_tokens[1:]))
            eligible_pair = min(freq, key = lambda p : self.merges.get(p, float('inf')))
            if eligible_pair not in self.merges:
                break #nothing more to merge 
                
            merging_idx = self.merges[eligible_pair]
            raw_tokens = merge(raw_tokens, eligible_pair, merging_idx)
        
        return raw_tokens
    
    def encode(self, text):
        text_chunks = re.findall(self.compiled_pattern, text)
        ret = []
        for chunk in text_chunks:
            chunk_byte = chunk.encode('utf-8')
            encoded_chunk = self._encode_chunk(chunk_byte)
            ret.extend(encoded_chunk)
        
        return ret 
    
    def decode(self, tokens):
        part_bytes = []
        for idx in tokens:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            else:
                raise ValueError(f"The token {idx} is invalid.")
        
        return b''.join(part_bytes).decode('utf-8', errors = 'replace')
    

reg = RegexTokenizer()
reg.train('/teamspace/studios/this_studio/Tokenization/TaylorSwiftWiki.txt', 1256, True)
text = 'Another attempt at a really random text !!!></ 😂'
print(reg.decode(reg.encode(text)) == text)
    



        
    
        
    



        
        

