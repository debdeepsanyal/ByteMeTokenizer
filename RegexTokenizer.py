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
        self.special_tokens = dict()
        self.inverse_special_tokens = dict()
    
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
            new_tokens = [merge(tokens, tup, swap_idx) for tokens in new_tokens]
            self.vocab[swap_idx] = self.vocab[tup[0]] + self.vocab[tup[1]]
            self.merges[tup] = swap_idx
        
        if verbose:
            print(f'Compression ratio {len(raw_tokens) / len(new_tokens) : .2f}')
        
    def register_special_tokens(self, special_tokens):
        """special tokens is a dictionary containing the special tokens and the respective ids.
        Example token - {<|endoftext|> : 1257} """
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v : k for k, v in self.special_tokens.items()}


        
    
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
    
    def special_encode(self, text, allowed_special = 'none_raise'):
        """ allowed_special can be of 3 types - 
        1> `all` - special tokens are present 
        2> `none` - no special tokens
        3> `none_raise` - raise an error if any of the special tokens are in the provided text """

        specials = dict()
        if allowed_special == 'all':
            assert self.special_tokens
            specials = self.special_tokens
        elif allowed_special == 'none':
            specials = dict()
        elif allowed_special == 'none_raise':
            specials = dict()
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            specials = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        
        if not specials:
            return self.encode(text)
        
        pattern = '(' + '|'.join(re.escape(k) for k in specials) + ')'
        splits = re.split(pattern, text)
        ids = []
        for split in splits:
            if split in specials:
                ids.append(specials[split])
            
            else:
                ids.extend(self.encode(split))
        
        return ids
        
            

        
        
    

reg = RegexTokenizer()
reg.train('/teamspace/studios/this_studio/Tokenization/TaylorSwiftWiki.txt', 1256, True)
text = 'this is just like the most random text fr ??//😂. with some <|padding|> for no reason. okay bye <|endoftext|>'
reg.register_special_tokens({'<|padding|>' : 10000, '<|endoftext|>' : 10001})
print(reg.special_encode(text, 'all'))

    



        
    
        
    



        
        

