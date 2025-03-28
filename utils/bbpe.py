import json
import regex as re
import pandas as pd
import os.path as osp
from tqdm import tqdm
from typing import List, Tuple

from functools import reduce
class BBPE():
    def __init__(self, vocab_size):
        self.max_vocab_size = vocab_size
        self.now_vocab_size = 256
        self.merge_rules = {}
        self.vocab = [bytes([i]) for i in range(256)]
        # From transformers/src/transformers/models/gpt2/tokenization_gpt2.py.
        # I think this is how to prevent BPE from merging across characters and add an exception for space.
        self.tokenizer_pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.pad_token, self.eos_token, self.sep_token = range(vocab_size, vocab_size + 3)
    
    def from_file(self, file_path):
        with open(file_path, 'r') as f:
            merge_rules = json.load(f)
        self.merge_rules = {eval(k):v for k, v in merge_rules.items()}
        self.max_vocab_size = len(self.merge_rules) + 256
        self.now_vocab_size = self.max_vocab_size
        for pair in self.merge_rules:
            self.vocab.append(self.vocab[pair[0]] + self.vocab[pair[1]])
        
    def get_corpus_length(self, tokens: List[List[int]]):
        """
            Count how many bytes are ther in the corpus.
        """
        return sum([len(token) for token in tokens])
    
    def count_most_freq(self, tokens: List[List[int]]):
        """
            Given utf8 token list, give back the most frequent pair and corresponding count.
        """
        freq = {}
        for token in tokens:
            for pair in zip(token[:-1], token[1:]):
                if pair in freq:
                    freq[pair] += 1
                else:
                    freq[pair] = 1
        
        return max(freq.items(), key=lambda item: item[1])
        
    def merge(self, tokens: List[List[int]], pair: Tuple[int, int], idx):
        """
            Replace all pairs in tokens with idx.
        """
        new_tokens = []
        for token in tokens:
            new_token = []
            i = 0 
            while i < len(token):
                if i != len(token) - 1 and (token[i], token[i + 1]) == pair:
                    new_token.append(idx)
                    i += 2
                else:
                    new_token.append(token[i])
                    i += 1
            new_tokens.append(new_token)
        return new_tokens
                    
    def build_vocab(self, text_list: List[str], save_merge=None):
        """
            Building vocabulary and merge rules from corpus.
        """
        tokens = reduce(lambda x, y:x+y, [re.findall(self.tokenizer_pat, text) for text in text_list])
        tokens = [list(token.encode('utf-8')) for token in tokens]
        
        origin_corpus_length = self.get_corpus_length(tokens)
        building_loop = tqdm(range(self.now_vocab_size, self.max_vocab_size))
        for idx in building_loop:
            pair, cnt = self.count_most_freq(tokens)
            self.merge_rules[pair] = idx
            tokens = self.merge(tokens, pair, idx)
            self.vocab.append(self.vocab[pair[0]] + self.vocab[pair[1]])
            self.now_vocab_size += 1
            
            now_corpus_length = self.get_corpus_length(tokens)
            desp = f'Building vocab {idx}... Compressing {now_corpus_length}/{origin_corpus_length}, {now_corpus_length/origin_corpus_length * 100:.2f}%, Most frequent: {cnt}'
            building_loop.set_description(desp)
            
        print("Done!")
        if save_merge:
            save_path = osp.join(save_merge, f'merge_rule_{self.max_vocab_size}.json')
            with open(save_path, 'w') as f:
                to_save = {str(k):v for k, v in self.merge_rules.items()}
                json.dump(to_save, f, indent=2)
            print(f"Save merge rules to {save_path}.")
    
    def get_merge_rank(self, pair):
        if pair not in self.merge_rules:
            return float('inf')
        return self.merge_rules[pair]
    
    def encode(self, text):
        tokens = re.findall(self.tokenizer_pat, text)
        tokens = [list(token.encode('utf-8')) for token in tokens]
        new_tokens = []
        for token in tokens:
            while True:
                if len(token) <= 1:
                    break
                pairs = zip(token[:-1], token[1:])
                to_merge = min(pairs, key=lambda pair:self.get_merge_rank(pair))
                if to_merge not in self.merge_rules:
                    break
                else:
                    token = self.merge([token], to_merge, self.merge_rules[to_merge])[0]
            new_tokens += token
        return new_tokens

    def decode(self, idxs):
        if len(idxs) == 0:
            return ""
        return reduce(lambda x, y:x+y, [self.vocab[idx] for idx in idxs]).decode('utf8', errors='ignore')
               
if __name__ == "__main__":
    # Code for test BBPE
    df = pd.read_csv('./deps/reviews.csv')
    data_list = df.iloc[:, 0].to_list()
    bbpe = BBPE(1024)
    bbpe.from_file('./deps/merge_rule.json')
    encoded = bbpe.encode("你好！请问你能编码这句话吗？这是在测试你的tokenize功能。")
    print(encoded)
    print(bbpe.decode(encoded))
