import json
import pandas as pd
import os.path as osp
from typing import List, Tuple
from transformers import BertTokenizer
bert_name = './deps/bert-base-chinese'
class CustomBertTokenizer():
    def __init__(self, vocab_size=2550):    # prior on the csv file
        super().__init__()
        self.pad_token, self.eos_token, self.sep_token = range(vocab_size, vocab_size + 3)
        self.tokenizer = BertTokenizer.from_pretrained(bert_name)
        
    def build_vocab(self, text_list: List[str], save_merge=None):
        self.vocab_map = {}
        self.inverse_vocab_map = {}

        word_cnt = 0
        for data in text_list:    
            encoded = self.tokenizer.encode(data)[1:-1]
            for idx in encoded:
                if str(idx) not in self.vocab_map:
                    self.vocab_map[str(idx)] = word_cnt
                    self.inverse_vocab_map[str(word_cnt)] = idx
                    word_cnt += 1
                    
        if save_merge:
            save_path = osp.join(save_merge, f'bert_map_{word_cnt}.json')
            with open(save_path, 'w') as f:
                to_save = {
                    "map": self.vocab_map,
                    "inverse_map": self.inverse_vocab_map
                }
                json.dump(to_save, f, indent=2)
            print(f"Save merge rules to {save_path}.")
    
    def from_file(self, file_path):
        with open(file_path, 'r') as f:
            map_rules = json.load(f)
        self.vocab_map = map_rules["map"]
        self.inverse_vocab_map = map_rules["inverse_map"]

    def encode(self, text):
        tokens = self.tokenizer.encode(text)[1:-1]
        map_tokens = [self.vocab_map[str(token)] for token in tokens]
        return map_tokens

    def decode(self, idxs):
        inverse_map_tokens = [self.inverse_vocab_map[str(idx)] for idx in idxs]
        return self.tokenizer.decode(inverse_map_tokens)

if __name__ == "__main__":
    # Code for test BBPE
    df = pd.read_csv('./deps/reviews.csv')
    data_list = df.iloc[:, 0].to_list()
    bert_tokenizer = CustomBertTokenizer()
    # bert_tokenizer.build_vocab(data_list, save_merge='./deps')
    bert_tokenizer.from_file('./deps/bert_map_2550.json')
    encoded = bert_tokenizer.encode("你好，你能编码这句话吗")
    print(encoded)
    print(bert_tokenizer.decode(encoded))
        