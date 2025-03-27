import json
import torch
import torch.nn as nn
import pandas as pd
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
from copy import deepcopy

class GPT2Dataset(torch.utils.data.Dataset):
    def __init__(self, config, tokenizer, build_tokenizer, device, mode='train'):
        # config is config.DATA for global config
        super().__init__()
        self.config = config
        
        # loading csv reviews
        df = pd.read_csv(config.data_root)
        raw_data = df.iloc[:, 0].to_list()
        raw_rating = df.iloc[:, 1].to_list()

        # construct tokenizer
        self.tokenizer = tokenizer
        if build_tokenizer:
            if config.from_file:
                self.tokenizer.from_file(config.from_file)
            else:
                if config.save_merge:
                    self.tokenizer.build_vocab(raw_data, config.save_merge)
                else:
                    self.tokenizer.build_vocab(raw_data)
                
        # Clean data if there is a command too short which is not good for learning
        self.data_list =  []
        self.rating_list = []
        for i, data in enumerate(raw_data):
            if len(data.encode('utf-8')) > 3:
                self.data_list.append(data)
                self.rating_list.append(raw_rating[i]) 
        
        print(f"Data cleaning done.\nRemain {len(self.data_list)} piece of commands.")
        
        train_length = int(len(self.data_list) * (1 - config.val_ratio))
        if mode == 'train':
            self.data_list = self.data_list[:train_length]
        elif mode == 'val':
            self.data_list = self.data_list[train_length:]
        
        # prefix for prompt
        self.pos_prefix = self.tokenizer.encode("好看")
        self.neg_prefix = self.tokenizer.encode("不好看")
        
        self.curr_iter = 0
        self.device = device
        
    def __len__(self):
        return len(self.data_list)
    
    def __next__(self):
        if self.curr_iter >= len(self):
            self.curr_iter = 0
            raise StopIteration()
        else:
            single_data = self.__getitem__(self.curr_iter)
            self.curr_iter += 1

        return single_data

    def __getitem__(self, idx):
        # Add encoded "好看" or "不好看" as prefix, separated by sep_token as prompt
        prefix = deepcopy(self.neg_prefix) if self.rating_list[idx] == 0 else deepcopy(self.pos_prefix)
        prefix.append(self.tokenizer.sep_token)
        sentence_encoded = self.tokenizer.encode(self.data_list[idx])
        single_data = prefix + sentence_encoded
        target = prefix + sentence_encoded[1:] + [self.tokenizer.eos_token]
        mask = [0] * len(prefix) + [1] * len(sentence_encoded)
        # input: encoded(好看) + [sep] + [tok1, tok2 ...tokn-1, tokn] + [pad...]
        # target: encoded(好看) + [sep] + [tok2, tok3 ...tokn, eos]  + [pad...]
        # mask: [0 for prefix/sep] + [1...] + [0 for pad]
        
        # padding
        if len(single_data) < self.config.max_tokens:
            to_pad = (self.config.max_tokens - len(single_data))
            single_data += [self.tokenizer.pad_token] * to_pad
            target += [self.tokenizer.pad_token] * to_pad
            mask += [0] * to_pad
        else:
            single_data = single_data[:self.config.max_tokens]
            target = target[:self.config.max_tokens]
            mask = target[:self.config.max_tokens]
        
        # assertions for collating
        assert len(single_data) == len(target)
        assert len(single_data) == len(mask)
        
        return {
            "input": torch.as_tensor(single_data).long().to(self.device),
            "target": torch.as_tensor(target).long().to(self.device),
            "mask": torch.as_tensor(mask).long().to(self.device)
        }
        