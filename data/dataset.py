import torch
import torch.nn as nn
import pandas as pd
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader

class GPT2Dataset(torch.utils.data.Dataset):
    def __init__(self, config, bbpe, device):
        # config is config.DATA for global config
        super().__init__()
        df = pd.read_csv(config.data_root)
        self.data_list = df.iloc[:, 0].to_list()
        self.rating_list = df.iloc[:, 1].to_list()
        self.curr_iter = 0
        self.bbpe = bbpe
        if config.from_file:
            self.bbpe.from_file(config.from_file)
        else:
            if config.save_merge:
                self.bbpe.build_vocab(self.data_list, config.save_merge)
            else:
                self.bbpe.build_vocab(self.data_list)
        self.pos_prefix = self.bbpe.encode("好看")
        self.neg_prefix = self.bbpe.encode("不好看")
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
        prefix = self.neg_prefix if self.rating_list[idx] == 0 else self.pos_prefix
        prefix.append(self.bbpe.sep_token)
        sentence_encoded = self.bbpe.encode(self.data_list[idx])
        single_data = prefix + sentence_encoded
        target = prefix + sentence_encoded[1:] + [self.bbpe.eos_token]
        if len(single_data) < self.config.max_tokens:
            to_pad = (self.config.max_tokens - len(single_data))
            single_data += [self.bbpe.pad_token] * to_pad
            target += [self.bbpe.pad_token] * to_pad
        else:
            single_data = single_data[:self.config.max_tokens]
            target += target[:self.config.max_tokens]
        mask = [0] * len(prefix) + [1] * (self.config.max_tokens - len(prefix))
        assert len(single_data) == len(target)
        assert len(single_data) == len(mask)
        return {
            "input": torch.as_tensor(single_data).long().to(self.device),
            "target": torch.as_tensor(target).long().to(self.device),
            "mask": torch.as_tensor(mask).long().to(self.device)
        }
        