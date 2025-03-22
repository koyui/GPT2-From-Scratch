import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

import os.path as osp

from torch.utils.data.dataloader import DataLoader
from tqdm import trange, tqdm
from omegaconf import OmegaConf

from models.model import GPT2
from data.utils import collate_fn
from data.dataset import GPT2Dataset
from utils.bbpe import BBPE

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

class Trainer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = GPT2(config.MODEL).to(device)
        self.bbpe = BBPE(config.MODEL.vocab_size)
        self.dataset = GPT2Dataset(config.DATA, self.bbpe, device)
        self.dataloader = DataLoader(
            self.dataset,
            config.DATA.batch_size if config.MODEL.phase == 'train' else 1,
            shuffle=True, 
            drop_last=False, 
            collate_fn=collate_fn
        )
        if self.config.MODEL.phase == 'train':
             if self.config.TRAIN.from_pretrained:
                self.load_model(self.config.TRAIN.from_pretrained)
        elif self.config.MODEL.phase == 'test':
            self.model.eval()
            
        
    def load_model(self, model_path):
        milestone = torch.load(model_path, map_location=device)
        self.model.load_state_dict(milestone)
            
    def save_model(self, epoch):
        time_stamp = time.strftime("%Y%m%d_%H%M%S")
        save_dir = self.config.TRAIN.model_save + '_' + time_stamp
        os.makedirs(save_dir, exist_ok=True)
        
        save_path = osp.join(save_dir, f'{epoch}.pth')
        torch.save(self.model.state_dict(), save_path)
        
    def generate(self, x):  
        while x.shape[1] < self.config.DATA.max_tokens:
            with torch.no_grad():
                logits = self.model(x)  # (B, T, vocab_size)
                logits = logits[:, -1, :]   # (B, vocab_size)
                probs = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, self.config.MODEL.topk, dim=-1)  # (B, topk), (B, topk)
                idxs = torch.multinomial(topk_probs, 1)    # (B, 1)
                token_id = torch.gather(topk_indices, -1, idxs)
                x = torch.cat([x, token_id], dim=-1)
        return x
    
    def train(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.TRAIN.lr)
        training_loop = trange(self.TRAIN.max_training_epochs)
        for epoch in training_loop:
            loss_list = []
            for batch in tqdm(self.dataloader):
                optimizer.zero_grad()
                _, loss = self.model(batch["input"], batch["target"], batch["mask"])
                loss_list.append(loss.item())
                optimizer.step()
                loss.backward()
            training_loop.set_description(f'Loss per batch: {sum(loss_list)/len(loss_list)}')
            if epoch % self.config.TRAIN.save_per == 0:
                self.save_model(epoch)
                
if __name__ == "__main__":
    config = OmegaConf.load('./configs/config.yaml')
    trainer = Trainer(config)
    