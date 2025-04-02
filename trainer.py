import os
import time
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F

import os.path as osp

from tqdm import trange, tqdm
from omegaconf import OmegaConf
from natsort import natsorted
from bert_score import BERTScorer
from torch.utils.data.dataloader import DataLoader

from models.model import GPT2
from data.utils import collate_fn
from data.dataset import GPT2Dataset
from utils.bbpe import BBPE
from utils.custom_bert_tokenizer import CustomBertTokenizer

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

class Trainer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if self.config.DATA.tokenizer == 'bbpe':
            self.tokenizer = BBPE(config.MODEL.vocab_size)
        elif self.config.DATA.tokenizer == 'bert':
            self.tokenizer = CustomBertTokenizer(config.MODEL.vocab_size)
            
        # tokenizer load or build in the dataset initialization
        # Must be here, or tokenzier will not be correctly constructed.
        self.train_dataset = GPT2Dataset(
            config.DATA, 
            self.tokenizer, 
            build_tokenizer=True, 
            device=device, 
            mode='train'
        )
        self.val_dataset = GPT2Dataset(
            config.DATA, 
            self.tokenizer, 
            build_tokenizer=False,
            device=device, 
            mode='val'
        )
        self.train_dataloader = DataLoader(
            self.train_dataset,
            config.DATA.batch_size,
            shuffle=False, 
            drop_last=False, 
            collate_fn=collate_fn
        )
        self.val_dataloader = DataLoader(
            self.val_dataset,
            config.DATA.batch_size,
            shuffle=False, 
            drop_last=False, 
            collate_fn=collate_fn
        )
        self.model = GPT2(config.MODEL, self.tokenizer).to(device)
        if self.config.MODEL.phase == 'train':
            if self.config.TRAIN.from_pretrained:
                self.load_model(self.config.TRAIN.from_pretrained)
                
            self.time_stamp = time.strftime("%Y%m%d_%H%M%S")
            if self.config.TRAIN.with_monitor:    
                if self.config.DATA.tokenizer == 'bbpe':
                    wandb.init(project="koyui_GPT2", name="train_" + self.time_stamp, config=dict(self.config))
                if self.config.DATA.tokenizer == 'bert':
                    # To ensure fair curve comparison
                    wandb.init(project="koyui_GPT2_bert", name="train_" + self.time_stamp, config=dict(self.config))
                
                
        elif self.config.MODEL.phase == 'test':
            self.load_model(self.config.TEST.from_pretrained)
            self.model.eval()
        
        elif self.config.MODEL.phase == 'evaluation':
            if self.config.EVALUATE.evaluate_folder:
                self.eval_folder = self.config.EVALUATE.evaluate_folder
                self.models_list = natsorted(os.listdir(self.eval_folder))
                self.time_stamp = osp.basename(self.eval_folder).split('gpt2_')[1]
                if self.config.DATA.tokenizer == 'bbpe':
                    wandb.init(project="koyui_GPT2", name="eval_" + self.time_stamp, config=dict(self.config))
                if self.config.DATA.tokenizer == 'bert':
                    wandb.init(project="koyui_GPT2_bert", name="eval_" + self.time_stamp, config=dict(self.config))
            else:
                self.load_model(self.config.EVALUATE.from_pretrained)
                self.model.eval()
            self.scorer = BERTScorer(model_type='./deps/bert-base-chinese', num_layers=5)   # From issues
             
    def load_model(self, model_path):
        milestone = torch.load(model_path, map_location=device)
        self.model.load_state_dict(milestone)

    def save_model(self, epoch):
        save_dir = self.config.TRAIN.model_save + '_' + self.time_stamp
        os.makedirs(save_dir, exist_ok=True)
        
        save_path = osp.join(save_dir, f'{epoch}.pth')
        torch.save(self.model.state_dict(), save_path)
        
    def generate(self, x):
        # TODO: Real generate with batch, now batch generate assumes all sentences in the same size without padding, 
        while x.shape[1] < self.config.DATA.max_tokens:
            with torch.no_grad():
                logits, _ = self.model(x)  # (B, T, vocab_size)
                logits = logits[:, -1, :]   # (B, vocab_size)
                probs = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, self.config.MODEL.topk, dim=-1)  # (B, topk), (B, topk)
                idxs = torch.multinomial(topk_probs, 1)    # (B, 1)
                token_id = torch.gather(topk_indices, -1, idxs)
                x = torch.cat([x, token_id], dim=-1)
        return x
    
    def train(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config.TRAIN.lr, 
            weight_decay=self.config.TRAIN.weight_decay
        )
        training_loop = trange(self.config.TRAIN.max_training_epochs)
        for epoch in training_loop:
            # Train
            loss_list = []
            for batch in tqdm(self.train_dataloader):
                optimizer.zero_grad()
                _, loss = self.model(batch["input"], batch["target"], batch["mask"])
                loss_list.append(loss.item())
                loss.backward()
                optimizer.step()
                
            train_loss = sum(loss_list)/len(loss_list)
            
            # Validation
            loss_list = []
            for batch in tqdm(self.val_dataloader):
                with torch.no_grad():
                    _, loss = self.model(batch["input"], batch["target"], batch["mask"])
                loss_list.append(loss.item())
            
            val_loss = sum(loss_list)/len(loss_list)
            
            training_loop.set_description(f'Train loss(element): {train_loss}, Validation loss(element): {val_loss}')
            
            wandb.log({"Train loss(element)": train_loss, "Validation loss(element)": val_loss})
            if epoch % self.config.TRAIN.save_per == 0:
                self.save_model(epoch)
    
    def batch_decode(self, batch):
        """
            batch: A batch of tokens generated by GPT2 model.
            Return a list of text decoded by our tokenizer.
        """
        
        if len(batch.shape) == 1:   # (T, )
            batch = batch.unsqueeze(0)  # (B, T)
        batch = batch.tolist()
        # print(batch)
        text_list = []
        for tokens in batch:
            eos_pos = len(tokens)
            sep_pos = None
            for pos, token in enumerate(tokens):
                if token == self.tokenizer.sep_token:
                    sep_pos = pos + 1
                elif token == self.tokenizer.eos_token:
                    eos_pos = pos
                    break
            decode_tokens = tokens[sep_pos:eos_pos]
            decode_tokens = [t for t in decode_tokens if t != self.tokenizer.pad_token]
            # print("Decoded tokens:", decode_tokens)
            # print("Length:", len(decode_tokens))
            text_list.append(self.tokenizer.decode(decode_tokens))
            print(text_list)
            
        return text_list
    
    def test(self):
        with open(config.TEST.from_file, 'r', encoding='utf-8') as f:
            text_list = f.readlines()
        results = []
        for text in text_list:
            prefix, suffix = text.split('[sep]')
            tokens = self.tokenizer.encode(prefix) + [self.tokenizer.sep_token] + self.tokenizer.encode(suffix)
            x = torch.as_tensor(tokens).unsqueeze(0).long().to(device)
            text = self.generate(x)
            decoded_text = self.batch_decode(text)
            results += decoded_text
            
        results = [result + '\n' for result in results]
        with open(self.config.TEST.result_save, 'w', encoding='utf-8') as f:
            f.writelines(results)
        print(f"Test done, save to {self.config.TEST.result_save}")
    
    def evaluate_single(self):
        loss_list = []
        for batch in tqdm(self.val_dataloader):
            with torch.no_grad():
                _, loss = self.model(batch["input"], batch["target"], batch["mask"])
            loss_list.append(loss.item())
        perplexity = torch.exp(torch.as_tensor(loss_list).mean()).item()

        val_datalist = self.val_dataset.data_list
        val_len = len(val_datalist)
        results = []
        val_len = min(val_len, self.config.EVALUATE.eval_length)
        for i in trange(val_len):
            if self.val_dataset.rating_list[i] == 1:
                tokens = self.tokenizer.encode("好看")
            else:
                tokens = self.tokenizer.encode("不好看")
            tokens.append(self.tokenizer.sep_token)
            x = torch.as_tensor(tokens).unsqueeze(0).long().to(device)
            text = self.generate(x)
            decoded_text = self.batch_decode(text)
            results += decoded_text
        P, R, F1 = self.scorer.score(results, val_datalist[:val_len])
        return perplexity, F1.mean().item()
        
    
    def evaluate(self):
        if self.config.EVALUATE.evaluate_folder:
            for model in self.models_list:
                self.load_model(osp.join(self.eval_folder, model))
                epoch = int(model.split('.')[0])
                perplexity, F1 = self.evaluate_single()
                wandb.log({"Perplexity": perplexity, "Bert-score(F1)": F1}, step=epoch)
        else:
            perplexity, F1 = self.evaluate_single()
            print("Perplexity:", perplexity)
            print("Bert-score(F1):", F1)

if __name__ == "__main__":
    config = OmegaConf.load('./configs/config.yaml')
    trainer = Trainer(config)
    if config.MODEL.phase == 'train':
        trainer.train()
    elif config.MODEL.phase == 'test':
        trainer.test()
    elif config.MODEL.phase == 'evaluation':
        trainer.evaluate()