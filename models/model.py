import math
import torch
import torch.nn as nn
from torch.nn import functional as F
    
class MultiheadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.qkv_proj = nn.Linear(config.n_emb, 3 * config.n_emb)
        self.n_head = config.n_head
        self.n_emb = config.n_emb
        self.out_proj = nn.Linear(config.n_emb, config.n_emb)
        self.head_dim = self.n_emb // self.n_head
        self.register_buffer("mask", torch.tril(torch.ones(config.max_tokens, config.max_tokens))
                             .view(1, 1, config.max_tokens, config.max_tokens)) 
            
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.n_emb, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = torch.einsum("bnij,bnkj->bnik", q, k).float() / math.sqrt(self.head_dim)  # (B, n_head, T, T)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = torch.einsum("bnij,bnjk->bnik", att, v).transpose(1, 2).contiguous().view(B, T, C)
        y = self.out_proj(y)
        return y
        

class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.n_emb, 4 * config.n_emb)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(4 * config.n_emb, config.n_emb)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x
    

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_emb)
        self.attn = MultiheadAttention(config)
        self.ln2 = nn.LayerNorm(config.n_emb)
        self.ffn = FFN(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_emb),
        self.wpe = nn.Embedding(config.max_tokens, config.n_emb),
        self.blocks = [TransformerBlock(config) for _ in range(config.n_layer)],
        self.ln_final = nn.LayerNorm(config.n_emb)
    
    def forward(self, idx):
        _, T = idx.shape
        assert T <= self.config.max_tokens, "Exceed the max tokens of the model."
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)   # (T, )
        pos_emb = self.wpe(pos) # (T, emb_size)
        tok_emb = self.wte(idx) # (B, T, emb_size)
        x = pos_emb + tok_emb
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        

class GPT2(nn.Module):
    def __init__(self, config):
        # config is config.MODEL for global config
        super().__init__()
        self.config = config
        self.transformer = Transformer(config)
        self.lm_head = nn.Linear(config.n_emb, config.vocab_size, bias=False)
        
    def forward(self, idx, targets=None, mask=None):
        # idx (B, T)
        x = self.transformer(idx)
        logits = self.lm_head(x)    # (B, T, v)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
            loss = mask * loss
        return logits, loss
        