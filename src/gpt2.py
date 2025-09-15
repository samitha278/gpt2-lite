import torch 
import torch.nn as nn
import torch.nn.functional as F 
from dataclasses import dataclass


device = 'cuda' if torch.cuda.is_available() else 'cpu'




@dataclass
class GPT2Config:
    block_size : int = 128
    vocab_size : int = 65
    n_layer : int = 6
    n_head : int = 6
    n_embd : int = 128
    

    
# ----------------------------------------------------------------------------------


class GPT2(nn.Module):
    
    def __init__(self,config):
        super().__init__()
        
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size,config.n_embd),
            wpe = nn.Embedding(config.block_size,config.n_embd),
            
            block = nn.Sequential(*[Block(config) for i in range(config.n_layer)]),
            
            ln = nn.LayerNorm(config.n_embd),
        
        ))
        
        lm_head = nn.Linear(config.n_embd,config.vocab_size, bias=False)
        
        
        
    def forward(self,x,targets= None):
        
        tx = self.transformer.wte(x)       #token embedding
        px = self.transformer.wpe(torch.arnage(self.config.block_size,device=device)) #positional embedding
        
        x = tx+px     # add both
        
        x = self.transformer.block(x) 
        
        x = self.transformer.ln(x)
        
        logits = self.lm_head(x)
        
        
        if targets is None:
            return logits
        
        else:
            loss = F.cross_entropy(logits.view(-1,self.config.n_embd),targets.view(-1))
            return logits,loss
            
            
            
    def generate(self,idx,max_token):
        pass

        
        
    
# ----------------------------------------------------------------------------------



class Block(nn.Module):
    
    
    def __init__(self,config):
        super().__init__()
        self.config = config
        
        
        self.multi_head = Attention(config)
        self.mlp = MLP(config)
        
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        
        
        
    def forward(self,x):
        
        x = self.multi_head(self.ln1(x)) + x
        x = self.mlp(self.ln2(x)) + x
        
        return x
        
        
        
# ----------------------------------------------------------------------------------        
        
        
class MLP(nn.Module):
       
    def __init__(self,config):
        super().__init__()
        self.config = config
        
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd,4*config.n_embd),
            nn.GELU(),
            nn.Linear(4*config.n_embd,config.n_embd)            
        )
        
    def forward(self,x):
        out = self.mlp(x)
        return out
        
        
        
# ----------------------------------------------------------------------------------       
        
class SelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()

        block_size = config.block_size

        self.n_head = n_embd = config.n_head
        self.n_embd = n_head = config.n_embd

        assert n_embd % n_head == 0
        self.head_size = n_embd // n_head
        
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)     # fan out : n_head * 3 * head_size  
        
        self.c_proj = nn.Linear(n_embd, n_embd)
        
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size)))



    def forward(self, x):
        B, T, C = x.size()  # C = n_embd = n_head * head_size
        
        qkv = self.c_attn(x)    # B,T, 3*n_embd

        q, k, v = qkv.split(self.n_embd, dim=2)    # each : B,T, n_head * head_size
        
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)    # B, n_head, T, head_size
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)    # ""     
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)    # ""     
        
        att = (q @ k.transpose(-2, -1)) * (self.head_size**-0.5)         # B, n_head, T, T
        att = att.masked_fill(self.bias[:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        y = att @ v        # B, n_head, T, head_size
        y = y.transpose(1, 2).contiguous().view(B, T, C)   # B, T , n_embd   (n_embd = n_head * head_size)
        
        y = self.c_proj(y)
        return y
        
# ----------------------------------------------------------------------------------          