import torch 
import torch.nn as nn
import torch.nn.functional as F 
from dataclasses import dataclass






@dataclass
class GPT2Config:
    block_size : int = 128
    vocab_size : int = 65
    n_layer : int = 6
    n_head : int = 6
    n_embd : int = 128
    



class GPT2(nn.Module):
    
    def __init__(self,config):
        super().__init__()
        
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size,config.n_embd),
            wpe = nn.Embedding(config.block_size,config.n_embd),
            
            h = nn.ModuleList(*[Block(config) for i in range(config.n_layer)]),
            
            ln = nn.LayerNorm(config.n_embd),
            lm_head = nn.Linear(config.n_embd,config.vocab_size)
        
        ))
        
        
        
    def forward(self,x):
        pass
        
        
        
    





class Block(nn.Module):
    
    
    def __init__(self,config):
        super().__init__()
        self.config = config
        
        
        self.multi_head = MultiHead()
        self.projection = nn.Linear(n_embd,)
        
        
        
        
        
        
class MPL(nn.Module):
       
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.mlp = nn.Sequential(
            nn.Linear(n_embd,4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd,n_embd)            
        )
        
    def forward(self,x):
        out = self.mlp(x)
        return out
        
        
        
        
        
class MultiHead(nn.Module):
    
    def __init__(self,config):
        super().__init__()
        self.config = config
        
        head_size = n_embd // n_head

        self.sa_heads = nn.ModuleList(*[SelfAttentionHead(head_size) for i in range(n_head)])
        self.projection = nn.Linear(n_embd,n_embd)
        
        
    def forward(self,x):
        
        out =  torch.cat([sa(x) for sa in self.sa_heads],dim=1)
        out = self.projection(out)
        
        return out
    
        
        
      
      
  
        
        
        
class SelfAttentionHead(nn.Module):
    
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.head_size = head_size
        
        self.k = nn.Linear(n_embd,head_size)
        self.q = nn.Linear(n_embd,head_size)
        self.v = nn.Linear(n_embd,head_size)
        
        self.register_buffer('tril' , torch.tril(torch.ones(block_size,block_size)))
        
        
        
    def forward(self,x):
        
        
        
        key = self.k(x)
        query = self.q(x)
        
        weight = query @ key.transpose(-2,-1) * self.head_size**-0.5
        
        
        
        
        
        