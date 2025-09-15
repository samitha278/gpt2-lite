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
        
class Attention(nn.Module):


  def __init__(self,config):
    super().__init__()

    block_size = config.block_size
    n_embd = config.n_embd
    n_head = config.n_head
    head_size = n_embd // n_head 
    self.head_size = head_size


    self.key = nn.ModuleList(nn.Linear(n_embd,head_size) for _ in range(n_head))
    self.query = nn.ModuleList(nn.Linear(n_embd,head_size) for _ in range(n_head))
    self.value = nn.ModuleList(nn.Linear(n_embd,head_size) for _ in range(n_head))

    self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))


  def forward(self,x):


    key = torch.stack([k(x) for k in self.key],dim=1)
    query = torch.stack([q(x) for q in self.query],dim=1)

    weight = query @ key.transpose(-1,-2)
    weight = weight.masked_fill(self.tril[:]==0,float('-inf'))
    weight = F.softmax(weight,dim=-1)

    value = torch.stack([v(x) for v in self.value],dim=1)

    out = weight @ value

    B,nh,T,C = out.shape
    out = out.permute(0,2,1,3)
    out = out.reshape(B,T,nh*C)

    return out
        
# ----------------------------------------------------------------------------------          