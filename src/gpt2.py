import torch 
import torch.nn as nn
import torch.nn.functional as F 
from dataclasses import dataclass


device = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class GPT2Config:
    block_size : int = 1024
    vocab_size : int = 50257
    n_layer : int = 12
    n_head : int = 12
    n_embd : int = 768


# ----------------------------------------------------------------------------------


class GPT2(nn.Module):

    def __init__(self,config):
        super().__init__()

        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size,config.n_embd),
            wpe = nn.Embedding(config.block_size,config.n_embd),

            h = nn.ModuleList([Block(config) for i in range(config.n_layer)]),

            ln_f = nn.LayerNorm(config.n_embd),

        ))

        self.lm_head = nn.Linear(config.n_embd,config.vocab_size, bias=False)


        #weight sharing
        self.transformer.wte.weight = self.lm_head.weight
        #Save ~38M parameters

        self.apply(self._init_weights)



    def _init_weights(self,module):

        if isinstance(module,nn.Linear):
            std = 0.02
            if hasattr(module,'FLAG'):
                std *= (2*self.config.n_layer) ** -0.5           #scaledown std
            torch.nn.init.normal_(module.weight,mean=0.0,std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)



    def forward(self,x,targets= None):

        B,T = x.shape
        assert T<= self.config.block_size   # positional embd table max size = block_size
        tx = self.transformer.wte(x)       #token embedding
        px = self.transformer.wpe(torch.arange(0,T,self.config.block_size,device=device)) #positional embedding

        x = tx+px     # add both

        for block in self.transformer.h:
          x = block(x)

        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)

        if targets is None:
            return logits
        else:
            loss = F.cross_entropy(logits.view(B*T,-1) ,targets.view(-1))
            return logits,loss



    @classmethod
    def from_pretrained(cls, model_type='gpt2'):
        from transformers import GPT2LMHeadModel
        assert model_type == 'gpt2'

        config_args = dict(n_layer=12, n_head=12, n_embd=768, vocab_size=50257, block_size=1024)
        config = GPT2Config(**config_args)
        model = GPT2(config)

        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = [k for k in sd_hf.keys() if not k.endswith(('.attn.masked_bias', '.attn.bias'))]

        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys)

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model



    def config_optimizers(self,weight_decay , learning_rate , device):

        param_dict = {name:p for name,p in self.named_parameters() if p.requires_grad}

        #weight decay for only for parameter (dimension >= 2) tensors
        decay_params = [p for name,p in param_dict.items() if p.dim()>=2]
        nondecay_params = [p for name,p in param_dict.items() if p.dim()<2]

        optim_groups = [
            {'params':decay_params , 'weight_decay' :weight_decay},
            {'params':nondecay_params,'weight_decay':0.0}
        ]

        use_fused = 'cuda' in device     # kernel fusion AdamW
        optimizer = torch.optim.AdamW(optim_groups,lr=learning_rate,betas =(0.9,0.95),eps=1e-8,fused=use_fused)
        return optimizer



# ----------------------------------------------------------------------------------



class Block(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.config = config

        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)


    def forward(self,x):

        x = self.attn(self.ln_1(x)) + x
        x = self.mlp(self.ln_2(x)) + x

        return x



# ----------------------------------------------------------------------------------


class MLP(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.config = config

        self.c_fc = nn.Linear(config.n_embd,4*config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4*config.n_embd,config.n_embd)
        self.c_proj.FLAG = 1


    def forward(self,x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)

        return x

# ----------------------------------------------------------------------------------

class SelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()

        block_size = config.block_size

        self.n_head = n_head = config.n_head
        self.n_embd = n_embd = config.n_embd


        assert n_embd % n_head == 0
        self.head_size = n_embd // n_head

        self.c_attn = nn.Linear(n_embd, 3 * n_embd)     # fan out : n_head * 3 * head_size

        self.c_proj = nn.Linear(n_embd, n_embd)

        self.c_proj.FLAG = 1

        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size)))



    def forward(self, x):
        B, T, C = x.size()  # C = n_embd = n_head * head_size

        qkv = self.c_attn(x)    # B,T, 3*n_embd

        q, k, v = qkv.split(self.n_embd, dim=2)    # each : B,T, n_head * head_size

        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)    # B, n_head, T, head_size
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)    # ""
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)    # ""

        # att = (q @ k.transpose(-2, -1)) * (self.head_size**-0.5)         # B, n_head, T, T
        # att = att.masked_fill(self.bias[:T,:T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)

        # y = att @ v        # B, n_head, T, head_size


        # Flash Attention
        y = F.scaled_dot_product_attention(q,k,v,is_causal=True)


        y = y.transpose(1, 2).contiguous().view(B, T, C)   # B, T , n_embd   (n_embd = n_head * head_size)

        y = self.c_proj(y)
        return y





# ----------------------------------------------------------------------------------

class DataLoader():

  def __init__(self,B,T):

    self.B = B
    self.T = T

    with open('input.txt', 'r') as f:
      text = f.read()

    enc = tiktoken.get_encoding('gpt2')
    self.tokens = torch.tensor(enc.encode(text))

    print(f'1 epoch size: {len(self.tokens//B*T)}')

    self.count = 0



  def get_batch(self):

    B,T = self.B , self.T

    temp = self.tokens[self.count:self.count+B*T+1]

    x = temp[:-1].view(B,T)   #inputs
    y = temp[1:].view(B,T)    #targets

    self.count += B*T

    # Reset
    if (self.count+B*T+1) > len(self.tokens):
      self.count = 0

    return x,y
