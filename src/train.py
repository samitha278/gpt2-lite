import torch
import torch.nn as nn
import torch.nn.functional as F 
import tiktoken
import time
import math
import numpy as np

from gpt2 import GPT2,GPT2Config 





device = 'cuda' if torch.cuda.is_available() else 'cpu'

enc = tiktoken.get_encoding('gpt2')

#------------------------------------------------------------------

# For wiki text
class DataLoader():

  def __init__(self,B,T,split):

    self.B = B
    self.T = T


    if split == 'train':
      train_tokens = np.load("/home/samitha/projects/gpt2-lite/wikitext_np/train.npy", mmap_mode="r") 
      self.tokens = torch.from_numpy(train_tokens.astype(np.int64))
    elif split == 'val':
      val_tokens   = np.load("/home/samitha/projects/gpt2-lite/wikitext_np/val.npy", mmap_mode="r")
      self.tokens   = torch.from_numpy(val_tokens.astype(np.int64))
    

    print(f'1 epoch size: {len(self.tokens)}')

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




#------------------------------------------------------------------



torch.manual_seed(278)
if torch.cuda.is_available():
    torch.cuda.manual_seed(278)


model = GPT2(GPT2Config(vocab_size = 50304))
model = model.to(device)
model = torch.compile(model)    # compile model into optimize form


# _____________________________________________________________________________


# Gradient Accumulation
total_batch_size = 2**16   # ~65K
B = 2**2       # mini batch size
T = 2**10     # contex length = 1024

ga_steps = total_batch_size // (B*T)  # gradient accumulation steps


data = DataLoader(B,T,'train')

import sys ; sys.exit(0)

# _____________________________________________________________________________

# Learning Rate
max_lr = 6e-4
min_lr = max_lr * 0.1

max_iter = 10000      # ~5 epochs , total train tokens ~119M , batch size ~65K 
warmup_steps = max_iter * 0.05

def next_lr(i):
    # warmup stage : linear
    if i < warmup_steps :
        return (max_lr/warmup_steps) * (i+1)

    if i > max_iter:
        return min_lr

    # cosine dacay
    decay_ratio = (i-warmup_steps) / (max_iter-warmup_steps)
    assert 0<= decay_ratio <=1
    c = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return min_lr + c * (max_lr - min_lr)


# _____________________________________________________________________________


losses = torch.zeros((max_iter,))
norms = torch.zeros((max_iter,))
lrs = torch.zeros((max_iter,))


# _____________________________________________________________________________

# Optimizer with weight decay custom function
optimizer = model.config_optimizers(weight_decay = 0.1 ,learning_rate = 6e-4,device=device)


#Gradient Scalar
scaler = torch.amp.GradScaler(device)     # Prevents gradient underflow when using FP16

#optimize
for i in range(max_iter):

    t0 = time.time()   # time start
    
        
    # Validation
    if i%100  == 0 :
      model.eval()      # evaluation mode
      val_data = DataLoader(B,T,'val')
      
      with torch.no_grad():
        
        val_steps = 32
        val_loss = 0.0
        
        for _ in range(val_steps):
          xb,yb = val_data.get_batch()
          xb,yb = xb.to(device), yb.to(device)
      
          with torch.autocast(device_type=device,dtype=torch.bfloat16):
            logits , loss = model(xb,yb)
          
          val_loss += loss.detach()  
        val_loss /= val_steps
        
        
        print(f'Validation loss : {val_loss.item():.4f}')
      
    
    
    # Inference 
    if i%100 == 0 and i>0:
      model.eval()
      
      
    with torch.no_grad():
      seq = 2
      max_tokens = 32
      
      tokens = enc.encode(f'Hello World!, I\'m gpt 2')
      tokens = torch.tensor(tokens,dtype = torch.long)
      tokens = tokens.unsqueeze(0).repeat(seq,1)
      
      x = tokens.to(device)
      gen = torch.Generator(device=device)
      gen.manual_seed(278)
      
      while x.size(1) < max_tokens:

        with torch.no_grad():
          logits = model(x)
          probs = F.softmax(logits[:,-1,:],dim = -1)
          topk_probs , topk_indicies = torch.topk(probs , 50 ,dim = -1)
      
          ix = torch.multinomial(topk_probs,  num_samples=1 ,generator=gen)    
          x_col = torch.gather(topk_indicies , -1 , ix)
          x = torch.cat((x,x_col),dim=1)
          
      for i in range(seq):
        tokens = x[i].tolist()
        text = enc.decode(tokens)
        print(text)
      
      
    
    # Train
    optimizer.zero_grad()

    loss_ = 0.0

    # Gradient Accumulation loop for mini batch
    for step in range(ga_steps):
        xb , yb = data.get_batch()
        xb , yb = xb.to(device),yb.to(device)

        #AMP
        with torch.autocast(device_type=device, dtype=torch.bfloat16):   # BF16
            logits , loss = model(xb,yb)

        loss /= ga_steps                  # normalize loss  
        
        loss_ += loss.detach()
        scaler.scale(loss).backward()     # multiplies loss by a scale factor




    norm = nn.utils.clip_grad_norm_(model.parameters(),1.0)    # inplace gradient clipping

    # find and set learning rate
    lr = next_lr(i)

    # update optimizer this new lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


    scaler.step(optimizer)            # unscales gradients then call optimizer step
    scaler.update()                   # adjusts the scale factor automatically each iteration


    torch.cuda.synchronize() if torch.cuda.is_available() else None

    t1 = time.time()   # time end
    t = (t1 - t0)*1000 # ms

    losses[i] = loss_.item()
    norms[i] = norm.item()
    lrs[i] = lr

    if i%10==0 : print(f'{i}/{max_iter}  {loss_.item():.4f}  {t:.4f} ms  norm:{norm.item():.4f}  lr:{lr:.4e}')