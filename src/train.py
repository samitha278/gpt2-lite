import torch
import torch.nn as nn
import torch.nn.functional as F 
import tiktoken
import time
import math

from gpt2 import GPT2,GPT2Config 



device = 'cuda' if torch.cuda.is_available() else 'cpu'



#------------------------------------------------------------------


class DataLoader():

  def __init__(self,B,T):

    self.B = B
    self.T = T

    with open('data/input.txt', 'r') as f:
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




#------------------------------------------------------------------




B = 4
T = 1024


torch.manual_seed(278)
if torch.cuda.is_available():
  torch.cuda.manual_seed(278)


model = GPT2(GPT2Config(vocab_size = 50304))
model = model.to(device)
model = torch.compile(model)    # compile model into optimize form


data = DataLoader(B,T)

# _____________________________________________________________________________

# Learning Rate
max_lr = 6e-4
min_lr = max_lr * 0.1

max_iter = 1000
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


optimizer = torch.optim.AdamW(model.parameters(),lr = 6e-4,betas = (0.9,0.95),eps = 1e-8)
#Gradient Scalar
scaler = torch.amp.GradScaler(device)     # Prevents gradient underflow when using FP16

#optimize
for i in range(max_iter):

  t0 = time.time()   # time start

  xb , yb = data.get_batch()
  xb , yb = xb.to(device),yb.to(device)

  #AMP
  with torch.autocast(device_type=device, dtype=torch.float16):   # FP16
    logits , loss = model(xb,yb)

  optimizer.zero_grad()

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

  losses[i] = loss.item()
  norms[i] = norm.item()
  lrs[i] = lr

  if i%100==0 : print(f'{i}/{max_iter}  {loss.item():.4f}  {t:.4f} ms  norm:{norm.item():.4f}  lr:{lr:.4e}')