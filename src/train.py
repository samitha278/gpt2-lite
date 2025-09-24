import torch
import torch.nn as nn
import torch.nn.functional as F 
import tiktoken
import time

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




max_iter = 100
lr = 3e-4
B = 4
T = 1024


torch.manual_seed(278)
if torch.cuda.is_available():
  torch.cuda.manual_seed(278)


model = GPT2(GPT2Config())
model = model.to(device)



data = DataLoader(B,T)
optimizer = torch.optim.AdamW(model.parameters(),lr = lr)
losses = torch.zeros((max_iter,))


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
  scaler.step(optimizer)            # unscales gradients then call optimizer step        
  scaler.update()                   # adjusts the scale factor automatically each iteration


  torch.cuda.synchronize()   

  t1 = time.time()   # time end
  t = (t1 - t0)*1000 # ms

  losses[i] = loss.item()

  if i%10==0 : print(f'{i}/{max_iter}   {loss.item()}    {t} ms')

 