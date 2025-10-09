import torch
import torch.nn as nn
import torch.nn.functional as F 
import tiktoken
import time
import math
import numpy as np
import os
import random

from gpt2 import GPT2,GPT2Config 





device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)
# import sys;sys.exit(0)


enc = tiktoken.get_encoding('gpt2')

#------------------------------------------------------------------

# For wiki text
class DataLoader():

  def __init__(self,B,T,split):

    self.B = B
    self.T = T

    if split == 'train':
      train_tokens = np.load("/home/samitha/Projects/gpt2-lite/wikitext_np/train.npy", mmap_mode="r") 
      self.tokens = torch.from_numpy(train_tokens.astype(np.int64))
    elif split == 'val':
      val_tokens   = np.load("/home/samitha/Projects/gpt2-lite/wikitext_np/val.npy", mmap_mode="r")
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


def load_checkpoint(checkpoint_path,model,optimizer):

  print(f"Loading checkpoint from {checkpoint_path}")
  checkpoint = torch.load(checkpoint_path, map_location=device,weights_only=False) # trust checkpoint file
  
  # load weights
  model.load_state_dict(checkpoint['model'])
  
  # load optimizer
  optimizer.load_state_dict(checkpoint['optimizer'])
            
  start_step = checkpoint.get('step', 0)
  val_loss = checkpoint.get('val_loss', 'N/A')
  print(f"Resumed from step {start_step}, val_loss: {val_loss}")
    
  return start_step
  
  

#------------------------------------------------------------------


torch.manual_seed(278)
if torch.cuda.is_available():
    torch.cuda.manual_seed(278)


model = GPT2(GPT2Config(vocab_size = 50304))
model = model.to(device)



# _____________________________________________________________________________


# Gradient Accumulation
total_batch_size = 2**16   # ~65K
B = 2**2       # mini batch size
T = 2**10     # contex length = 1024

ga_steps = total_batch_size // (B*T)  # gradient accumulation steps


data = DataLoader(B,T,'train')

# import sys ; sys.exit(0)

# _____________________________________________________________________________

# Learning Rate
max_lr = 6e-4
min_lr = max_lr * 0.1

max_iter = 12000  # total train tokens ~119M , batch size ~65K 
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


# losses = torch.zeros((max_iter,))
# norms = torch.zeros((max_iter,))
# lrs = torch.zeros((max_iter,))


# _____________________________________________________________________________

# og directory to write checkpoints and log 
log_dir = "log_1"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")

# _____________________________________________________________________________


# Optimizer with weight decay custom function
optimizer = model.config_optimizers(weight_decay = 0.1 ,learning_rate = 6e-4,device=device)


#Gradient Scalar
# scaler = torch.amp.GradScaler(device)     # Prevents gradient underflow when using FP16


#------------------------------------------------------------------


# Load Trained Model
resume_from = None # "/home/samitha/projects/gpt2-lite/src/log/model_39999.pt"
start_step = 0

if resume_from and os.path.exists(resume_from):
    start_step = load_checkpoint(resume_from, model, optimizer)
    start_step += 1
else:
    if resume_from:
      print(f'{resume_from} not found')
    print('starting from step 0')
    
    # with open(log_file, "w") as f: 
    #   pass


#------------------------------------------------------------------

# compile model into optimize form
model_compile = torch.compile(model)    

# Training main Loop
for i in range(start_step,max_iter):

    t0 = time.time()   # time start
    
    final_step = (i == max_iter - 1) 
    
    #----------------------------------------------------------------------
     
    # Validation
    if i% 200  == 0 or final_step:
      model.eval()      # evaluation mode
      val_data = DataLoader(B,T,'val')
      
      with torch.no_grad():
        
        val_steps = 64
        val_loss = 0.0
        
        for _ in range(val_steps):
          xb,yb = val_data.get_batch()
          xb,yb = xb.to(device), yb.to(device)
      
          with torch.autocast(device_type=device,dtype=torch.bfloat16):
            logits , loss = model(xb,yb)
          
          val_loss += loss.detach()  
        val_loss /= val_steps
        
        
        print(f'Validation loss : {val_loss.item():.4f}')
        with open(log_file, "a") as f:
            f.write(f"{i} val {val_loss.item():.4f}\n")
            
        # Check point
        if i>start_step and (i % 10000 == 0 or final_step):
          checkpoint_path = os.path.join(log_dir, f"model_{i:05d}.pt")
          checkpoint = {
              'model': model.state_dict(),
              'config': model.config,
              'step': i,
              'val_loss': val_loss.item(),
              'optimizer': optimizer.state_dict()
          }
          torch.save(checkpoint, checkpoint_path)
          print(f"Checkpoint saved: {checkpoint_path}")
      
    #----------------------------------------------------------------------
    
    # Inference 
    if (i % 200 == 0 and i>start_step) or final_step:
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
            
        for j in range(seq):
          tokens = x[j].tolist()
          text = enc.decode(tokens)
          print(text)
      
    #----------------------------------------------------------------------
    
    # Train
    optimizer.zero_grad()

    loss_ = 0.0

    # Gradient Accumulation loop for mini batch
    for step in range(ga_steps):
        xb , yb = data.get_batch()
        xb , yb = xb.to(device),yb.to(device)

        #AMP
        with torch.autocast(device_type=device, dtype=torch.bfloat16):   # BF16
            logits , loss = model_compile(xb,yb)

        loss /= ga_steps                  # normalize loss  
        
        loss_ += loss.detach()
        #scaler.scale(loss).backward()     # multiplies loss by a scale factor
        loss.backward()



    norm = nn.utils.clip_grad_norm_(model.parameters(),1.0)    # inplace gradient clipping

    # find and set learning rate
    lr = next_lr(i)

    # update optimizer this new lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


    #scaler.step(optimizer)            # unscales gradients then call optimizer step
    #scaler.update()                   # adjusts the scale factor automatically each iteration

    optimizer.step()    # update parameters

    torch.cuda.synchronize() if torch.cuda.is_available() else None

    t1 = time.time()   # time end
    t = (t1 - t0)*1000 # ms

    # losses[i] = loss_.item()
    # norms[i] = norm.item()
    # lrs[i] = lr

    print(f'{i}/{max_iter}  {loss_.item():.4f}  {t:.4f} ms  norm:{norm.item():.4f}  lr:{lr:.4e}')
    with open(log_file, "a") as f:
        f.write(f"{i} train {loss_.item():.6f}\n")