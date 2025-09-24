import torch
import torch.nn as nn
import torch.nn.functional as F 
import tiktoken

from gpt2 import GPT2,GPT2Config 



device = 'cuda' if torch.cuda.is_available() else 'cpu'


torch.manual_seed(278)
if torch.cuda.is_available():
  torch.cuda.manual_seed(278)

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




max_iter = 1000
lr = 3e-4
B = 32
T = 8


#initialize model
model = GPT2(GPT2Config())
model = model.to(device)

#create dataloader
data = DataLoader(B,T)

#optimizer object
optimizer = torch.optim.AdamW(model.parameters(),lr = lr)

#for store losses
losses = torch.zeros((max_iter,))

#optimize
for i in range(max_iter):
  
    #get_batch
    xb , yb = data.get_batch()
    xb , yb = xb.to(device),yb.to(device)

    #forward pass the model
    logits , loss = model(xb,yb)

    optimizer.zero_grad()   #if not gradients will added to previous ones
    loss.backward()         #backpropagation and calculate gradients
    optimizer.step()        #update the parameters

    losses[i] = loss.item() #store loss

    if i%100==0:print(f'{i}/{max_iter}   {loss.item()}')


    



print(losses[-1])
 