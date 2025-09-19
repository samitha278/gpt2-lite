import torch
import torch.nn as nn
import torch.nn.functional as F 

import gpt2 



device = 'cuda' if torch.cuda.is_available() else 'cpu'
max_iter = 10000
lr = 1e-3

config = gpt2.GPT2Config







#------------------------------------------------------------------


#text read
with open('data/input.txt', 'r') as f:
    text = f.read()
    

#get characters from input.txt
chars = sorted(list(set(text)))
vocab_size = len(chars)


#encoding and decoding
stoi = {s:i for i,s in enumerate(chars)}
itos = {i:s for i,s in enumerate(chars)}

encode = lambda s: [stoi[ch] for ch in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = encode(text)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

#------------------------------------------------------------------


def get_batch(split):
    bs = config.block_size
    data = train_data if split=='train' else val_data
    
    ix = torch.randin(len(data)-bs,(bs,))
    x = torch.tensor([data[i:i+bs] for i in ix])
    y = torch.tensor([data[i+1:i+1+bs] for i in ix])

    x,y = x.to(device) , y.to(device)
    return x,y


#------------------------------------------------------------------



gpt2_model = gpt2.GPT2(config)
gpt2_model = gpt2_model.to(device)


#------------------------------------------------------------------


optimizer = torch.optim.adamW(gpt2_model.parameters, lr = lr)

for i in range(max_iter):
    
    xb,yb = get_batch('train')
    logits , loss = gpt2_model(xb,yb)
    
    #train
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()


    #print losses
    if i% (max_iter/10) == 0:
        print(f'{i}/{max_iter}  {loss}')
    if i == max_iter-1:
        print(f'{max_iter}/{max_iter}  {loss}')
        
        
#------------------------------------------------------------------


