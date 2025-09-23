import torch
import torch.nn as nn
import torch.nn.functional as F 
import tiktoken

import gpt2 






device = 'cuda' if torch.cuda.is_available() else 'cpu'
max_iter = 20 # 10000
lr = 3e-4



#------------------------------------------------------------------

#Initialize model

config = gpt2.GPT2Config

gpt2_model = gpt2.GPT2(config)
gpt2_model = gpt2_model.to(device)


#------------------------------------------------------------------


#text read
with open('data/input.txt', 'r') as f:
    text = f.read()
    


#get encoder
enc = tiktoken.get_encoding('gpt2')


#prepare data to train
data = text[:1000]
tokens = enc.encode(data)        #encoding 



#Just one batch

B,T = 4,8
temp = torch.tensor(tokens[:B*T+1])
temp.to(device)
xb = temp[:-1].view(B,T)
yb = temp[1:].view(B,T)



#------------------------------------------------------------------




#Train
losses = torch.zeros((max_iter,))


optimizer = torch.optim.AdamW(gpt2_model.parameters(),lr=lr)

for i in range(max_iter):

    # xb,yb = get_batch('train')
    
    logits , loss = gpt2_model(xb,yb)
    
    
    #train
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    
    
    losses[i] = loss.item()   # store losses
    



print(losses[-1])
 