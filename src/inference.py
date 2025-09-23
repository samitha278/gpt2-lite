import torch
import torch.nn as nn
import torch.nn.functional as F 

import gpt2 



device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)





# Model
model = gpt2.GPT2.from_pretrained()

model.eval()
model.to(device)





new_max_tokens = 32
n_seq = 5





# Encode
import tiktoken

enc = tiktoken.get_encoding('gpt2')

prompt = "Hello World ! I'm LLM"

tokens = enc.encode(prompt)
tokens = torch.tensor(tokens ,dtype = torch.long)
tokens = tokens.unsqueeze(0).repeat(n_seq,1)           # n_seq , n_token

x = tokens.to(device)







# Sampling loop

while x.size(1) < new_max_tokens:

  with torch.no_grad():
    
    
    logits = model(x)

    probs = F.softmax(logits[:,-1,:],dim = -1)
    
    topk_probs , topk_indicies = torch.topk(probs , 50 ,dim = -1)
    

    ix = torch.multinomial(topk_probs,  num_samples=1)
    
    x_col = torch.gather(topk_indicies , -1 , ix)

    x = torch.cat((x,x_col),dim=1)








# Decode

for i in range(n_seq):
  
  
  tokens = x[i].tolist()
  text = enc.decode(tokens)
  print(text)