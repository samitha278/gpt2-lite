import torch
import torch.nn as nn
import torch.nn.functional as F 

import gpt2 



device = 'cuda' if torch.cuda.is_available() else 'cpu'
max_iter = 10000
lr = 1e-3

config = gpt2.GPT2Config









# Model
model = gpt2.GPT2.from_pretrained()

model.eval()
model.to(device)


# Sampling from the model
import tiktoken

enc = tiktoken.get_encoding('gpt2')

prompt = "Hello World ! I'm LLM"

tokens = enc.encode(prompt)
tokens = torch.tensor([tokens] ,dtype = torch.long)

x = tokens.to(device)

new_max_tokens = 20

while x.size(1) < new_max_tokens:

  with torch.no_grad():
    logits = model(x)

    probs = F.softmax(logits[:,-1,:],dim = -1)

    new_token = torch.multinomial(probs,  num_samples=1)

    x = torch.cat([x,new_token],dim=1)


out = x.view(-1).tolist()

text = enc.decode(out)
print(text)