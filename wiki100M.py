import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
import os

# output files
os.makedirs("wikitext_np", exist_ok=True)
train_file = "wikitext_np/train.npy"
val_file   = "wikitext_np/val.npy"

# load dataset
dataset = load_dataset("wikitext", "wikitext-103-v1")     # 100M tokens

# gpt2 tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']

def tokenize(doc):
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    return np.array(tokens, dtype=np.uint16)

# tokenize train
train_tokens = []
for doc in tqdm(dataset["train"], desc="Tokenizing train"):
    train_tokens.append(tokenize(doc))
train_tokens = np.concatenate(train_tokens)
np.save(train_file, train_tokens)

# tokenize validation
val_tokens = []
for doc in tqdm(dataset["validation"], desc="Tokenizing val"):
    val_tokens.append(tokenize(doc))
val_tokens = np.concatenate(val_tokens)
np.save(val_file, val_tokens)

print(f"Saved train: {train_tokens.shape}, val: {val_tokens.shape}")
