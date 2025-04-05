---
title: 使用Pytorch从零构建Llama3大模型--组件实现代码
date: 2025-03-22
authors: [KenForever1]
categories: 
  - llm
labels: []
comments: true
---

## 如何实现Input Block

<!-- more -->
```python
# Import necessary libraries
import torch
from torch import nn
from torch.nn import functional as F

import math
import numpy as np
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List
import pandas as pd
from matplotlib import pyplot as plt

### Step 1: Input Block ###

# Using Tiny Shakespeare dataset for character-level tokenizer. Some part of the following character-level tokenizer is referenced from Andrej karpathy's GitHub (https://github.com/karpathy/nanoGPT/blob/master/data/shakespeare_char/prepare.py) which I found is explained very well.
# Load tiny_shakespeare data file (https://github.com/tamangmilan/llama3/blob/main/tiny_shakespeare.txt)

device: str = 'cuda' if torch.cuda.is_available() else 'cpu'   # Assign device to cuda or cpu based on availability

# Load tiny_shakespeare data file.
with open('tiny_shakespeare.txt', 'r') as f:
  data = f.read()

# Prepare vocabulary by taking all the unique characters from the tiny_shakespeare data
vocab = sorted(list(set(data)))

# Training Llama 3 model requires addtional tokens such as <|begin_of_text|>, <|end_of_text|> and <|pad_id|>, we'll add them into vocabulary
vocab.extend(['<|begin_of_text|>','<|end_of_text|>','<|pad_id|>'])
vocab_size = len(vocab)

# Create a mapping between characters with corresponding integer indexes in vocabulary.
# This is important to build tokenizers encode and decode functions.
itos = {i:ch for i, ch in enumerate(vocab)}
stoi = {ch:i for i, ch in enumerate(vocab)}

# Tokenizers encode function: take a string, output a list of integers
def encode(s):
  return [stoi[ch] for ch in s]

# Tokenizers decode function: take a list of integers, output a string
def decode(l):
  return ''.join(itos[i] for i in l)

# Define tensor token variable to be used later during model training
token_bos = torch.tensor([stoi['<|begin_of_text|>']], dtype=torch.int, device=device)
token_eos = torch.tensor([stoi['<|end_of_text|>']], dtype=torch.int, device=device)
token_pad = torch.tensor([stoi['<|pad_id|>']], dtype=torch.int, device=device)

prompts = "Hello World"
encoded_tokens = encode(prompts)
decoded_text = decode(encoded_tokens)

### Test: Input Block Code ###
# You need take out the triple quotes below to perform testing
"""
print(f"Lenth of shakespeare in character: {len(data)}")
print(f"The vocabulary looks like this: {''.join(vocab)}\n")
print(f"Vocab size: {vocab_size}")
print(f"encoded_tokens: {encoded_tokens}")
print(f"decoded_text: {decoded_text}")
"""
### Test Results: ###
"""
Lenth of shakespeare in character: 1115394
The vocabulary looks like this: 
 !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz<|begin_of_text|><|end_of_text|><|pad_id|>

Vocab size: 68
encoded_tokens: [20, 43, 50, 50, 53, 1, 35, 53, 56, 50, 42]
decoded_text: Hello World
"""
```

## 如何实现RMS Norm

```python
# Step2: The Decoder Block
# Note: Since the Llama 3 model is developed by Meta, so to be in sync with their codebase and for future compatibility,
# I will use most of the code from Meta GitHub with some necessary changes required to achieve our goal.

# Define parameters dataclass: we'll use these parameters during model building, training and inference.
# Note: Since we want to see the results of training and inferencing faster rather than focusing on high accuracy, we're taking lower values for most of the parameters which are set higher in the Llama 3 model.

@dataclass
class ModelArgs:
    dim: int = 512              # embedding dimension
    n_layers: int = 8           # number of model decoder blocks
    n_heads: int = 8            # number of heads for queries embedding
    n_kv_heads: int = 4         # number of heads for keys and values embedding
    vocab_size: int = len(vocab) # Length of vocabulary
    multiple_of: int = 256        # Require to calculate dim of feedfoward network
    ffn_dim_multiplier: Optional[float] = None  # Require to calculate dim of feedfoward network
    norm_eps: float = 1e-5                       # Default Epsilon value set for the RMSNorm calculation
    rope_theta: float = 10000.0   # Default theta value for the RePE calculation

    max_batch_size: int = 10     # Max batch size
    max_seq_len: int = 256         # Max sequence length

    epochs: int = 2500             # Total number of training iteration
    log_interval: int = 10        # Number of interval to print the logs and loss values   
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'   # Assign device to cuda or cpu based on availability 
```

## RoPE的实现

```python
## Step2b: The RoPE
def precompute_freqs_cis(dim:int, seq_len: int, theta: float=10000.0):
  # Computing Theta value for each dim pair which is dim/2
  device = ModelArgs.device
  freqs = 1.0 / (theta ** (torch.arange(0, dim, 2,device=device)[:(dim//2)].float()/dim))

  # Computing range of positions(m) in the sequence
  t = torch.arange(seq_len, dtype=torch.float32, device=device)

  # freqs gives all the Theta value range for all the position of tokens in the sequence
  freqs = torch.outer(t, freqs).to(device)

  # This is the rotation matrix which needs to be converted to Polar form in order to perform rotation to the embedding
  freqs_cis = torch.polar(torch.ones_like(freqs).to(device), freqs).to(device)
  return freqs_cis

def reshape_for_broadcast(freqs_cis, x):
  ndim = x.ndim
  assert 0<=1<ndim
  assert freqs_cis.shape == (x.shape[1],x.shape[-1]), "the last two dimension of freqs_cis, x must match"
  shape = [d if i==1 or i==ndim-1 else 1 for i,d in enumerate(x.shape)]
  return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:
  device = ModelArgs.device
  # Applying rotary positional encoding to both query and key embedding together
  # First: The last dimension of xq and xk embedding needs to be reshaped to make it a pair. As rotation matrix is applied to each pair of dim.
  # Next: convert both xq and xk to complex number as the rotation matrix is only applicable to complex number
  xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)).to(device)    #xq_:[bsz, seq_len, n_heads, head_dim/2]
  xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2)).to(device)    #xk_:[bsz, seq_len, n_heads, head_dim/2]

  # The rotation matrix(freqs_cis) dimensions across seq_len(dim=1) and head_dim(dim=3) should match with the embedding
  # Also, the shape freqs_cis should be the same with xq and xk, hence change the shape of freqs_cis:[seq_len,head_dim] -> freqs_cis:[1,seq_len,1,head_dim]
  freqs_cis = reshape_for_broadcast(freqs_cis, xq_)

  #Finally, perform rotation operation by multiplying with freqs_cis.
  #After the rotation is completed, convert both xq_out and xk_out back to real number and return
  xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3).to(device) #xq_out:[bsz, seq_len, n_heads, head_dim]
  xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3).to(device) #xk_out:[bsz, seq_len, n_heads, head_dim]
  return xq_out.type_as(xq), xk_out.type_as(xk)

### Test: RoPE Code ###
# Note: x_norm is calculated during RMSNorm and is being used for testing here.
# You need take out the triple quotes below to perform testing
"""
head_dim = ModelArgs.dim//ModelArgs.n_heads
wq = nn.Linear(ModelArgs.dim, ModelArgs.n_heads * head_dim, bias=False, device=device)
wk = nn.Linear(ModelArgs.dim, ModelArgs.n_kv_heads * head_dim, bias=False, device=device)
xq = wq(x_norm)
xk = wk(x_norm)
print(f"xq.shape: {xq.shape}")
print(f"xk.shape: {xk.shape}")

xq = xq.view(xq.shape[0],xq.shape[1],ModelArgs.n_heads, head_dim)
xk = xk.view(xk.shape[0],xk.shape[1],ModelArgs.n_kv_heads, head_dim)
print(f"xq.re-shape: {xq.shape}")
print(f"xk.re-shape: {xk.shape}")

freqs_cis = precompute_freqs_cis(dim=head_dim, seq_len=ModelArgs.max_seq_len)
print(f"freqs_cis.shape: {freqs_cis.shape}")

xq_rotate, xk_rotate = apply_rotary_emb(xq, xk, freqs_cis)
print(f"xq_rotate.shape: {xq_rotate.shape}")
print(f"xk_rotate.shape: {xk_rotate.shape}")
"""
### Test Results: ###
"""
xq.shape: torch.Size([10, 256, 512])
xk.shape: torch.Size([10, 256, 256])
xq.re-shape: torch.Size([10, 256, 8, 64])
xk.re-shape: torch.Size([10, 256, 4, 64])
freqs_cis.shape: torch.Size([256, 32])
xq_rotate.shape: torch.Size([10, 256, 8, 64])
xk_rotate.shape: torch.Size([10, 256, 4, 64])
"""
```

## 分组注意力机制实现

```python
## The Attention Block [Step2c: The KV Cache; Step2d: Group Query Attention]
## As mentioned before, the naming convention follows original the meta's LLama3 GitHub

class Attention(nn.Module):
  def __init__(self, args: ModelArgs):
    super().__init__()
    self.args = args
    # Embedding dimension
    self.dim = args.dim
    # Number of heads assigned to Query
    self.n_heads = args.n_heads
    # Number of heads assigned to Key and values. If "None", the number will be same as Query.
    self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
    # Dimension of each head relative to model dimension
    self.head_dim = args.dim // args.n_heads
    # Number of repetition in order to make time Key, Value heads to match Query heads number
    self.n_rep = args.n_heads // args.n_kv_heads

    # Weight initialize for Keys, Querys, Values and Oupt. Notice that the out_feature value of weight for q and kv are based on it's heads
    self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False, device=device)
    self.wk = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False, device=device)
    self.wv = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False, device=device)
    self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False, device=device)

    # Initialize caches to store Key, Values at start. (KV Cache Implementation)
    self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim), device=args.device)
    self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim), device=args.device)

  def forward(self, x: torch.Tensor, start_pos, inference):
    # Shape of the input embedding: [bsz,seq_len,dim]
    bsz, seq_len, _ = x.shape
    # Mask will be used during 'Training' and is not required for 'inference' due to the use of KV cache.
    mask = None

    xq = self.wq(x)  #x[bsz,seq_len,dim]*wq[dim,n_heads * head_dim] -> q[bsz,seq_len,n_heads * head_dim]
    xk = self.wk(x)  #x[bsz,seq_len,dim]*wq[dim,n_kv_heads * head_dim] -> k[bsz,seq_len,n_kv_heads * head_dim]
    xv = self.wv(x)  #x[bsz,seq_len,dim]*wq[dim,n_kv_heads * head_dim] -> v[bsz,seq_len,n_kv_heads * head_dim]

    # Reshaping Querys, Keys and Values by their number of heads. (Group Query Attention Implementation)
    xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)      #xq[bsz,seq_len,n_heads, head_dim]
    xk = xk.view(bsz, seq_len, self.n_kv_heads, self.head_dim)   #xk[bsz,seq_len,n_kv_heads, head_dim]
    xv = xv.view(bsz, seq_len, self.n_kv_heads, self.head_dim)   #xv[bsz,seq_len,n_kv_heads, head_dim]

    # Model - Inference Mode: kv-cache is enabled at inference mode only.
    if inference:
      # Compute rotation matrix for each position in the sequence
      freqs_cis = precompute_freqs_cis(dim=self.head_dim, seq_len=self.args.max_seq_len * 2)
      # During inferencing, we should only take the rotation matrix range from the current position of the tokens.
      freqs_cis = freqs_cis[start_pos : start_pos + seq_len]
      # Apply RoPE to Queries and Keys embeddings
      xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

      self.cache_k = self.cache_k.to(xq)
      self.cache_v = self.cache_v.to(xq)
      # Store Keys and Values token embedding into their respective cache [KV Cache Implementation]
      self.cache_k[:bsz, start_pos:start_pos + seq_len] = xk
      self.cache_v[:bsz, start_pos:start_pos + seq_len] = xv

      # Assign all the previous tokens embeddings upto current tokens position to Keys and Values variable for Attention Calculation
      keys = self.cache_k[:bsz, :start_pos + seq_len]
      values = self.cache_v[:bsz, :start_pos + seq_len]

      # At this point, they Keys and Values shape aren't same with Queries Embedding which has to be in order to computer attention score
      # Use repeat_kv function to make Keys,Values shape same as queries shape
      keys = repeat_kv(keys, self.n_rep)      #keys[bsz,seq_len,n_heads,head_dim]
      values = repeat_kv(values, self.n_rep)  #values[bsz,seq_len,n_heads,head_dim]

    # Mode - Training mode: KV-Cache not implemented
    else:
      # Compute rotation matrix and apply RoPE to queries and keys for for training.
      freqs_cis = precompute_freqs_cis(dim=self.head_dim, seq_len=self.args.max_seq_len)

      #xq[bsz,seq_len,n_heads, head_dim], xk[bsz,seq_len,n_heads, head_dim]
      xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

      # Use repeat_kv function to make Keys,Values shape same as the queries shape
      #keys[bsz,seq_len,n_heads,head_dim], #values[bsz,seq_len,n_heads,head_dim]
      keys = repeat_kv(xk, self.n_rep)
      values = repeat_kv(xv, self.n_rep)

      # For training mode, we'll compute mask and apply to the attention score later
      mask = torch.full((seq_len, seq_len),float("-inf"),device=self.args.device)
      mask = torch.triu(mask, diagonal=1).to(self.args.device)

    # To compute attention, we'll need to perform a transpose operation to reshape all queries, keys and values bring heads at dim 1 and seq at dim 2
    xq = xq.transpose(1,2)                  #xq[bsz,n_heads,seq_len,head_dim]
    keys = keys.transpose(1,2)              #keys[bsz,n_heads,seq_len,head_dim]
    values = values.transpose(1,2)          #values[bsz,n_heads,seq_len,head_dim]

    # Computing attention score
    scores = torch.matmul(xq, keys.transpose(2,3)).to(self.args.device)/math.sqrt(self.head_dim)
    if mask is not None:
      scores = scores + mask

    # Apply softmax to the attention score
    scores = F.softmax(scores.float(), dim=-1).type_as(xq)
    # Matrix multiplication of attention score with the values
    output = torch.matmul(scores, values).to(self.args.device)

    # We get the contextual embedding for each head
    # All heads need to be reshaped back and combined to give a single single contextual attention output
    # Shape change: output[bsz,n_heads,seq_len,head_dim] -> output[bsz,seq_len, n_heads,head_dim] -> output[bsz,seq_len, n_heads * head_dim]
    output = output.transpose(1,2).contiguous().view(bsz, seq_len, -1)

    # shape: output [bsz,seq_len,dim]
    return self.wo(output)

# If the number of keys/values heads is less than query heads, this function expands the key/values embeddings with the required number of repetition
def repeat_kv(x:torch.Tensor, n_rep: int)-> torch.Tensor:
  bsz, seq_len, n_kv_heads, head_dim = x.shape
  if n_rep == 1:
    return x
  return (
      x[:,:,:,None,:]
      .expand(bsz,seq_len,n_kv_heads,n_rep, head_dim)
      .reshape(bsz,seq_len,n_kv_heads * n_rep, head_dim)
  )


### Test: Repeat_kv function ###
# note: xk, x_norm is already calculated during RoPE, RMSNorm testing and is being used for testing here.
# You need take out the triple quotes below to perform testing
"""
n_rep = ModelArgs.n_heads // ModelArgs.n_kv_heads
keys = repeat_kv(xk, n_rep)
print(f"xk.shape: {xk.shape}")
print(f"keys.shape: {keys.shape}")

## Test: Attention function
# You need take out the triple quotes below to perform testing

attention = Attention(ModelArgs)
x_out = attention(x_norm,start_pos=0, inference=False)
print(f"x_out.shape: {x_out.shape}")
"""
### Test Results: ###
"""
xk.shape: torch.Size([10, 256, 4, 64])
keys.shape: torch.Size([10, 256, 8, 64])
x_out.shape: torch.Size([10, 256, 512])
"""
```

## FeedWard 实现

```python
## Step2e: The Feedfoward Network (SwiGLU activation)
class FeedForward(nn.Module):
  def __init__(self, dim:int, hidden_dim:int, multiple_of:int, ffn_dim_multiplier: Optional[float]):
    super().__init__()
    # Models embedding dimension
    self.dim = dim

    # We must use the hidden dimensions calculation shared by Meta which is the ideal one for this model
    # Hidden dimension are calculated such that it is a multiple of 256.
    hidden_dim = int(2 * hidden_dim/3)
    if ffn_dim_multiplier is not None:
      hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

    # define hiddne layers weights
    self.w1 = nn.Linear(self.dim, hidden_dim, bias=False, device=device)
    self.w2 = nn.Linear(hidden_dim, self.dim, bias=False, device=device)
    self.w3 = nn.Linear(self.dim, hidden_dim, bias=False, device=device)

  def forward(self, x):
    # Shape: [bsz,seq_len,dim]
    return self.w2(F.silu(self.w1(x)) * self.w3(x))



### Test: FeedForward module ###
# note: x_out is already computed at Attention testing and is being used for testing here.
# You need take out the triple quotes below to perform testing
"""
feed_forward = FeedForward(ModelArgs.dim, 4 * ModelArgs.dim, ModelArgs.multiple_of, ModelArgs.ffn_dim_multiplier)
x_out = rms_norm(x_out)
x_out = feed_forward(x_out)
print(f"feed forward output: x_out.shape: {x_out.shape}")
"""

### Test Results: ###
"""
feed forward output: x_out.shape: torch.Size([10, 256, 512])
"""
```

## Decoder Block实现

```python
## Step2f: The Decoder Block. The class name is assigned as TransformerBlock to match the name of Meta llama 3 code base.

class TransformerBlock(nn.Module):
  def __init__(self, args: ModelArgs):
    super().__init__()
    self.args = args
    # Initilizate RMSNorm for attention
    self.attention_norm = RMSNorm(dim=args.dim, eps = args.norm_eps)
    # Initilizate Attention class
    self.attention = Attention(args)
    # Initilizate RMSNorm for feedfoward class
    self.ff_norm = RMSNorm(dim=args.dim, eps = args.norm_eps)
    # Initilizate feedfoward class
    self.feedforward = FeedForward(args.dim, 4 * args.dim, args.multiple_of, args.ffn_dim_multiplier)

  def forward(self, x, start_pos, inference):
    # start_pos = token position for inference mode, inference = True for inference and False for training mode
    # i) pass input embedding to attention_norm and then pass to attention block.
    # ii) the output of attention is then added to embedding(before norm)
    h = x + self.attention(self.attention_norm(x), start_pos, inference)

    # i) pass attention output to ff_norm and then pass to the feedforward network.
    # ii) the output of feedforward network is then added to the attention output(before ff_norm)
    out = h + self.feedforward(self.ff_norm(h))
    # Shape: [bsz,seq_len,dim]
    return out


### Test: TransformerBlock ###
# You need take out the triple quotes below to perform testing
"""
x = torch.randn((ModelArgs.max_batch_size, ModelArgs.max_seq_len, ModelArgs.dim), device=device)
transformer_block = TransformerBlock(ModelArgs)
transformer_block_out = transformer_block(x,start_pos=0, inference=False)
print(f"transformer_block_out.shape: {transformer_block_out.shape}")
"""

### Test Results: ###
"""
transformer_block_out.shape: torch.Size([10, 64, 128])
"""
```
