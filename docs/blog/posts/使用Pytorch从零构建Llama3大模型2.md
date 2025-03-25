---
title: 使用Pytorch从零构建Llama3大模型--深入了解输出模块以及训练推理
date: 2025-03-25
authors: [KenForever1]
categories: 
  - llm
labels: []
comments: true
---

朋友们，书接上文，上一篇的Llama3还没有分享完，接着分享输出模块（Output Block）和训练、推理。

## Output Block

最终Decode Block的解码器（decoder）输出将输入到Output Block中。首先，它被输入到 RMSNorm 中。然后，它将被输入到线性层中用于生成logits。

接下来，会发生以下两种操作之一。

<!-- more -->

+ 如果是做推理，则计算 top_p 概率并生成下一个token。如果达到最大生成长度或生成了结束符token，则停止生成。

+ 如果是训练，则使用目标标签计算损失，并重复训练，直到达到指定的最大训练轮数。

通过一个图清晰的展示了训练和推理的不同过程。

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*TSiiz1znMsW0EMUDq2Cx7w.png)

然后把前面的三个大的组件input Block、decoder Block、output Block结合在一起，就构成了Llama3的结构。

看一下代码：
```python
# This is the Llama 3 model. Again, the class name is maintained as Transformer to match with Meta Llama 3 model.

class Transformer(nn.Module):
  def __init__(self, params: ModelArgs):
    super().__init__()
    # set all the ModelArgs in params variable
    self.params = params
    # Initilizate embedding class from the input block
    self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

    # Initialize the decoder block and store it inside the ModuleList. 
    # This is because we've 4 decoder blocks in our Llama 3 model. (Official Llama 3 has 32 blocks)
    self.layers = nn.ModuleList()
    for layer_id in range(params.n_layers):
      self.layers.append(TransformerBlock(args=params))

    # Initilizate RMSNorm for the output block
    self.norm = RMSNorm(params.dim, eps = params.norm_eps)
    
    # Initilizate linear layer at the output block.
    self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

  def forward(self, x, start_pos=0, targets=None):
    
    # start_pos = token position for inference mode, inference = True for inference and False for training mode
    # x is the batch of token_ids generated from the texts or prompts using tokenizers.
    # x[bsz, seq_len] -> h[bsz, seq_len, dim]
    h = self.tok_embeddings(x)

    # If the target is none, Inference mode is activated and set to "True" and "False" if Training mode is activated.
    if targets is None:
      inference = True
    else:
      inference = False

    # The embeddings (h) will then pass though all the decoder blocks.
    for layer in self.layers:
      h = layer(h, start_pos, inference)

    # The output from the final decoder block will feed into the RMSNorm
    h = self.norm(h)

    # After normalized, the embedding h will then feed into the Linear layer. 
    # The main task of the Linear layer is to generate logits that maps the embeddings with the vocabulary size.
    # h[bsz, seq_len, dim] -> logits[bsz, seq_len, vocab_size]
    logits = self.output(h).float()
    loss = None

    # Inference mode is activated if the targets is not available
    if targets is None:
      loss = None
    # Training mode is activated if the targets are available. And Loss will be calculated for further model training. 
    else:
      # 交叉熵计算损失
      loss = F.cross_entropy(logits.view(-1, self.params.vocab_size), targets.view(-1))

    return logits, loss


### Test: Transformer (Llama Model) ###
# You need take out the triple quotes below to perform testing
"""
model = Transformer(ModelArgs).to(ModelArgs.device)
print(model)
"""
```

通过上面的代码构造成了Llama3的结构，如图所示：

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*WG1DxQDPocze_Y0nurNdFQ.png)

## 如何训练Llama3模型

和上面Output Block中的训练图示流程一样，直接看代码更加清晰：

```python
## Step 4: Train Llama 3 Model:

# Create a dataset by encoding the entire tiny_shakespeare data token_ids list using the tokenizer's encode function that we've built at the input block section
dataset = torch.tensor(encode(data), dtype=torch.int).to(ModelArgs.device)
print(f"dataset-shape: {dataset.shape}")

# Define function to generate batches from the given dataset
def get_dataset_batch(data, split, args:ModelArgs):
  seq_len = args.max_seq_len
  batch_size = args.max_batch_size
  device = args.device

  train = data[:int(0.8 * len(data))]
  val = data[int(0.8 * len(data)): int(0.9 * len(data))]
  test = data[int(0.9 * len(data)):]

  batch_data = train
  if split == "val":
    batch_data = val

  if split == "test":
    batch_data = test
  
  # Picking random starting points from the dataset to give random samples for training, validation and testing.
  
  ix = torch.randint(0, len(batch_data) - seq_len - 3, (batch_size,)).to(device)
  x = torch.stack([torch.cat([token_bos, batch_data[i:i+seq_len-1]]) for i in ix]).long().to(device)
  y = torch.stack([torch.cat([batch_data[i+1:i+seq_len], token_eos]) for i in ix]).long().to(device)
  
  return x,y

### Test: get_dataset function ###
"""
xs, ys = get_dataset_batch(dataset, split="train", args=ModelArgs)
print([(decode(xs[i].tolist()), decode(ys[i].tolist())) for i in range(len(xs))])
"""

# Define a evaluate loss function to calculate and store training and validation loss for logging and plotting
@torch.no_grad()
def evaluate_loss(model, args:ModelArgs):
  out = {}
  model.eval()

  for split in ["train", "val"]:
    losses = []
    for _ in range(10):      
      xb, yb = get_dataset_batch(dataset, split, args)
      _, loss = model(x=xb, targets=yb)
      losses.append(loss.item())
    out[split] = np.mean(losses)

  model.train()
  return out

# Define a training function to perform model training
def train(model, optimizer, args:ModelArgs):
    epochs = args.epochs
    log_interval = args.log_interval
    device = args.device
    losses = []   
    start_time = time.time()

    for epoch in range(epochs):
        optimizer.zero_grad()
        
        xs, ys = get_dataset_batch(dataset, 'train', args)
        xs = xs.to(device)
        ys = ys.to(device)
        logits, loss = model(x=xs, targets=ys)
        loss.backward()
        optimizer.step()

        if epoch % log_interval == 0:
            batch_time = time.time() - start_time
            x = evaluate_loss(model, args)
            losses += [x]            
            print(f"Epoch {epoch} | val loss {x['val']:.3f} | Time {batch_time:.3f}")
            start_time = time.time()
    
    # Print the final validation loss
    print("validation loss: ", losses[-1]['val'])
    # Display the interval losses in plot 
    return pd.DataFrame(losses).plot()
```


上面定义了训练函数。让我们使用以下代码块开始训练，并在训练完成后在图中观察训练结果。

```python
## Start training our Llama 3 model
model = Transformer(ModelArgs).to(ModelArgs.device)
optimizer = torch.optim.Adam(model.parameters())

train(model, optimizer, ModelArgs)
```

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*JS08OxC3GYrUiisOiks1TQ.png)

上面的图像显示了训练过程和损失。训练已经进行了 2500 个 epoch。

> 使用带有默认 GPU 和 RAM 设置的 Google Colab 完成训练过程大约需要 10 分钟，这非常快。

在最后一个 epoch 的验证损失为 2.19，考虑到我们正在使用的训练数据量和 epoch 数量，这被认为是可以接受的。为了显著降低损失，需要增加训练数据的大小、增加 epoch 的数量以及提高 GPU 或处理能力。

完成了模型的训练后，为了验证一下效果，就来到了推理过程。通过推理，看看在给定新的输入提示时，模型生成输出文本的效果如何。

## 如何推理Llama3模型

下面定义了generate方法，它会根据输入的prompts，得到output_tokens和output_texts。

```python
## Step 5: Inference Llama 3 Model:
# This function generates text sequences based on provided prompts using the LLama 3 model we've built and trained.

def generate(model, prompts: str, params: ModelArgs, max_gen_len: int=500, temperature: float = 0.6, top_p: float = 0.9):

    # prompt_tokens: List of user input texts or prompts
    # max_gen_len: Maximum length of the generated text sequence.
    # temperature: Temperature value for controlling randomness in sampling. Defaults to 0.6.
    # top_p: Top-p probability threshold for sampling prob output from the logits. Defaults to 0.9.
    # prompt_tokens = [0]
    bsz = 1  #For inferencing, in general user just input one prompt which we'll take it as 1-batch
    prompt_tokens = token_bos.tolist() + encode(prompts)
    assert len(prompt_tokens) <= params.max_seq_len, "prompt token length should be small than max_seq_len"
    total_len = min(len(prompt_tokens)+max_gen_len, params.max_seq_len)   

    # this tokens matrix is to store the input prompts and all the output that is generated by model.
    # later we'll use the tokenizers decode function to decode this token to view results in text format
    tokens = torch.full((bsz,total_len), fill_value=token_pad.item(), dtype=torch.long, device=params.device)

    # fill in the prompt tokens into the token matrix
    tokens[:,:len(prompt_tokens)] = torch.tensor(prompt_tokens, dtype=torch.long, device=params.device)

    #create a prompt_mask_token for later use to identify if the token is a prompt token or a padding token
    # True if it is a prompt token, False if it is a padding token
    input_text_mask = tokens != token_pad.item()

    #now we can start inferencing using one token at a time from the prompt_tokens list starting with the first position.
    prev_pos = 0
    for cur_pos in range(1, total_len):
      with torch.no_grad():
        logits, _ = model(x=tokens[:,prev_pos:cur_pos], start_pos=prev_pos)
      if temperature > 0:      
        probs = torch.softmax(logits[:, -1]/temperature, dim=-1)
        next_token = sample_top_p(probs, top_p)        
      else:
        next_token = torch.argmax(logits[:, -1], dim=-1)        

      next_token = next_token.reshape(-1)

      # only replace the token if it's a padding token
      next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
      tokens[:, cur_pos] = next_token

      prev_pos = cur_pos
      if tokens[:,cur_pos]==token_pad.item() and next_token == token_eos.item():
        break

    output_tokens, output_texts = [], []    

    for i, toks in enumerate(tokens.tolist()):
      # eos_idx = toks.index(token_eos.item())
      if token_eos.item() in toks:
        eos_idx = toks.index(token_eos.item())
        toks = toks[:eos_idx]

      output_tokens.append(toks)
      output_texts.append(decode(toks))
    return output_tokens, output_texts

# Perform top-p (nucleus) sampling on a probability distribution.
# probs (torch.Tensor): Probability distribution tensor derived from the logits.
# p: Probability threshold for top-p sampling.
# According to the paper, Top-p sampling selects the smallest set of tokens whose cumulative probability mass exceeds the threshold p. 
# The distribution is renormalized based on the selected tokens.
def sample_top_p(probs, p):
    probs_sort, prob_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(prob_idx, -1, next_token)    
    # Sampled token indices from the vocabular is returned 
    return next_token
```

然后就可以对提示进行推理并检查生成的输出。

```python
## Perform the inferencing on user input prompts
prompts = "Consider you what services he has done"
output_tokens, output_texts = generate(model, prompts, ModelArgs)
output_texts = output_texts[0].replace("<|begin_of_text|>", "")
print(output_texts)

## Output ##
Consider you what services he has done.

CLARENCE:
Why, I say, let me see the poor of your father,
And all the shepherd's anchor in the cause
Of your honour in the people of the tribunes
Of the world there is a horseman's face,
That worthy stronger in the
```

现在这个模型已经可以根据提示，返回和训练样本学习到的相关输出。如果采用更多的训练轮数、有了更大的训练数据，我们将实现更高的准确性。

在推理部署开源大模型时，通常会采用一些开源框架，比如vllm、lmdeploy、sglang、tensorRT-LLM等框架。这些框架的推理原理和上面的代码一样的，但他们结合了工程实践，实现了优化算法，比如优化了kv-cache，同时采用了tp、pp等对模型进行切分，部署到多卡或者多机上。采用这些框架会推理效率会更高。

前面介绍的组件实现代码篇幅过大，可以访问[build_llama3_from_scratch](https://github.com/tamangmilan/llama3/blob/main/build_llama3_from_scratch.ipynb)。