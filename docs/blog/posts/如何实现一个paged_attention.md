---
title: 如何实现paged_attention--基于flash-attention的PagedAttention内核实现缓存管理器
date: 2025-04-06
authors: [KenForever1]
categories: 
  - llm
labels: []
comments: true
---

类似linux操作系统管理内存的机制，paged_attention用于管理LLM推理时kv cache的显存分配，通过页表机制，优化显存分配，减少碎片。

<!-- more -->

## paged_attention介绍

传统上，请求的键值缓存有以下两点：

+ 存储在连续的内存空间中；
+ 预先分配最大上下文长度的内存（对于 Llama3 为 8192）。

这会导致严重的内存碎片，例如，如果一个请求的实际长度被生成为 792 个标记，那么大约 90%（=7400/8192）的预分配内存会被碎片化，即无法被其他任何请求使用。

为了减少内存碎片并提高请求吞吐量（批量大小），分页注意力（PagedAttention）提供了一种非连续的键值缓存内存管理方案，大致遵循操作系统分页。这确保了内存碎片仅在每个请求的最后分配块中发生：在下面的图表中，用红色勾勒出的部分，请求 A 在物理块 3 中有 3 个tokens，请求 B 在物理块 2 中有 2 个tokens。

![](https://raw.githubusercontent.com/KenForever1/CDN/main/pagedattention.png)

从代码上看attention和paged_attention的区别：
```python
# attention
y = attn(k_cache=k_cache, v_cache=v_cache, ...)
# paged_attention
y = paged_attn(k_cache=k_cache_paged, v_cache=v_cache_paged, block_table=block_table, ...)
```

与k_cache不同，k_cache_paged是非连续的，并且由所有请求共享。物理块 0~8 可以分配给任何请求，这就是为什么我们传入block_table，它包含每个请求对逻辑块到物理块的分配。例如，在上面的图表中，block_table看起来像{0: [7,1,3], 1: [5,2]}（0 和 1 分别是请求 A 和 B 的索引）。

## 基于flash-attention实现缓存管理器

万丈高楼拔地起，我们可以基于现有的基础架构，比如基于flash-attention的PagedAttention内核实现缓存管理器，也可以从零开始搭建。

今天介绍的是基于[Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)，它采用了flash-attention的PagedAttention内核实现。用户只需要实现缓存管理器。它与缓存管理器一起使用（例如在 vLLM 中），该缓存管理器管理何时分配和释放块以及构建块表。缓存管理器的实现取决于你如何构建推理引擎，因此flash-attention没有实现这样的缓存管理器。

用户实现实现一个缓存管理器，该缓存管理器管理何时分配和释放块以及构建块表，也就是下面代码中的block_table，然后传递block_table给flash_attention。

```python
from flash_attn import flash_attn_with_kvcache

y = flash_attn_with_kvcache(q, k_cache_paged, v_cache_paged, k, v, cache_seqlens=cache_seqlens, block_table=block_table, causal=True)
```
### flash_attn_with_kvcache介绍

```python
def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens: Optional[Union[(int, torch.Tensor)]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0, # 0.0 means deactivated
    rotary_interleaved=True,
    alibi_slopes=None,
    num_splits=0,
    return_softmax_lse=False,
):
```

如果 k 和 v 不为 None，k_cache 和 v_cache 将被原地更新为来自 k 和 v 的新值。这对于decoding很有用：你可以传入上一步的缓存键/值，并使用当前步的新键/值进行更新，然后使用更新后的缓存进行注意力计算，所有这些都在一个内核中完成。

如果你传入 k / v，你必须确保缓存足够大以容纳新值。例如，KV 缓存可以预先分配最大序列长度(max_seq_len)，并且你可以使用 cache_seqlens 来跟踪批处理中每个序列的当前序列长度。

如果你想详细了解flash_attn_with_kvcache，请参考[flash-attention接口说明](https://github.com/Dao-AILab/flash-attention/blob/478ee666cccbd1b8f63648633003059a8dc6827d/flash_attn/flash_attn_interface.py#L1492)。


### 实现CacheManager

实现详细代码参考: [tspeterkim/paged-attention-minimal](https://github.com/tspeterkim/paged-attention-minimal?tab=readme-ov-file)

实现CacheManager类，用于管理缓存。Flash Attention目前支持的块大小为256。

```python
block_size = 256
class CacheManager:
    def __init__(self, tokens, block_size=block_size, batch_size=bsz, n_kv_heads=n_kv_heads, head_dim=head_dim):
        self.block_size = block_size
        self.batch_size = bsz
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.num_blocks = (max_seq_len // block_size) * 5 # TODO: make this dynamic
        # [batch_id, (index, filled_positions)]
        self.block_table = {i: [] for i in range(batch_size)}
        self.free_blocks = set(range(self.num_blocks))

        self.k_cache_paged = torch.randn(self.num_blocks, block_size, n_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
        self.v_cache_paged = torch.randn(self.num_blocks, block_size, n_kv_heads, head_dim, device=device, dtype=torch.bfloat16)

        seq_lens = (tokens != -1).sum(1)
        for i, t in enumerate(seq_lens.tolist()): 
            num_blocks_to_reserve = math.ceil(t / block_size)
            num_filled_positions = t % block_size
            for b in range(num_blocks_to_reserve):
                index = self.get_free_block()
                if b == num_blocks_to_reserve-1:
                    self.block_table[i].append((index, num_filled_positions))
                else:
                    self.block_table[i].append((index, block_size))

    # Returns a free block to allocate more tokens to.
    # For simplicity, I raise an error when we run out of free blocks.
    # In the actual implementation, it solves this through scheduling and preemption (see paper)
    def get_free_block(self):
        if len(self.free_blocks) == 0:
            raise Exception('No more free blocks. Implement scheduling and preemption.')
        index = random.choice(list(self.free_blocks))
        self.free_blocks.remove(index)
        return index

    # Gets the logical block table that PagedAttention uses
    # TODO: Serial computation makes it slow. Is there a faster way?
    # 将block_table转换为tensor，用于PagedAttention的输入
    def get_block_table(self):
        max_len = max(len(b) for b in self.block_table.values())
        block_table = [[-1] * max_len for _ in range(self.batch_size)]
        # i is batch index, j is block index
        for i, b in self.block_table.items():
            for j, (index, _) in enumerate(b):
                block_table[i][j] = index
        return torch.tensor(block_table, dtype=torch.int32, device=device)

    def get_kv_cache(self):
        return self.k_cache_paged, self.v_cache_paged

    # Specific to my KV implementation. Returns the last sequence position given the block table.
    def get_last_pos(self):
        last_pos = [(len(b)-1)*self.block_size + b[len(b)-1][1]-1 for b in self.block_table.values()]
        return torch.tensor(last_pos, dtype=torch.int32, device=device)

    # Frees request's blocks.
    # Here, I leave one block, and free the rest. This is a limitation imposed by my kv cache implementation.
    # TODO: Avoid this limitation.
    def free_memory(self, index):
        blocks = self.block_table[index]
        if len(blocks) == 1:
            return
        for i, _ in blocks[1:]:
            self.free_blocks.add(i)
        self.block_table[index] = blocks[:1]

    # Updates block table and filled positions.
    # TODO: Again, pretty slow. Faster parallel way?
    def update(self, eos_reached, input_text_mask):
        for i, (eos, is_prompt) in enumerate(zip(eos_reached, input_text_mask)):
            if is_prompt: # if the token is part of the original prompt, we skip
                continue
            if eos: # free the request's blocks since we have generated the complete answer
                self.free_memory(i)
                continue

            old_index, n = self.block_table[i][-1]
            if n == self.block_size: # allocate new block if necessary
                new_index = self.get_free_block()
                self.block_table[i].append((new_index, 1))
            else: # otherwise, just use the next available slot in the block
                self.block_table[i][-1] = (old_index, n+1)

    def get_fragmented_memory_size(self):
        size = 0
        for b in self.block_table.values():
            _, filled = b[-1] # only the last block has fragmentation
            size += (self.block_size - filled) * n_kv_heads * head_dim * 2 * 2
        return size

# Create CacheManagers for each layer
# 为每层创建CacheManager
cms = [CacheManager(tokens) for _ in range(n_layers)]
```
如何使用这个缓存管理器来执行paged_attention操作？

在forward函数中，我们需要在每个层上执行paged_attention操作。
```python
def forward(tokens, start_pos):
    bsz, T = tokens.shape
    final_embedding = embedding_layer(tokens)
    freqs_cis = freqs_cis_max[start_pos:start_pos+T, :]

    for layer in range(n_layers):
        q_layer = model[f'layers.{layer}.attention.wq.weight']
        k_layer = model[f'layers.{layer}.attention.wk.weight']
        v_layer = model[f'layers.{layer}.attention.wv.weight']
        w_layer = model[f'layers.{layer}.attention.wo.weight']
        ......

        # 调用该层的CacheManager以获取block_table和kv_cache
        block_table = cms[layer].get_block_table()
        # print k_cache_paged.shape: torch.Size([160, 256, 8, 64]), 为(self.num_blocks, block_size, n_kv_heads, head_dim)
        k_cache_paged, v_cache_paged = cms[layer].get_kv_cache()
        cache_seqlens = torch.where(eos_reached, cms[layer].get_last_pos(), torch.tensor([start_pos]*bsz, dtype=torch.int32, device=device))
        # 执行paged_attention
        y = flash_attn_with_kvcache(q, k_cache_paged, v_cache_paged, k, v, cache_seqlens=cache_seqlens, block_table=block_table, causal=True)

        # (Pdb) p tokens.shape
        # torch.Size([1, 44])
        # (Pdb) p y.shape
        # torch.Size([1, 44, 32, 64])
        # (Pdb) p q.shape
        # torch.Size([1, 44, 32, 64])
        # (Pdb) p k.shape
        # torch.Size([1, 44, 8, 64])
        # (Pdb) p v.shape
        # torch.Size([1, 44, 8, 64])

        # 从paged_attention中获取的结果, 输入到下一层计算
        stacked_qkv_attention = y.view(bsz, T, dim)

        # (Pdb) p stacked_qkv_attention.shape
        # torch.Size([1, 44, 2048])

        embedding_delta = torch.matmul(stacked_qkv_attention, w_layer.T)
        ......
```

在decode过程中，调用forward函数，并且调用cms[layer].update(eos_reached, input_text_mask)来更新CacheManager的block_table和free_memory。

```python
# Do inference
for cur_pos in range(min_prompt_len, max_seq_len):
    next_token = forward(tokens[:,prev_pos:cur_pos], prev_pos)
    next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
    tokens[:, cur_pos] = next_token
    
    pdb.set_trace()

    # Update CacheManagers. Increment filled positions + allocate new block if required.
    for layer in range(n_layers):
        cms[layer].update(eos_reached.tolist(), input_text_mask[:, cur_pos].tolist())

    eos_reached |= (~input_text_mask[:, cur_pos]) & (torch.isin(next_token, stop_tokens))
    prev_pos = cur_pos

    if all(eos_reached):
        break
```


!!! note
    如果你想调试python文件，可以通过在代码中添加`import pdb; pdb.set_trace()`来设置断点。然后执行python文件时，程序会暂停在断点处。

## 参考
https://github.com/tspeterkim/paged-attention-minimal?tab=readme-ov-file