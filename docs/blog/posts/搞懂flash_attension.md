---
title: 搞懂flash_attention
date: 2025-02-16
authors: [KenForever1]
categories: 
  - LLM推理
labels: [LLM推理]
pin: true
comments: true
---

[TOC]

本文记录了学习flash_attention遇到的一些好的文章，帮助你搞懂flash_attention。

我们知道现在的LLM大模型主流是基于attention搭建的，attention的计算效率也决定了生产场景中大模型的可用性。flash_attention目前有三个版本，分别是flash_attention和flash_attention2和flash_attention3，它们的目的都是采取一系列的优化手段，提高attention的计算效率。

<!-- more -->

根据[Flash Attention原理详解](https://zhuanlan.zhihu.com/p/676655352)，flashattention的核心思想是减少HBM的访问，将QKV切分为小块后放入SRAM中。FlashAttention 优化了显存存取，要搞懂flashattension就要搞懂softmax的优化计算，[手撕online softmax, Flash Attention前传，一撕一个不吱声](https://zhuanlan.zhihu.com/p/5078640012)，[手撕LLM-Flash Attention从softmax说起](https://zhuanlan.zhihu.com/p/663932651)。


```python
X_batch = torch.randn(4, 6)
_, d = X_batch.shape

X_batch_block_0 = X_batch[:, :d//2]
X_batch_block_1 = X_batch[:, d//2:]

# we parallel calculate  different block max & sum
X_batch_0_max, _ = X_batch_block_0.max(dim = 1, keepdim = True)
X_batch_0_sum = torch.exp(X_batch_block_0 - X_batch_0_max).sum(dim = 1, keepdim = True)

X_batch_1_max, _ = X_batch_block_1.max(dim = 1, keepdim = True)
X_batch_1_sum = torch.exp(X_batch_block_1 - X_batch_1_max).sum(dim = 1, keepdim = True)

# online batch block update max & sum
X_batch_1_max_update = torch.maximum(X_batch_0_max, X_batch_1_max) # 逐个元素找最大值
X_batch_1_sum_update = X_batch_0_sum * torch.exp(X_batch_0_max - X_batch_1_max_update) \
                     + torch.exp(X_batch_block_1 - X_batch_1_max_update).sum(dim = 1, keepdim = True) # block sum

X_batch_online_softmax = torch.exp(X_batch - X_batch_1_max_update) / X_batch_1_sum_update
print(X_batch_online_softmax)
```

下面这句为啥和公式不一样呢？实际上是加的下面图中的两个红框。

![](https://raw.githubusercontent.com/KenForever1/CDN/main/online_softmax.png)

```python
X_batch_1_sum_update = X_batch_0_sum * torch.exp(X_batch_0_max - X_batch_1_max_update) \
                     + torch.exp(X_batch_block_1 - X_batch_1_max_update).sum(dim = 1, keepdim = True) # block sum
```

### flash_attention2

[手撕LLM-FlashAttention2只因For循环优化的太美](https://zhuanlan.zhihu.com/p/670085985),Flash Attention 2比Flash Attention 1加速2x, 计算效率达到GEMM性能的50~73%。

+ 减少非乘法计算
+ 优化QKV for循环顺序
+ 采用shared memory减少通信

### flash_attention3

[gpu-mode-cutlass-and-flashattention-3](https://research.colfax-intl.com/gpu-mode-cutlass-and-flashattention-3/)

### 实现Softmax

为了使用CUDA实现Softmax并通过PyTorch调用，需要编写CUDA核函数和C++包装器, 创建两个文件：main.cpp和softmax.cu。然后，在Python中加载并调用CUDA扩展。

详细代码实现参考[KenForever1/online_softmax](https://github.com/KenForever1/cpp_idioms/tree/main/cuda/online_softmax)。

### 实现FlashAttention

[tspeterkim/flash-attention-minimal](https://github.com/tspeterkim/flash-attention-minimal/blob/main/bench.py)。
