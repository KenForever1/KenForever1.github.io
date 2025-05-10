---
title: LLM推理：tp和head_num有什么关系
date: 2024-12-09
authors: [KenForever1]
categories: 
  - LLM推理
labels: []
comments: true
---

根据一个报错信息，引入了一个head_num和tensor_para_size的关系。
<!-- more -->

```bash
> /usr/local/lib/python3.10/site-packages/lmdeploy/turbomind/deploy/target_model/base.py(75)__init__()                                           
-> assert self.model_config.head_num % self.tensor_para_size == 0
(Pdb) p self.model_config
ModelConfig(model_name='', chat_template='', model_arch='InternVLChatModel', head_num=64, kv_head_num=8, hidden_units=8192, vocab_size=151674, embedding_size=151674, num_layer=80, inter_size=[29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568, 29568], norm_eps=1e-06, attn_bias=1, start_id=0, end_id=151645, size_per_head=128, group_size=64, weight_type='bfloat16', session_len=32768, tp=6, model_format='hf', expert_num=[], expert_inter_size=0, experts_per_token=0, moe_shared_gate=False, norm_topk_prob=False, routed_scale=1.0, topk_group=1, topk_method='greedy', moe_group_num=1, q_lora_rank=0, kv_lora_rank=0, qk_rope_dim=0, v_head_dim=0, tune_layer_num=1)
```

当我设置tensor_para_size = 6时，在使用lmdeploy部署InternVLChatModel模型时，出现了`AssertionError: assert self.model_config.head_num % self.tensor_para_size == 0`的错误。
也就是说，`self.model_config.head_num`（64）不能被`self.tensor_para_size`整除。

如果使用pytorch backend也会报类似的错误。

使用的lmdeploy版本：
```bash
lmdeploy                      0.7.0.post3
```

模型：
```bash
InternVL2_5-78B-MPO
```

为什么有这个要求呢？

head_num：指的是模型中注意力机制的头的数量。多头注意力机制允许模型在不同的子空间中学习不同的注意力分布。

kv_head_num：指的是键和值（key-value）的头的数量，有时在模型优化时与head_num不同。

tensor_para_size：表示张量并行的大小，即模型在多设备上并行化时的切分数量。

```python
# head_num is divisble by tp but kv_head_num is not
# and tp is divisble by kv_head_num
assert self.model_config.head_num % self.tensor_para_size == 0
self.repeat_kv = 0
if (self.tensor_para_size > self.model_config.kv_head_num
        and self.tensor_para_size % self.model_config.kv_head_num == 0):
    self.repeat_kv = (self.tensor_para_size // self.model_config.kv_head_num)
    self.model_config.kv_head_num = self.tensor_para_size

self.model_config.verify()
assert self.model_config.kv_head_num % self.tensor_para_size == 0
```

在pytorch backend中，也有检查逻辑：
```python
# https://github1s.com/InternLM/lmdeploy/blob/v0.7.1/lmdeploy/pytorch/config.py#L153-L158
# check for tp
assert model_config.num_attention_heads % tp == 0
if model_config.num_key_value_heads >= tp:
    assert model_config.num_key_value_heads % tp == 0
else:
    assert tp % model_config.num_key_value_heads == 0
```

```bash
ModelConfig(hidden_size=8192, num_layers=80, num_attention_heads=64, num_key_value_heads=8, bos_token_id=151643, eos_token_id=151645, head_dim=128, k_head_dim=128, v_head_dim=128, sliding_window=-1, dtype=torch.float16, vocab_size=151674, 
```

tp的值除了和head_num有关，和kv_head_num也有关。

在TP并行场景下，hidden_state输入会经过QKV投影矩阵运算，通过将多头注意力中的head_num维度切分到不同计算设备上。例如当TP_size=8时，每个设备仅处理总头数的1/8部分。经过矩阵运算后，各设备独立处理分片后的Q/K/V矩阵（shape为(bs, head_num/8, seq_len, head_dim)），最后通过AllReduce操作合并各设备的计算结果‌

每个设备处理相同比例的注意力头，避免了显存和算力的不均衡分配。

https://github1s.com/InternLM/lmdeploy/blob/v0.7.1/lmdeploy/turbomind/deploy/module.py中会调用save_split函数，划分。


## InternVL2_5-78B-MPO模型结构：

https://huggingface.co/OpenGVLab/InternVL2_5-78B#quick-start


"ViT-MLP-LLM"架构
ViT：Vision Transformer
MLP：多层感知机
LLM：大型语言模型（ InternLM 2.5 、 Qwen 2.5）

![](https://cdn-uploads.huggingface.co/production/uploads/64119264f0f81eb569e0d569/BiiyXN6NOk0p-3rl3ueyL.png)

与之前的版本一样，我们应用了pixel unshuffle operation，将视觉tokens的数量减少到原来的四分之一。此外，我们采用了与 InternVL 1.5 类似的动态分辨率策略，将图像分割成 448×448 像素的瓦片（tiles）。从 InternVL 2.0 开始的关键区别在于，我们还引入了对多图像和视频数据的支持。


> 上文讲到，FP8 模型之所以无法 TP32 运行，主要因为 DeepSeek R1/V3 模型保存的参数是 FP8 128x128 量化的。Attention 还好，128 个头做 TP16 或者 TP32 都没问题，问题主要出在专家的计算上。
> https://zhuanlan.zhihu.com/p/1895040317134198573