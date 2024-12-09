---
title: text-generation-inference推理框架剖析
date: 2024-12-09
authors: [KenForever1]
categories: 
  - LLM推理
labels: []
---


今天要介绍的主题是TGI（text-generation-inference）， 是huggingface开源可用于生产环境的LLM大模型的推理部署服务。

由Router和InferServer构成。Router由Rust实现，InferServer由python端实现。Router相当于一个代理，面向业务会启动一个WebServer，包括对业务请求采用合适的策略进行动态Batch调整，实现大的吞吐和低的延迟。
InferServer对各种LLM大语言模型进行支持，启动模型推理服务。
Router和InferServer之间通过Protobuf定义消息和格式，通过GRPC方式就可以对推理服务进行访问。

介绍可以参考文档[text-generation-inference-doc](https://huggingface.co/docs/text-generation-inference/en/index)。

## TGI优势

+ 通过Open Telemetry进行分布式追踪和Prometheus监控指标。
+ 它支持高级注意力机制，如Flash Attention和Paged Attention，确保推理优化。
+ 该框架还允许通过各种配置和更细粒度的按请求配置（如引导解码以生成结构化输出）来调整服务器。

## LLM推理的两个重要阶段[译]

本小节翻译自，[Anatomy of TGI for LLM Inference](https://medium.com/@martiniglesiasgo/anatomy-of-tgi-for-llm-inference-i-6ac8895d903d)。

当用户提供一段Prompt，进行tokennized得到一组tokens。输入给模型，模型服务在GPU上进行推理。LLM推理依次分为了Prefill阶段和Decode阶段。Prefill阶段对全部Prompt输入tokens进行推理，产生FirstToken（第一个Token）。然后Decode是一个SelfAttenstion的过程，将新生成的Token加到原始tokens中，然后不断的Decode过程，直到生成EOS结束。将得到的tokens转换成语言表示，就是用户得到的回答。

### Prefill

在Prefill阶段，用户输入的Prompt经过tokennized得到tokens，这个过程通过分词将句子转换成更小的单元，这个过程可以参考[模型和分词器](https://transformers.run/c2/2021-12-11-transformers-note-2/)。然后得到的tokens通过一次Forward前向推理，得到FirstToken。
比如，你问大模型，“美国的首都是？”，大模型经过Prefill阶段，返回FirstToken: “华盛顿”。

### Decode

在Decode阶段，大语言模型通过自回归特性不断生成新的token。在这个阶段，基于Prefill预填充阶段的初始标记，模型一次生成一个标记的文本。每个新生成的标记都被添加到输入序列中，为模型处理创建新的上下文。比如，在生成“华盛顿”作为初始标记后，新的序列变为“美国的首都是哪里？华盛顿”。然后使用这个更新后的序列来生成下一个标记。

模型以迭代的方式继续这个过程，每个新标记影响下一个标记的生成。通过这种自回归方法模型即保持上下文，又生成连贯的回应。Decode解码阶段持续进行，直到生成一个序列结束（EOS）标记，或者达到由max_new_tokens指定的最大序列长度。此时，在CPU上对生成的序列进行去分词处理，将标记转换回可读的文本。

### 为什么要分为预填充（Prefill）和解码（Decode）阶段？

这两个阶段的计算特性不同，预填充阶段只需要一个前向传递，而解码阶段涉及多个传递。每个传递都依赖于之前生成的标记。解码阶段的自回归特性导致处理时间更长，计算开销随着总序列长度的增加而呈二次方增长。因此，预填充和解码阶段的分离是必要的。

为了优化这个过程，减少计算开销的二次方增长，采用了一种称为KV缓存的技术。KV缓存保存在预填充和解码阶段每个标记位置生成的中间状态，称为KV缓存。通过将这些KV缓存存储在GPU内存中，模型避免了重复计算，从而减少了计算开销。这种优化提高了解码阶段的效率。

## 动态Batch策略

未完...

## 参考

[Anatomy of TGI for LLM Inference](https://medium.com/@martiniglesiasgo/anatomy-of-tgi-for-llm-inference-i-6ac8895d903d)

[什么是 GPT？Transformer 工作原理的动画展示（2024）](http://arthurchiao.art/blog/visual-intro-to-transformers-zh/)