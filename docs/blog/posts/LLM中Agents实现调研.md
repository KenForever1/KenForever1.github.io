
---
title: LLM中Agents实现调研
date: 2025-02-20
authors: [KenForever1]
categories: 
  - LLM推理
labels: [LLM推理]
comments: true
---


### 最简单的Agent元素构成

根据[Tiny Agent](https://github.com/datawhalechina/tiny-universe/tree/main/content/TinyAgent)实现一个最简单的Agent需要如下必须的元素：

<!-- more -->

+ 一个大模型（可以通过对话的方式交流，提问题，它反馈结果。高级的LLM有思维链，比如DeepSeek R1）
+ 很多个工具类（包括了工具的描述信息、使用参数）
+ 采用大模型和工具类构造的Agent（包括ReAct形式的system prompt，这里采用了React范式的Agent）

大模型的反馈行为被我们的system prompt所约束，一方面是大模型的思考如何解决问题，另一方面，大模型可以利用我们提供的工具描述信息，选择合适的工具解决问题。构造了最终简单的Agent。

构建ReAct形式的system prompt：

```
Answer the following questions as best you can. You have access to the following tools:

google_search: Call this tool to interact with the 谷歌搜索 API. What is the 谷歌搜索 API useful for? 谷歌搜索是一个通用搜索引擎，可用于访问互联网、查询百科知识、了解时事新闻等。 Parameters: [{'name': 'search_query', 'description': '搜索关键词或短语', 'required': True, 'schema': {'type': 'string'}}] Format the arguments as a JSON object.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [google_search]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!
```

整体调用流程：
![](https://github.com/datawhalechina/tiny-universe/blob/main/content/TinyAgent/images/Agent.png)

一次完整的Agent调用从request发起问题，到收到最终的Response。整个过程和LLM交互两次，调用工具函数一次。

+ 第一次解析用户的提问，选择调用的工具和参数；
+ 第二次将工具返回的结果与用户的提问整合。


## Agent的模式React结构

[ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)

> 虽然大型语言模型（LLMs）在语言理解和交互式决策任务中展现出了令人瞩目的能力，但它们的推理能力（例如思维链提示）和行动能力（例如行动计划生成）主要作为独立的主题进行研究。在本文中，我们探索使用 LLMs 以交错的方式生成推理轨迹和特定任务的行动，从而在两者之间实现更大的协同作用：推理轨迹帮助模型推导、跟踪和更新行动计划以及处理异常情况，而行动则允许它与外部资源（如知识库或环境）进行交互以收集额外信息。我们将我们的方法（名为 ReAct）应用于一系列不同的语言和决策任务，并展示了其相对于最先进基线的有效性，以及相对于没有推理或行动组件的方法在人类可解释性和可信度方面的改进。具体而言，在问答（HotpotQA）和事实验证（Fever）方面，ReAct 通过与简单的维基百科 API 交互，克服了思维链推理中普遍存在的幻觉和错误传播问题，并生成了比没有推理轨迹的基线更具可解释性的类人任务解决轨迹。在两个交互式决策基准（ALFWorld 和 WebShop）上，ReAct 分别以 34%和 10%的绝对成功率优于模仿和强化学习方法，同时仅使用一两个上下文示例进行提示。

![](https://raw.githubusercontent.com/KenForever1/CDN/main/react.jpeg)

图1展示了在HotPotQA问答任务和AlfWorld文本游戏中，不同提示方法的对比情况，具体如下：
1. **HotPotQA任务**：比较了标准（standard）、思维链（CoT，仅推理）、仅行动（Act-only）和ReAct（推理+行动）这4种提示方法。在该任务中，模型需要通过推理和信息检索来回答问题，图中省略了提示中的上下文示例，仅展示了模型生成的任务解决轨迹（包括行动、思考过程）以及环境反馈（观察结果）。从图中可以直观地看到不同方法在解决问题时的差异，例如仅行动方法可能在复杂推理需求下难以生成正确的最终行动来完成任务，而ReAct方法通过交错的推理和行动步骤，能够更好地利用外部信息进行推理和决策 。
2. **AlfWorld游戏**：对比了仅行动（Act -only）和ReAct这两种提示方法。AlfWorld游戏要求智能体在模拟的家庭环境中通过文本行动完成目标，同样，图中呈现了两种方法在解决游戏任务时的轨迹。可以发现，仅行动方法可能无法根据环境上下文准确理解和执行任务，出现如在未找到物品时仍重复错误行动的情况；而ReAct方法通过推理步骤，能更好地规划行动顺序、确定物品位置等，提高任务完成的成功率。

图1通过对比不同提示方法在两种不同类型任务中的表现，直观地展示了ReAct方法在结合推理和行动方面的优势，为后续研究和分析提供了可视化的依据。 

## MetaGPT多智能体协作框架

在论文[MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework](https://arxiv.org/abs/2308.00352)中MetaGPT多智能体协作框架引入了SOPs（标准操作程序，Standardized Operating Procedures）。 MetaGPT 元编程框架，将人类工作流程融入基于大语言模型（LLMs）的多智能体协作中，以解决复杂任务。人类在协作实践中形成的标准化操作流程（SOPs）有助于任务分解与协调。

### MetaGPT：引入SOPs

MetaGPT 框架：通过明确角色分工，将复杂任务分解为具体子任务；依据软件开发 SOP 建立工作流程，各智能体顺序协作。采用结构化通信接口和发布 - 订阅机制，提高通信效率并减少信息过载；引入可执行反馈机制，迭代优化代码生成质量。

开源项目地址：[MetaGPT](https://github.com/geekan/MetaGPT/)

> 基于大型语言模型（LLM）的智能体社会在自动问题解决方面取得了显著进展。现有的基于大型语言模型（LLM）的多智能体系统已经能够解决简单的对话任务。然而，由于简单地链接 LLM 所导致的级联幻觉引起的逻辑不一致，更复杂任务的解决方案变得复杂。在这里，我们引入 MetaGPT，这是一个创新的元编程框架，将高效的人类工作流程纳入基于 LLM 的多智能体协作中。MetaGPT 将标准化操作程序（SOPs）编码到提示序列(prompt sequences)中，以实现更精简的工作流程，从而使具有类人领域专业知识的智能体能够验证中间结果并减少错误。
> MetaGPT 采用流水线范式为各种智能体分配不同的角色，有效地将复杂任务分解为涉及许多智能体共同工作的子任务。在协作软件工程基准测试中，MetaGPT 生成的解决方案比以前基于聊天的多智能体系统更加连贯。

多个智能体协作构成societies of agents。“agents” 通常指的是具有一定自主性和决策能力的个体或实体，可以是软件代理、机器人等。“societies of agents” 描述的是由多个这样的智能体组成的社会系统，这些智能体之间可能通过交互、合作、竞争等方式来实现共同的目标或完成特定的任务。

+ 对该框架中的角色专业化、工作流程以及结构化通信进行了解释，并说明了如何在标准化操作流程（SOPs）的背景下组织一个多智能体系统。

+ 介绍了一种可提高角色间通信效率的通信协议。我们还实现了结构化通信接口以及一种有效的发布 - 订阅机制。这些方法使智能体能够从其他角色处获取定向信息，并从环境中获取公开信息。最后。

+ 引入了可执行反馈，这是一种用于在运行时进一步提高代码生成质量的自我修正机制。

通信协议示例（左）以及带有可执行反馈的迭代编程示例（右）。

![](https://raw.githubusercontent.com/KenForever1/CDN/main/sops.jpeg)

左图：智能体使用共享消息池来发布结构化消息。它们还可以根据自身的设定订阅相关消息。右图：在生成初始代码后，工程师智能体运行并检查是否存在错误。如果出现错误，该智能体将检查存储在内存中的过往消息，并将这些消息与产品需求文档（PRD）、系统设计以及代码文件进行对比。

例如：在产品开发例子中，各个Agent(产品经理、架构师、RD开发、QA测试等)相互协作的过程。

![](https://raw.githubusercontent.com/KenForever1/CDN/main/develop_co.jpeg)

MetaGPT 中的智能体通过**文档和图表（结构化输出）**而非对话来进行交流。这些文档包含了所有必要的信息，避免了出现不相关或缺失的内容。


### AFlow自动化框架

在MetaGPT最新的论文[AFlow: Automating Agentic Workflow Generation](https://openreview.net/forum?id=z5uVAKwmjf)中，介绍了一种名为 AFlow 的自动化框架，用于优化由代码表示的工作流，以解决大型语言模型（LLMs）工作流构建中人力成本高、可扩展性和通用性受限的问题。

> 大型语言模型（LLM）在解决不同领域的复杂任务方面展现出了显著的潜力，通常通过采用遵循详细指令和操作序列的智能工作流来实现。然而，构建这些工作流需要大量的人力投入，限制了其可扩展性和通用性。最近的研究试图实现这些工作流的自动生成和优化，但现有方法仍然依赖于初始的手动设置，无法实现完全自动化和有效的工作流生成。为了应对这一挑战，我们将工作流优化重新表述为对代码表示的工作流的搜索问题，其中调用 LLM 的节点通过边连接。我们引入了 AFlow，这是一个自动化框架，它使用蒙特卡洛树搜索有效地探索这个空间，通过代码修改、树状结构经验和执行反馈迭代地优化工作流。在六个基准数据集上的实证评估证明了 AFlow 的有效性，与最先进的基线相比平均提高了 5.7%。此外，AFlow 使较小的模型能够在特定任务上以 GPT-4o 推理成本的 4.55%（以美元计算）超越 GPT-4o。



## smolaents解析

[huggingface smolagents框架](https://huggingface.co/docs/smolagents/index)


## 一些agents框架集合

在项目[awesome-llm-agents](https://github.com/kaushikb11/awesome-llm-agents?tab=readme-ov-file)中收集了一些agents框架。