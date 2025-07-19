---
title: DeepSeek-V3_DeepSeekMoE机制是什么？
date: 2025-02-17
authors: [KenForever1]
categories: 
  - LLM推理
labels: [LLM推理]
pin: true
comments: true
---

<!-- [TOC] -->

今天我们一起来阅读一篇文章[deepseek-v3-explained-2-deepseekmoe](https://medium.com/ai-advances/deepseek-v3-explained-2-deepseekmoe-106cffcc56c1)，文章通过巧妙的例子生动讲解了DeepSeekMoE机制的原理。DeepSeekMoE是DeepSeek模型中的另一个关键架构创新。

将解释 Mixture-of-Experts （MoE） 的**工作原理**，是什么让它在 LLM 中如此受欢迎以及它面临的挑战。我们还将讨论**专家专业化与知识共享之间的权衡**，以及 DeepSeekMoE 如何设计以取得更好的权衡。为了使这些概念更直观，文章通过**餐厅做菜选择厨师**的例子，来类比分解它们，通过厨师在厨房中的角色来说明 MoE 中的每个元素。

<!-- more -->

+ [DeepSeek-V3_MLA注意力机制](https://kenforever1.github.io/blog/%E4%B8%80%E6%96%87%E6%90%9E%E6%87%82deepseek-v3_mla%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6/)

## 背景知识

### LLM 中的 MoE （Mixture-of-Experts）

MoE 来源于[ Adaptive Mixture of Local Experts ](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf)


![Outrageously Large Neural Network 论文中的 MoE layer](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe/01_moe_layer.png)

主要由 三个核心部分 组成：

+ 输入层（Input Layer）：接收数据，并将其编码成向量表示。

+ 专家网络（Experts）：多个子模型，每个专家负责不同的知识领域。

+ 门控网络（Gating Network）：决定输入数据应该由哪些专家来处理，并分配权重。

门控网络 会分析输入数据，给出一个“专家选择”概率分布。例如，一个 MoE 可能有 16 个专家，但每次推理时只会激活其中 2-4 个专家，减少计算量。
选定的专家会对输入数据进行计算，并将结果加权合并后输出。
这种方式让 MoE 既能 灵活调用不同的专家，又能 减少计算开销，使得训练和推理更高效。

在 LLM 的上下文中，MoE 通常是指将 Transformer 模型中的 FFN 层替换为 MoE 层，如下图所示：

![MoE 层图示](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*8dTLPVw1yooXm8hZxlMHfA.png)

上图来自于[GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668)

更具体地说，左侧显示了一叠 N 个 Transformer 层。每个层都有一个 MHA 子层，后跟一个 FFN 子层。相比之下，右侧展示了一叠 N/2 个 Transformer 层，其中下层 Transformer 层中的 FFN 子层已被 MoE 层取代。每个 Transformer 层中的 FFN 子层都将被 MoE 层替换。在实践中，我们可以在 Transformer 层的特定间隔处实现 MoE 以替换 FFN。

如果我们进一步研究 MoE 层，我们会发现它包含一个 Gating operation（门控操作），后跟一堆 FFN，每个 FFN 都与标准 FFN 子层具有相同的架构。这些 FFN 层在 MoE 中被称为 “专家”，门控作经过训练以选择应激活哪个专家来处理特定输入。

![具有门控和多个 FFN 作为专家的 MoE 层](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*Ng79FsDkyfukSn9Il3sgGA.png)



MoE 的一般架构可以更正式地描述如下: 

![](https://miro.medium.com/v2/resize:fit:1236/format:webp/1*_yyTwA8cQH8ai_4QazomYA.png)

[DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/pdf/2401.06066)

+ $ u^l_t $ 和 $ h^l_t $ 是第 $l$ 个 Transformer 层中第 t 个令牌的输入和输出隐藏状态。

+ $ FFN_i $ 是 N 位专家中的第 i 位专家。

+ $ g_{i， t} $ 是给定标记 t 和专家 i 的门控值，通过在 softmax 输出上应用 TopK 操作获得。

+ 等式（5）中的 $ e^l_i $ 通常被称为第 i 个专家的质心，可以通过汇总过去路由到第 i 个专家的所有输入标记来计算：

![](https://miro.medium.com/v2/resize:fit:874/format:webp/1*rLUsOhXpdrtPNK8dyVqGdw.png)

现在让我们以从（5）到（3）的相反顺序逐步解释上述方程：

+ 在方程 （5） 中，$ u^l_t 和 e^l_i $ 之间的内积测量当前输入标记与过去路由到第 i 个专家系统的平均输入标记的接近程度。直观地说，如果专家 i 处理了很多与当前类似的 input token，那么它也应该更擅长处理当前 token。然后，将 Softmax 应用于此 inner-product，将其转换为 distribution。由于我们有 N 个专家，因此每个 token 也将有 N 个 $ s_{i， t} $值。

+ 在方程 （4） 中，我们将 TopK 应用于所有 $ s_{i， t} $ 值，生成稀疏$ g_{i， t} $ 值。

+ 在方程 （3） 中，稀疏$ g_{i， t} $值用于选择 K 个专家来计算输出隐藏状态。

换句话说，N 个专家中只有 K 个会被激活第 t 个标记，因为 K 通常非常小，因此 $ g_{i， t} $值是稀疏的。在这样的设计中，由于额外的 FFN 而增加模型中的可训练参数总数，但只有一小部分参数会在前向传递期间被激活。

这就是为什么我们经常看到带有 MoE 的 LLM 将其模型大小描述为“XX 个总参数, 每个 Token 激活了YY”，其中 YY 远小于 XX，如 DeepSeek-V3 示例所示：

> “它总共包含 236B 个参数，其中 21B 为每个令牌激活......”

那么，如果 MoE 引入更多参数，它有什么好处呢？

###  MoE 的好处和挑战

MoE 的伟大之处在于它反映了许多具有相似原理的现实生活场景，因此我们可以使用这些示例来更直观地理解它。

现在想象一下，我们正在为一家同时供应中国菜和意大利菜的餐厅招聘厨师，我们有两个选择：

+ 选项 1：聘请一位擅长中餐和意大利菜的厨师，这样他或她就可以单独处理每道菜。这类似于标准 Transformer 模型，其中单个 FFN 子层将处理所有输入token。

+ 选项 2：聘请多名厨师，每位厨师都擅长中餐或意大利菜，外加一名主厨，根据他们的专业知识为这两位厨师分配订单。这类似于 MoE 方法，其中每个厨师都充当专家，而主厨充当选择专家的门控机制。

通过上面的类比，很明显，选项 2 不仅使招聘更容易，而且可以确保两种菜肴都以更高的质量准备。相比之下，找到一位精通多种美食的厨师要困难得多, 我们可能不得不在菜肴的质量上妥协。


回到我们的 LLM 场景，MoE 的动机部分与规模假设有关，即在大数据规模上扩展 LLM 时可能会出现涌现能力，这就是为什么我们目睹了当今 LLM 变得越来越大，例如 GPT 模型已经从 117M 扩展到 175B。

然而，并不是每个人都有特权训练如此大规模的 LLM，而 MoE 提供了一个折衷方案：它允许我们扩大模型大小以增加模型容量，同时通过仅激活每个输入token总参数的一小部分来保持训练和推理成本可控。


在deepseek论文中[DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/pdf/2401.06066)
所示，您可以训练仅激活 0.3B 参数的 2B 模型、激活 2.8B 参数的 16B 模型，甚至仅激活 22.2B 参数的 145B 模型。在每种情况下，一次只使用总参数的 1/7 左右，显著提高了训练和推理效率。

然而，每种设计都有其自身的局限性，并带来了新的挑战。在 MoE 的情况下，它的性能在很大程度上取决于门控机制的有效性，因为无法保证它总是将每个输入token路由到最佳专家，并且有可能少数专家经常被激活大多数输入token，而其他专家则坐在那里而没有暴露足够的训练token。这通常被称为 “专家崩溃” 问题。这也会导致其他问题，例如负载不平衡（因为大多数输入token都路由到一小部分专家）和不稳定（当输入token路由到没有接受足够任务培训的专家时，结果会很差）。

因此，在提到MoE的时候，你经常会看到**负载均衡**这个词语。

DeepSeekMoE 还提出了一些负载均衡策略，比如无需辅助损失的负载均衡策略。具体来说，作者采用了 DeepSeek AI 论文 [Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664) 中的负载均衡策略，具体来说，其通过动态更新每个专家的偏置（b）来维持专家的负载均衡，而不会引入额外的干扰梯度。

![](https://raw.githubusercontent.com/KenForever1/CDN/main/moe-loss-free-balance.jpg)

### 知识专业化与知识共享

当我们在上面的餐厅示例中做出招聘决定时，我们也在知识专业化与知识共享(Knowledge Specialization vs. Knowledge Sharing)之间进行权衡：
选项 1 优先考虑通才，但可能会牺牲深度，而选项 2 优先考虑专业化。这种权衡存在于许多组织的实际场景中，例如公司、团队等。

它也存在于 MoE 中，但以更隐晦的方式存在。从理论上讲，每个专家都应该专注于特定方面，因为只有 input token 的一个子集被路由到每个专家，并且所有专家仍然共享一些共同知识，因为它们共享许多参数。与真实的组织不同，很难确定每个专家的专业程度以及他们共享知识的程度。

权衡专业化和知识共享是 MoE 架构的一个关键设计考虑因素，因为过度专业化和过度冗余都不是理想的。

在前一种情况下，拥有过于专业的专家可能会导致训练和推理不稳定，而任何次优的路由都可能导致性能不佳。同时，这通常会导致容量利用率不足，因为高度专业化的专家只能处理一小部分tokens。

在后一种情况下，如果 expert 变得过于相似，MoE 引入的额外参数不会带来成比例的LLM能力上的增益，这肯定是对有限计算资源的浪费。

在下一节中，我们将了解 DeepSeekMoE 如何设计以实现两者的更好权衡。

## DeepSeekMoE 架构

DeepSeekMoE 利用两项关键创新来平衡 MoE 中的知识专业化与知识共享，即**细粒度专家细分（fine-grained expert segmentation）和共享专家隔离（shared expert isolation）**。

![DeepSeekMoE](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*H5Yj8S5ETQDIm8W9vGGQsA.png)

### 细粒度的专家细分

DeepSeekMoE 中提出了细粒度的专家细分，以促进专家专业化，背后的直觉非常简单：随着为输入令牌激活的专家越多，处理该令牌所需的知识更有可能被不同的专家分解和获取。

在我们前面的餐厅示例中，这类似于将每个厨师划分为专业技能，如下图所示。最初，我们由一名厨师准备所有中餐，另一名厨师处理所有意大利菜。在应用精细的专家细分后，将每种菜系所需的技能划分为多个专家，这样我们就会有一组中餐厨师和另一组意大利菜厨师，每个厨师只需要掌握该菜系的特定技能。

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*3cmaI2Dmf2VYKFGNatUbbw.png)

下图也说明了这一点，其中在子图 （a） 中，每个输入令牌被路由到 N 个专家中的 2 个，而在 （b） 中，每个令牌将被路由到 2N 个专家中的 4 个。在更一般的情况下，我们可以将专家数量从 N 增加到 mN，同时将每个专家 FFN 的中间隐藏维度减少到 1/m，并为每个输入token激活 m 倍的专家。这样，（a） 和 （b） 的总体计算成本将大致保持不变。

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*H5Yj8S5ETQDIm8W9vGGQsA.png)

虽然作者没有为这种策略的有效性提供任何理论证明，但他们做了设计实验来验证他们的想法，我们将在评估部分介绍。

### 共享专家隔离

DeepSeekMoE 提出的另一种技术是隔离一些共享的专家以减少冗余。背后的原理是，如果我们用一些共享的专家来学习不同任务中需要的共同知识，这可能会给其他专家更多的自由来摆脱这些共同知识，从而减少这些未共享的专家之间的冗余。简单来说，就是共享专家负责大家都会用到的通识，专门领域的专家负责专有知识。

在我们的餐厅示例中，这类似于将所有厨师进一步分为两组，如下图所示，其中上部显示的第一组处理一般烹饪技能，例如基本刀工、烹饪技术和调味原理，而第二组的厨师更专注于他们自己的专业菜肴。

例如，饺子厨师可以只专注于饺子折叠和蒸，而不需要担心摆盘技术，而意大利面厨师可以只专注于制作更好的意大利面，而不需要学习切碎技术。因此，可以减少厨师之间的知识冗余。

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*wP7dl5-TgRPB8FH2ALqyKg.png)

在图（c） 还显示了如何在 DeepSeekMoE 中实施此策略，其中选择一个专家作为共享专家（以绿色突出显示），因此所有输入token都将路由到该专家，而无需通过 Router。同时，激活的 Specialized Expert 数量从 4 个减少到 3 个，因此激活的 Expert 总数与图 3 （b） 相同。

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*H5Yj8S5ETQDIm8W9vGGQsA.png)

综上所述，下图右侧可以更正式地表述 DeepSeekMoE 架构，将其与以前的通用 MoE 进行比较，以突出差异：

![（左）General MoE vs. （右） DeepSeekMoE](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*OlhdjvDCPyvcww3ZFHtdBA.png)


+ 方程（11）与之前的方程（5）相同

+ 方程 （10） 类似于方程 （4），但这里我们应用 TopK 从 $（mN — K_s）$ 专家中选择 $（mK — K_s）$，其中$ K_s $表示共享专家的数量。

+ 方程 （9） 将方程 （3） 中的第一项分成两个子项，分别对应于共享专家和路由专家。

同样，原始论文中没有理论证明所提出的策略的有效性，但正如我们将在下一节中看到的那样，评估结果确实表明，添加共享专家可以提高绩效并减少知识冗余。

## 评估

正如我们之前提到的，尽管这两种策略背后的直觉听起来都很合理，但作者没有提供任何理论证据来证明它们的合理性，因此不清楚它们是否真的可以帮助解决专业化与知识共享之间的紧张关系，以及它们可以在多大程度上提供帮助。

基本上，我们想要理解三个核心问题：

+ DeepSeekMoE 能取得更好的结果吗？

+ 精细的专家细分是否以及在多大程度上有利于专业化？

+ 共享专家隔离能否以及在多大程度上减少冗余？

为了理解这些问题，作者精心设计了一系列实验，在这里提及它们至关重要。

### DeepSeekMoE 能达到更好的效果吗？

首先，作者研究了他们的方法是否可以带来更好的整体性能。为了验证这一点，他们训练了一系列具有可比性的 总参数/被激活参数 的模型，并在不同的任务中评估它们。他们的主要结果总结在下表中，最佳指标以粗体突出显示。

![整体性能](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*_KjjtZOk1wtb2rckAjIwmQ.png)

几个要点:

+ 蓝色突出显示的列将标准 Transformer （Dense） 与两种 MoE 架构（[Hash Layer](https://arxiv.org/abs/2106.04426) 和[Switch Transformer](https://arxiv.org/abs/2101.03961)）进行比较，表明在可比的激活参数下，MoE 架构可以实现明显更好的性能。

+ 以绿色突出显示的列进一步将 DeepSeekMoE 与另一种 MoE 方法 [GShard](https://arxiv.org/abs/2006.16668) 进行了比较，表明在可比的激活参数下，DeepSeekMoE 实现了明显更好的性能。

然而，获得更好的性能并不一定意味着在专业化与知识共享之间做出更好的权衡，因此我们仍然需要其他实验。

### DeepSeekMoE 对专业化有好处吗？

直接衡量专家的专业化是困难的，相反，作者从相反的方向设计了一个有趣的实验，通过禁用一些排名靠前的专家，看看会发生什么。

直观地说，当专家更专业时，他们应该更难被替换，因此，**禁用路由排名靠前的专家应该对性能产生更大的影响**。

更具体地说，他们通过在 DeepSeekMoE 和 GShard x 1.5 中禁用顶级路由的专家来进行实验，后者作为基线，因为这两种方法在没有专家禁用时具有可比的 **Pile loss**，请参见下图中对应于比率 0 的最左点：

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*YEGo-xTVIlu5-hWzVjXPjQ.png)

> [Pile](https://pile.eleuther.ai/): n 800GB Dataset of Diverse Text for Language Modeling.

随着被禁用的路由专家比例的增加，DeepSeekMoE 始终产生更高的 Pile 损失，这表明 DeepSeekMoE 中的路由专家更加专业，因此更难被其他人取代。

### DeepSeekMoE 是否减少了知识冗余？

按照类似的想法，作者还禁用了共享专家并激活了另一个路由专家，以查看是否可以通过添加其他路由专家来替换共享专家。

结果，他们观察到“Pile loss显着增加，从 1.808 上升到 2.414”，这证实了共享专家获得的知识在某种程度上是独特的，并且路由专家没有接受过足够的培训来涵盖这部分知识。换句话说，路由专家更专业，冗余更少。

## 总结

在本文中，我们通过以餐厅中场景为例子来解释 DeepSeekMoE，这是 DeepSeek-V2 和 DeepSeek-V3 等 DeepSeek 模型中采用的主要架构创新之一。

更具体地说，我们介绍了 MoE 的一般运作方式、它的好处和挑战，以及专家专业化与知识共享之间的权衡。接下来，我们解释了 DeepSeekMoE 中的两个关键要素：细粒度专家细分和共享专家隔离。我们还在评估部分讨论了它的性能。

DeepSeekMoE 可以通过促进专家的专业化来获得更好的结果，其计算成本与一般 MoE 架构相当，从而提高计算效率。

