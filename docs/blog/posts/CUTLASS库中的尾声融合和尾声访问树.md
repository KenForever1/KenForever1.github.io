---
title: CUTLASS库中的尾声融合(Epilogue Fusion)和Epilogue Visitor Trees
date: 2025-02-22
authors: [KenForever1]
categories: 
  - LLM推理
labels: [LLM推理]
pin: true
comments: true
---

[TOC]

CUTLASS库中的尾声融合(Epilogue Fusion)和尾声访问树(Epilogue Visitor Trees)

GEMM 在 NVIDIA GPU 上的高性能实现分为两个阶段：mainloop和epilogue。

+ mainloop负责实际 GEMM 计算的部分

+ 其中进行了后处理（例如，元素激活、缩放）和数据存储

这篇文章中，我们将研究 CUTLASS 的尾声融合(epilogue fusion)方案。EVT在论文[Epilogue Visitor Tree (EVT)](https://dl.acm.org/doi/pdf/10.1145/3620666.3651369)中。

<!-- more -->

首先，概述了尾声阶段和 EVT。然后将展示如何使用 CUTLASS 定义的 EVT 和手动构建的 EVT 将简单的 EVT 添加到 CUTLASS GEMM 内核中。然后，给出了一个为新颖用例开发 EVT 的扩展示例，该示例介绍了一些更高级的工具：归约操作和拓扑访问者。[代码示例](https://github.com/ColfaxResearch/cfx-article-src/blob/master/evt/README.md)


## 尾声阶段和 EVT

在内核中，尾声阶段在主循环阶段之后，处理输出张量的后处理。在最普通的情况下，这个阶段只是将矩阵乘积存储到全局内存（GMEM）中。然而，许多人工智能工作负载需要对输出进行额外的处理：添加偏置项、计算像 GELU 这样的逐元素激活函数，或者应用更复杂的归约类型函数，如层归一化或均方根归一化。这些计算可能还需要加载额外的数据，例如在应用残差连接或使用一组真实标签计算损失时。将这些操作合并(或融合)到 GEMM 内核的尾声中通常是有益的。融合后的内核比使用额外的内核来处理后处理有几个优势。

+ 共享内存（SMEM）中通用矩阵乘（GEMM）的输出数据可以在融合内核中立即进行后处理，而单独的内核需要额外的全局内存（GMEM）到共享内存（SMEM）的传输。

+ 在融合内核中，当矩阵乘法（GEMM）结果仍在寄存器中时，可能会应用一些后处理操作。

+ 额外的内核启动会产生额外的延迟和开销。

这种在 GEMM 主循环和内核出口之间合并额外处理的过程称为**尾声融合（epilogue fusion）**。

在实现尾声融合时的一个难点是有许多类型的操作需要融合。尾声可能包含基本上任意的计算序列，并且可能需要内核加载或存储额外的数据。为每个不同的尾声模式编写融合内核会迅速导致内核数量不可管理地激增。此外，程序员可能想要尝试新颖的尾声，对于一个正确融合的尾声，这通常需要对内核代码进行大量更改。为了解决这个问题，CUTLASS 使用一种称为**访问者模式的设计模式**。


在这种模式下，各种类型的尾声在专门的尾声访问者对象中实现。CUTLASS GEMM 内核被设计为接受任意的尾声访问者对象来处理输出数据。然后，尾声访问者将访问输出数据并进行处理。使用这种模型，添加新的尾声只需要创建一个新的专门访问者类，并将其与当前访问者进行交换。

由于尾声可能涉及复杂的操作序列，因此尾声访问者必须是可组合的。尾声访问者树（EVT）是组织成树状结构的访问者集合，它们共同作为一个单独的访问者进行操作。树中的每个叶节点代表一个基本操作，例如加法、乘法、加载或存储。非叶节点通常是树访问者（稍后我们将讨论一个例外情况）。当树访问者访问数据时，它递归地将任务委托给其子节点，并将子节点的输出作为其自身操作的输入。树的根节点的输出最终存储到 GMEM。计算的一个基本示例如图 1所示。

$$
\mathrm{ReLU}(\alpha+\mathbf{AB}+++\beta+\mathbf{C})
$$

![图 1.尾声访客树的一个简单示例。每个树访问者都由一个操作（红色）和一组子节点组成，这些子节点本身可能是树访问者，也可能获取矩阵图块（绿色）或标量（蓝色）。树的输出是其根节点的输出。](https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/10/image-9.png?resize=1536%2C1377&ssl=1)


尾声访客树抽象由 CUTLASS 以两种方式支持。首先，常见的尾声有预先构建的带有用户友好别名的访客树。其次，开发人员可以为定制的尾声编写自己的访客树。然后，CUTLASS 将从提供的树生成融合内核。我们将通过这两种方法的简单示例进行讲解，然后讨论如何创建更复杂的树。

## 使用尾声（Epilogue）和 EVT

在本文中，我们将重点关注 CUTLASS 3.X 版本中针对 EVT 的语法，该语法目前仅支持 NVIDIA Hopper™架构，并且仅适用于 warp 专用内核。对于较旧的版本，请使用 2.X 语法中的访问器——请参阅[cutlass/epilogue/threadblock/fusion/visitor_2x.hpp](https://github.com/NVIDIA/cutlass/blob/cc3c29a81a140f7b97045718fb88eb0664c37bd7/include/cutlass/epilogue/threadblock/fusion/visitor_2x.hpp)以及示例 35以了解其用法。

在 CUTLASS 3.X API 中构建内核的基本方法是基于CollectiveMainloop和CollectiveEpilogue。

```c++
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int,int>, // ProblemShape [M,N,K,L]
    CollectiveMainloop,
    CollectiveEpilogue
>;
```

CUTLASS 提供了多种不同的方法来创建一个CollectiveEpilogue，我们将按照复杂性递增的顺序进行介绍。

### DefaultEpilogue

对于许多仅使用逐元素运算符的常见尾声，实现尾声融合的最短路径是DefaultEpilogue。可以如下定义一个CollectiveEpilogue。

```c++
using CollectiveEpilogue = cutlass::epilogue::collective::DefaultEpilogue<
    cutlass::gemm::TagToStrideC_t<LayoutC>,
    cutlass::gemm::TagToStrideC_t<LayoutC>,
    cutlass::epilogue::thread::LinearCombination<ElementC, 1, ElementAccumulator, ElementAccumulator>>;
```

在include/cutlass/epilogue/thread中还包括LinearCombinationReLU等更多的运算符可以使用。

DefaultEpilogue不使用访问者树。相反，它只是循环遍历输出片段（数据）并应用指定的操作。所以它不是为复杂的尾声而设计的。


### 内置EVTs

如果您需要更复杂的内容，那么您将需要使用 EVT。CUTLASS 提供了各种使用 EVT 构建的常见操作，可以在 [include/cutlass/epilogue/fusion/operations.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/epilogue/fusion/operations.hpp) 中找到。


要使用内置的尾声，我们需要使用CollectiveBuilder。

```c++ hl_lines="1 2 3 13"
using EVTOp = cutlass::epilogue::fusion::LinCombEltAct<
  cutlass::epilogue::thread::ReLU,
  ElementD, ElementCompute, ElementC, ElementScalar>;
 
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      Shape<_128,_128,_64>, Shape<_1,_1,_1>, // grid and cluster shapes
      cutlass::epilogue::collective::EpilogueTileAuto, // automatically compute epilogue tile size
      ElementAccumulator, ElementCompute, // dtypes
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      EpilogueScheduleType, // need TMA warp-specialized to use EVT
      EVTOp
    >::CollectiveOp;
```

上述代码示例使用 EVT 实现了带 ReLU 激活的LinearCombination。对于EVTOp，我们从cutlass::epilogue::fusion中选择了合适的操作。模板参数当然取决于所讨论的操作，因此请参考operations.hpp以获取有关特定操作的更多详细信息。对于我们的LinCombEltAct示例，第一个参数是激活函数（有关更多选项，请参阅cutlass/epilogue/thread/activation.h），其余的是输入和输出的数据类型以及用于累加的数据类型。

此尾声需要额外的参数，标量alpha和beta。对于使用 CollectiveBuilder 构建的 GEMM，在初始化内核时，可以将这些参数与内核的其余参数一起指定。内核的参数如下所示。

```c++
typename Gemm::Arguments arguments {
    cutlass::gemm::GemmUniversalMode::kGemm, // GEMM mode (batched, grouped, // etc.)
    problem_size,
    {block_A.get(), stride_A,                // pointers and strides for mainloop
      block_B.get(), stride_B},
    {{},                   // arguments.epilogue.thread, modified below
      block_C.get(), stride_C,                // pointers and strides for epilogue
      block_D.get(), stride_D},
    hw_info                                  // hardware info
};
```

对 EVT 的参数可在arguments.epilogue.thread中找到。对于内置的 EVT，这是一个由方便命名的参数组成的扁平结构，这样我们就可以写：

```c++
arguments.epilogue.thread.alpha = alpha;
arguments.epilogue.thread.beta = beta;
Gemm gemm;
gemm.initialize(arguments, workspace_ptr);
// workspace_ptr points to additional GMEM workspace, allocated elsewhere
```

### 对 EVT 的结构进行拆解分析

如果内置操作都不符合你的需求，那么你需要通过自己构建访问者树来创建自定义 EVT。为了讨论这个过程，我们将看看内置的LinCombEltAct是如何构建的，因为这些内置操作是使用与创建自定义 EVT 相同的构建块创建的。

```c++
using Sm90LinearCombination = 
  Sm90EVT<Sm90Compute<homogeneous_multiply_add, ElementOutput, ElementCompute, RoundStyle>, // beta * C + (alpha * acc)
    Sm90ScalarBroadcast<ElementScalar>, // beta
    Sm90SrcFetch<ElementSource>, // C
    Sm90EVT<Sm90Compute<multiplies, ElementCompute, ElementCompute, RoundStyle>, // alpha * acc
      Sm90ScalarBroadcast<ElementScalar>, // alpha
      Sm90AccFetch // acc
    >
  >;
 
using Sm90LinCombEltAct =
  Sm90EVT<Sm90Compute<ActivationFn, ElementOutput, ElementCompute, RoundStyle>, // activation(beta * C + (alpha * acc))
    Sm90LinearCombination<ElementCompute, ElementCompute, ElementSource, ElementScalar, RoundStyle> // beta * C + (alpha * acc)
  >;
```

CUTLASS 访问者树的核心是Sm90EVT，它是Sm90TreeVisitor的别名。这个类代表树中的非叶节点。第一个参数是与此节点关联的操作，而后面的所有参数都是子节点。模板参数允许任意数量的节点——例如，Sm90LinCombEltAct中的激活函数接受一个节点，而Sm90LinearCombination中的融合乘法加法操作接受三个节点。

Sm90Compute是一个节点操作，它将一个节点定义为计算节点。第一个模板参数是一个逐元素操作（例如 ReLU、FMA），其他参数确定所使用的数据类型和浮点舍入方式。

![图 2. 左： Sm90LinCombEltAct的树结构。非叶节点（黑色）是树访问者，即Sm90EVT节点。右：计算的另一种视图，它将每个树访问者替换为它执行的操作，并使计算流向下移动。](https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/10/image-11.png?resize=2048%2C1080&ssl=1)


与内置 EVT 一样，我们需要传入参数 alpha 和 beta 来运行 GEMM。然而，对于自定义 EVT，我们不能再使用平面命名参数接口，因为可能存在同一类型节点的多个实例。相反，参数形成一棵反映 EVT 结构的树。

Sm90EVT节点以以下形式获取参数：

```
{first_child_args, ... last_child_args, node_op_args}
```

对于这个树，我们可以这样写:

```c++
arguments.epilogue.thread =
{    // unary op: activation(beta * C + (alpha * acc))
  {    // ternary op (FMA): beta * C + (alpha * acc)
     {{beta}, {beta_ptr}}, // args to Sm90ScalarBroadcast
     {},                   // no args to Sm90SrcFetch (kernel knows about C)
     {                     // binary op : alpha * acc
       {{alpha}, {alpha_ptr}}, // args to Sm90ScalarBroadcast
       {},                     // no args to Sm90AccFetch
       {}                  // op args: multiplies
     },                    // end binary op
     {} // op args: multiply_add
   },   // end ternary op
   activation_args // op args: activation
 };   // end unary op
```
请注意，树访问者节点的node_op_args出现在所有子节点的参数之后——而在Sm90EVT的模板参数中，节点操作出现在子节点之前。因此，操作树和参数树没有相同的结构。两者之间的关系如图 3所示。

![图 3. 左：来自图 2.的 EVT。右：相关Arguments结构体的树。通过将每个树访问者的节点操作移动到末尾来修改树结构。](https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/10/image-7.png?w=1914&ssl=1)


## 更复杂的例子：二元交叉熵损失

让我们开发一个更复杂的、具有实际应用价值且不是由 CUTLASS 预先定义的示例：二元交叉熵损失(binary cross-entropy loss)。作为动机，假设我们正在训练一个机器学习模型来检测图像中的对象。对于提供的每张图像，模型应该标注它是否包含一个人、一只狗、一辆公共汽车等等。给定的图像可能包含任意数量的这些对象，并且有大量的对象需要考虑。在这种被称为极端多标签分类的情况下，评估模型的一种潜在方法是将每个标签视为一个单独的二元分类问题，独立评估模型在每个问题上的性能，并汇总结果。这将引导我们得到以下损失函数

$$
\mathrm{Loss}=-\frac{1}{n}\sum_{i=1}^n\sum_{j=1}^L\left[C_{ij}\log\sigma(f_{ij})+(1-C_{ij})\log(1-\sigma(f_{ij}))\right],
$$

+ n是训练样本的数量

+ L是可能的标签数量

+ $ C_{ij} $ 是真实标签矩阵，其中如果第 i 个样本实际上具有标签 j，$ C_{ij} $ 则等于 1，否则为 0。

+ $ f_{ij} $ 是模型输出的矩阵，因此每个 $ f_{ij} $ 都是一个实数，如果模型更有信心第 i 个示例属于类别 j，则该实数更大。

+ $ \sigma $ 是sigmoid function

![图 4. 二元交叉熵损失的计算图。此图不是树状图，并且包括向量广播（绿色）和归约（黄色）。](https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/10/image-15.png?resize=768%2C850&ssl=1)

这带来了一系列新的复杂情况：

+ 除了标量之外，我们现在还需要广播行向量 $ b^T $。我们可以使用 EVT 节点Sm90RowBroadcast来实现这一点。（同样，对于广播列向量，也有 EVT 节点Sm90ColBroadcast。）

+ 结果必须简化为标量，我们可以使用新的 EVT 节点Sm90ScalarReduction来实现。（也有用于行和列简化的 EVT 节点。）

+ 我们需要加载一个额外的矩阵，即标签矩阵C，理想情况下使用 TMA、管道(pipeline)和 warp-specialization。CUTLASS 的 GEMM 内核期望执行计算 $ D = AB + C $，因此无论如何都会接受一个额外的输入矩阵C，我们可以使用Sm90SrcFetch访问它。如果我们不想这样做，或者如果我们需要加载多个额外的矩阵，我们可以使用Sm90AuxLoad。

+ 该图不再是树：在计算中，$ \sigma(f_{ij}) $ 和 $ C_{ij} $ 都被使用了两次。我们可以通过重新加载或重新计算这些矩阵两次将图转换为树，但这会带来不良的性能成本。这个问题是可以解决的，但它的解决方案更复杂，需要解释一下。


### 拓扑访问者(Topological visitors)


EVTs 是用树表示的计算图。在访问过程中，树被递归地遍历；每个树访问者节点调用其每个子节点的访问方法，并使用其指定的节点操作组合它们的结果。重要的是，每个节点预计只被访问一次。但一般来说，计算图不一定是树，而是有向无环图。实际上，这意味着一个节点的输出可能被多个其他节点需要。

如果我们仍然将这样的图表示为一棵树，仅使用树访问者，那么我们实际上必须复制所需的节点；每个需要输出的父节点都有一个。这种方法效率低下，因为它会导致大量重复工作。相反，我们使用一个称为“拓扑访问者”的节点。虽然树访问者用于表示计算图中的单个操作，但拓扑访问者表示该图的“任何子图”。

拓扑访问者在其子图中的每个节点都有一个子访问者。在访问过程中，它以拓扑顺序将任务委托给其子访问者，用已访问的子访问者的输出填充每个子访问者的输入。这里的“拓扑顺序”意味着在计算图中，任何子节点都不会在其前驱节点之前被访问——换句话说，当访问一个后代节点时，它的所有输入都必须准备好。拓扑访问者的返回值是它访问的最后一个节点的返回值。

![图 5。一个简单的非树有向无环图。右侧是可以遍历此有向无环图的拓扑访问者的代码。](https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/10/image-1.png?w=943&ssl=1)


一个简单的例子如图 5所示。这个计算图有两个节点，1 和 2，它们都需要节点 0 的结果，所以我们应该用拓扑访问器来构建相关的 EVT。节点 0 不需要任何输入，因为它只返回累加器的值。节点 1 和 2 各取一个输入，即节点 0 的输出。节点 3 取两个输入，即节点 1 和节点 2 的输出。最后，拓扑访问器返回节点 3 的输出。

此时，EVT 是一棵树，有一个根（拓扑访问者）和四片叶子（计算图的编号节点）。

图的右侧给出了拓扑访问器的 CUTLASS 语法。第一个模板参数是计算的数据类型。第二个是元组序列。其余的模板参数是被访问的节点（它们本身可以是树或拓扑访问器）。节点按照它们在参数中出现的顺序进行枚举，第一个是节点 0。回到元组，它们显示了节点依赖关系，其中第 N 个元组列出了其输出将用作节点 N 的输入的节点。

总之，拓扑访问者的目的是将非树形有向无环图（DAG）转换为树。这意味着，作为一个经验法则，拓扑访问者只需要访问计算图的非树部分。如图 5 所示，这部分通常是“在分支和合并之间”，从生成多个计算流的地方开始，到它们重新组合的地方结束。

### 使用拓扑访问器构建EVT

使用拓扑访问器，我们可以重用累加器和标签矩阵中的数据，而无需重新加载它。在将树写成 CUTLASS 类型之前，我们还可以进行一些调整。让我们回到之前的损失公式，可以进行调整简化。


$$
\mathrm{Loss}=-\frac{1}{n}\sum_{i=1}^n\sum_{j=1}^L\left[C_{ij}\log\sigma(f_{ij})+(1-C_{ij})\log(1-\sigma(f_{ij}))\right],
$$
$$
\log(1-\sigma(x))=\log\left(1-\frac{1}{1+e^{-x}}\right)=\log\left(\frac{e^{-x}}{1+e^{-x}}\right)=-x+\log\sigma(x)
$$
因此，该公式简化为
$$
\sum_{i=1}^n\sum_{j=1}^L\left[(1-C_{ij})(-f_{ij})+\log\sigma(f_{ij})\right]
$$

+ 它通过消除 C 的重复使用简化了计算图。

+ 从性能角度来看，这只需要对每个项进行一次对数运算，而不是两次，从而减少了对相对低吞吐量的特殊函数单元的负载。

+ 从数值稳定性的角度来看，如果 $ f_{ij} $ 很大（以至于 $ 1 - \sigma(f_{ij}) $ 趋近于0），那么原始公式会趋于下溢。但这个公式不会。

其次，如果 $ -f_{ij} $ 很大（以至于 $ \sigma(f_{ij}) $ 趋近于0），新公式仍然会下溢。有几种方法可以处理这个问题，但最简单的可能是对 $ \sigma(f_{ij}) $ 的输出进行截断，使其永远不会太接近 0。进行这些更改后，我们得到了<图 6>中的计算图。这个图仍然不是一棵树，所以我们必须在相关的 EVT 中使用拓扑访问器。

![图 6. 左：二值交叉熵损失的简化且在数值上更稳定的计算图。右：相关的 EVT，它使用一个拓扑访问器（黑色）遍历编号节点。](https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/10/image-14.png?resize=2048%2C1125&ssl=1)

对于像这样的复杂图形，可以像我们在下面所做的那样，用类型别名缩写 EVT 的部分内容。

```c++ hl_lines="10 30 35 36 37 38"
using CMinus1 =
  Sm90EVT<
    Sm90Compute<cutlass::minus, ElementCompute, ElementCompute, RoundStyle>,
    Sm90SrcFetch<TC>,
    Sm90ScalarBroadcast<ElementScalar>
  >;
using MatmulPlusBias =
  Sm90EVT<
    Sm90Compute<cutlass::plus, ElementCompute, ElementCompute, RoundStyle>,
    Sm90ColBroadcast<0, CtaTileShapeMNK, ElementBias, Stride<_1, _0, _0>>,
    Sm90AccFetch
  >;
using TopoVisitor =
  Sm90TopologicalVisitor<
    ElementCompute,
    cute::tuple<
      cute::seq<>,
      cute::seq<>,
      cute::seq<0, 1>,
      cute::seq<0>,
      cute::seq<3>,
      cute::seq<4>,
      cute::seq<2, 5>,
    >,
    MatmulPlusBias,
    CMinus1,
    Sm90Compute<cutlass::multiplies, ElementCompute, ElementCompute, RoundStyle>,
    Sm90Compute<cutlass::epilogue::thread::Sigmoid, ElementCompute, ElementCompute, RoundStyle>,
    Sm90Compute<cutlass::epilogue::thread::Clamp, ElementCompute, ElementCompute, RoundStyle>,
    Sm90Compute<FastLog, ElementCompute, ElementCompute, RoundStyle>,
    Sm90Compute<cutlass::plus, ElementCompute, ElementCompute, RoundStyle>
  >;
using BCELossEVT =
  Sm90EVT<
    Sm90ScalarReduction<
      cutlass::plus,       // register reduce function
      cutlass::atomic_add, // GMEM reduce function
        ElementScalar, ElementCompute, RoundStyle,
        Stride<_0, _0, _0>>, // no batching here
    TopoVisitor
  >;
```

拓扑访问者的参数是它访问的每个节点的参数列表。整个 EVT 的参数如下：

```c++
BCELossEVT::Arguments args_BCE =
{
  { // TopoVisitor [(C - 1) * (bias + AB) + log(clamp(sigmoid(bias + AB)))]
    { // args to MatmulPlusBias = bias + AB (node 0)
      {d_bias_BCE.data().get(), 0, stride_bias_BCE}, // args to ColBroadcast
      {},  // args to AccFetch
      {}   // op args: plus
    },
    { // args to CMinus1 = C - 1 (node 1)
      {}, // args to SrcFetch
      {{ElementScalar(1.0)}}, // args to ScalarBroadcast
      {}  // op args: minus
    },
    {}, // op args: multiplies (node 2)
    {}, // op args: sigmoid (node 3)
    {0.001f, 0.999f},   // op args: clamp (node 4)
    {}, // op args: log (node 5)
    {}, // op args: plus (node 6)
  },
  {d_result, 0, stride_result} // args to ScalarReduction
};
```

## 图编译及进一步优化

正如这个例子所示，构建 EVT 的过程并非完全简单。理想情况下，人们希望用像 Python 这样的高级语言以数学方式描述尾声，并让一个自动化系统在应用明显优化的同时将其解析为 EVT。EVT 论文的作者将这样的系统称为“深度学习编译器”，并在[论文的 GitHub 仓库](https://github.com/ColfaxResearch/cfx-article-src/blob/master/evt/README.md)中以[torch.fx](https://pytorch.org/docs/stable/fx.html) 的形式实现它。CUTLASS 在其 [Python 接口](https://github.com/NVIDIA/cutlass/blob/main/examples/python/04_epilogue_visitor.ipynb)中提供了一个简单的 Python 到 C++版本。

EVT 编译器算法执行的一些优化应该被任何手动编写尾声访问者树的人考虑。

+ **运算符融合**：用其组合的快速实现来替换一系列运算符。

+ **算子裂变**：将一个算子分解成一个序列，以便在其他地方进行算子融合。

+ **修剪未使用的节点**：也许太明显而无需提及，但该论文指出，在训练机器学习模型时，通常不必计算损失本身——只需计算其梯度！

+ **归约消除**：由于归约需要线程间协作，因此它是一个常见的瓶颈。在某些情况下，可以消除归约操作。作为一个简单的例子，一个one-hot的行和是一个常数向量，其元素全为 1。

## 结论

在本文中，我们对尾声融合和尾声访问者树进行了详细的讨论。我们介绍了尾声融合及其在高性能通用矩阵乘法（GEMM）工作负载中的重要性。然后，我们讨论了尾声访问者树如何提供一种独立于内核主循环本身开发可融合尾声的方法。

接下来，我们展开了 CUTLASS 为尾声融合提供的不同接口：DefaultEpilogue、预构建的 EVT 和自定义 EVT。最后，我们通过为二元交叉熵创建一个 EVT 给出了一个复杂的真实世界示例。这个[示例以及关于各种 CUTLASS EVT 节点](https://github.com/ColfaxResearch/cfx-article-src/tree/master/evt)的补充文档可在我们的 GitHub上找到。



（文章结合作者理解，译自[epilogue_visitor_tree/](https://research.colfax-intl.com/epilogue_visitor_tree/)）