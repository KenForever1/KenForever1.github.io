---
title: 探索ggml的实现
date: 2025-09-01
authors: [KenForever1]
categories: 
  - cpp
labels: [cpp]
pin: true
comments: true
---

<!-- more -->

本文主要介绍ggml的调试入门环境搭建、ggml的核心数据结构、内存管理与布局、计算图构建与执行、以及gguf文件格式构成，会分成多个小节介绍。

本文起源于作者ggml学习过程中了解的资料，包括[xsxszab的ggml-deep-dive系列文章](https://xsxszab.github.io/posts/ggml-deep-dive-i/)、以及阅读源码，记录和分享自己的理解过程。（感谢xsxszab绘制的关于内存布局的图例，为ggml的理解更加浅显易懂。）

ggml是采用c/c++编写高度优化的tensor张量计算库，没有外部依赖。存在不同的backend实现，从mac的metal、x86的avx实现以及arm的neon指令实现、gpu的cuda、hip、opencl等实现。llama.cpp项目就是使用ggml进行模型加载、推理的。

## 探索ggml的实现--vscode调试环境的搭建

在阅读源码的时候，找一个最简单的example例子跑通，然后跟着源码进行调试，是很快理解原理的一种方法。 

在linux、或者mac下都方便编译。

编译依赖：

+ gcc/g++或者 clang编译
+ cmake： 用于管理构建项目
+ ccache：可选项，加快编译速度用

```bash
git clone https://github.com/ggml-org/ggml.git
```

上面的依赖安装很简单，以ubuntu为例：
```bash
apt update
apt install -y gcc g++ cmake ccache
```

### 配置调试目标

ggml的example中包括了很多例子，先从简单的例子看起。

```bash
examples/simple/simple-ctx.cpp
```

修改项目的cmake配置，即添加-g方便debug调试。找到examples/simple/CMakeLists.txt配置文件，为simple-ctx这个可执行文件添加-g参数。
```bash
set(TEST_TARGET simple-ctx)
add_executable(${TEST_TARGET} simple-ctx.cpp)
target_compile_options(${TEST_TARGET} PRIVATE -g)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)
```
上面只是为可执行文件添加了-g调试信息，如果要调试ggml.so这个库，则需要编译ggml.so库时指定Debug类型。

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=DEBUG
cmake --build build
```
查看编译输出，可以看到ggml.so库被编译成with debug_info。
```bash
file build/src/libggml.so 
build/src/libggml.so: ELF 64-bit LSB shared object, x86-64, version 1 (SYSV), dynamically linked, BuildID[sha1]=db376e303daeef672c8002ed6cebed1da303a706, with debug_info, not stripped
```

### vscode配置debug

点击右边的Run and Debug按钮，create a launch.json file，创建一个配置文件，把下面的内容粘贴进去。

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "debug simple-ctx",
            "type": "lldb",
            "request": "launch",
            "program": "${workspaceFolder}/build/bin/simple-ctx",
            "cwd": "${workspaceFolder}"
        }
    ]
}
```
接下来就可以按照vscode的界面button调试了，如果要调试其它可执行程序，修改launch.json的program字段即可。

可以看到，通过上面的编译debug模式和添加-g参数，可以进入可执行程序和ggml.so库的源码进行调试了了。

![](https://raw.githubusercontent.com/KenForever1/CDN/main/ggml-debug.png)

![](https://raw.githubusercontent.com/KenForever1/CDN/main/ggml-debug1.png)

## 探索ggml的实现--GGML的内存管理

本小节通过最简单的一个例子, ./examples/simple/simple-ctx.cpp，理解GGML实现两个矩阵执行矩阵乘法的核心工作流程。

这个例子具有以下特点，也因此作为我们理解的起点。

+ 在cpu上执行最简单的两个矩阵的乘法计算，与硬件显卡无关

+ 所有的计算都在cpu上执行，所有的内存分配都在RAM中完成

!!![warning]
    GGML采用c语言风格实现，所以在内存管理上，通过struct中的指针和偏移量来管理内存，我们需要跟踪指针值、计算偏移量，来理解它的内存布局。

### 认识ggml_context

跳过不重要的函数（比如： ggml_time_init），断点debug进入load_model函数。首先看到一个ctx_size的计算，通过ctx_size作为参数，调用了ggml_init函数, 初始化了struct ggml_context *。

![](https://raw.githubusercontent.com/KenForever1/CDN/main/ggml-ctx.png)

暂时先跳过复杂的ctx_size计算，重点关注这个GGML最重要的函数之一：ggml_init。

#### 理解ggml_init

ggml_init函数接受一个参数ggml_init_params，这个参数中包含内存分配的参数。

+ mem_size_: 内存池大小，也就是ctx_size（提前计算出来的）

+ mem_buffer_: 内存池，也就是ctx_buffer，这个内存池用于存储ggml_tensor结构体和数据。这里传递的NULL初始化，表示由内部分配

+ no_alloc_: 是否使用用户传递的内存池，如果为true，则使用用户传递的内存池。这个传递的false，表示由内部分配内存池。

```c++
struct ggml_context * ggml_init(struct ggml_init_params params) {
    // ......
}

struct ggml_init_params {
    // memory pool
    size_t mem_size;   // bytes
    void * mem_buffer; // if NULL, memory will be allocated internally
    bool   no_alloc;   // don't allocate memory for the tensor data
};

```

进入ggml_init函数，可以看到，首先在堆上分配了一个struct ggml_context *。

![](https://raw.githubusercontent.com/KenForever1/CDN/main/ggml_init_func.png)

到这里，ggml_context内存布局如下图所示，我们接下来会一步一步了解内部的进一步分配。

![](https://raw.githubusercontent.com/KenForever1/CDN/main/ggml-ctx-graph1.png)

### GGML的tensor表示

在ggml_init函数执行完成后，紧接着就是对两个矩阵tensor的创建和内存赋值。包括了ggml_new_tensor_2d的调用，一直调用到真正的实现函数ggml_new_tensor_impl。

```c++
// create tensors
model.a = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, cols_A, rows_A);
model.b = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, cols_B, rows_B);

memcpy(model.a->data, a, ggml_nbytes(model.a));
memcpy(model.b->data, b, ggml_nbytes(model.b));
```

关于view_src的内容可以跳过，主要用于处理张量视图，现在可以不用关心。

![](https://raw.githubusercontent.com/KenForever1/CDN/main/ggml-new-tensor-impl.png)

首先，一个问题就是：GGML的张量维度是如何表示的？

这里我们可以和pytorch的张量表示进行对比，因为你可以发现，它们的表示方式正好相反。

ggml采用一个四维数组表示shape信息。在pytorch中一个[batch, channels, height, width]的张量，ggml中表示为[width, height, channels, batch]。

pytorch从最外层到最内层的数据，是从左到右表示的。这里最内层，表示在内存存储上连续的维度。

比如row优先存储，一个3 * 4的矩阵（3行，4列），那最内层维度就是列，也就是4列。pytorch表示为[3, 4]，ggml表示为[4, 3]。

![picture from https://en.wikipedia.org/wiki/Row-_and_column-major_order](https://raw.githubusercontent.com/KenForever1/CDN/main/row-colomn-matrix.png)

上面计算出的data_size，先计算单行的size，然后乘以行数，得到矩阵的size。这里的type是float32类型，如果是量化类型，size就会不同，后面用到在讨论。

```c++
size_t obj_alloc_size = 0;

if (view_src == NULL && !ctx->no_alloc) {
    // allocate tensor data in the context's memory pool
    obj_alloc_size = data_size;
}

struct ggml_object * const obj_new = ggml_new_object(ctx, GGML_OBJECT_TYPE_TENSOR, GGML_TENSOR_SIZE + obj_alloc_size);
GGML_ASSERT(obj_new);
```

如果view_src为空，且no_alloc为false，则调用ggml_new_object函数，分配一个struct ggml_object *，并初始化。分配的size就是tensor的size，这里就是data_size。

!!! [warning]
    view_src表示该obj是其它内存的视图，不需要分配内存，直接复用内存。

分配的ggml_object结构体有什么用呢？

#### 理解ggml_new_object

直接看代码，可能有点困难，我们对关键点进行总结。

+ 在这个例子中，我们首先计算了context的size，据此分配了ggml_context。接下来的内存分配，包括obj的分配，都是在context的memory pool中完成。

+ ggml_object的定义可以看出，有一个next指针指向下一个ggml_object。因此它是通过链表来管理内存的，每个ggml_object对象都是一个链表节点。

+ 初始状态下，ggml_object的objects_begin和objects_end都为NULL，表示这个链表为空。

+ ggml_object的用途是什么呢？ggml通过它来隐式的管理各种资源--包括tensor张量、计算图、work buffer等。链接具有O(n)的查找时间复杂度。

![](https://raw.githubusercontent.com/KenForever1/CDN/main/ggml-new-obj.png)

分配一个ggml_object对象，需要多大的内存呢？

```c++
struct ggml_object * const obj_new = ggml_new_object(ctx, GGML_OBJECT_TYPE_TENSOR, GGML_TENSOR_SIZE + obj_alloc_size);

static const size_t GGML_TENSOR_SIZE = sizeof(struct ggml_tensor);
```

以这里ggml_object分配的是tensor类型为例子：
包括了 struct ggml_object的size + struct ggml_tensor的size + obj_alloc_size（也就是tensor的data内存大小）。当然ggml_tensor struc的size和obj_alloc_size还需要进行内存对齐。

到此，我们的ggml_context内存布局（没有分配新的内存，所有的内存都是分配ctx时分配）如下图所示：

![](https://raw.githubusercontent.com/KenForever1/CDN/main/ggml-ctx-graph2.png)


#### ggml_tensor如何定义

上节，我们的第一个tensor，以及通过ggml_object进行了表示管理，并为其在ggml_context中分配了内存，那么ggml_tensor结构体的定义呢？

我们继续看ggml_new_tensor_impl函数中的下半部分内容。

```c++
struct ggml_tensor * const result = (struct ggml_tensor *)((char *)ctx->mem_buffer + obj_new->offs);

*result = (struct ggml_tensor) {
    /*.type         =*/ type,
    /*.buffer       =*/ NULL,
    /*.ne           =*/ { 1, 1, 1, 1 },
    /*.nb           =*/ { 0, 0, 0, 0 },
    /*.op           =*/ GGML_OP_NONE,
    /*.op_params    =*/ { 0 },
    /*.flags        =*/ 0,
    /*.src          =*/ { NULL },
    /*.view_src     =*/ view_src,
    /*.view_offs    =*/ view_offs,
    /*.data         =*/ obj_alloc_size > 0 ? (void *)(result + 1) : data,
    /*.name         =*/ { 0 },
    /*.extra        =*/ NULL,
    /*.padding      =*/ { 0 },
};

for (int i = 0; i < n_dims; i++) {
    result->ne[i] = ne[i];
}

result->nb[0] = ggml_type_size(type);
result->nb[1] = result->nb[0]*(result->ne[0]/ggml_blck_size(type));
for (int i = 2; i < GGML_MAX_DIMS; i++) {
    result->nb[i] = result->nb[i - 1]*result->ne[i - 1];
}

ctx->n_objects++;

return result;
```

result指针指向了ctx->mem_buffer + obj_new->offs，也就是ggml_tensor结构体对象所分配的内存。

![](https://raw.githubusercontent.com/KenForever1/CDN/main/ggml-ctx-graph3.png)

ggml_tensor结构体的关键字段如下：

+ data：指向tensor张量数据存储起始地址，这里就是ggml_tensor结构体自身之后的第一个字节。

+ ne：一个大小为4的数组，表示每个维度的元素数量，这里是[2, 4, 1, 1]。

+ nb：一个大小为4的数组，表示每个维度的元素字节数，这里是[4, 8, 32, 32]。

如果不考虑量化，计算方式如下:

```C++
nb[0] = sizeof(float);
nb[1] = nb[0] * ne[0];
nb[2] = nb[1] * ne[1];
nb[3] = nb[2] * ne[2];
```

通过上面的内容我们以及了解了ggml_new_tensor_2d的工作原理。在simple-ctx例子中，会调用两次以分配两个tensor张量，一次为x，一次为y，分配后的内存布局如下图所示：

![](https://raw.githubusercontent.com/KenForever1/CDN/main/ggml-ctx-graph4.png)

然后通过memcpy将数据复制到ggml_tensor的data字段中。

![](https://raw.githubusercontent.com/KenForever1/CDN/main/ggml-ctx-graph5.png)

这里的张量数据直接硬编码定义到了源代码中，因此不需要加载GGU文件，再更复杂的例子中会通过GGU文件加载数据。

### 要点

+ ggml 通过ggml_context来处理内存分配

+ ggml_context中通过链表来管理内存，每个节点都是ggml_object结构体，隐式管理tensor张量、计算图、work buffer等资源。

+ ggml_tensor的维度表示和pytorch是相反的，这个需要注意。

## 探索ggml的实现--GGML计算图

本小节介绍GGML如何构建和管理计算图相关的数据结构。

### 在ggml_context中创建计算图

上一节，完成了ggml_context的创建，并且完成了矩阵计算所需的ggml_tensor张量的创建。(load_model函数中实现的功能)

```c++
simple_model model;
load_model(model, matrix_A, matrix_B, rows_A, cols_A, rows_B, cols_B);

// perform computation in cpu
struct ggml_tensor * result = compute(model);
```

接下来重点研究的就是compute函数，分为构建计算图、执行图计算、获取结果（获取最后一个计算节点的输出结果）三个步骤。

```c++
// compute with backend
struct ggml_tensor * compute(const simple_model & model) {
    struct ggml_cgraph * gf = build_graph(model);

    int n_threads = 1; // number of threads to perform some operations with multi-threading

    ggml_graph_compute_with_ctx(model.ctx, gf, n_threads);

    // in this case, the output tensor is the last one in the graph
    return ggml_graph_node(gf, -1);
}
```

在build_graph函数中，首先会调用**ggml_new_graph**，这个函数可以和前一节中ggml_new_tensor函数一样的方式理解。它也是创建了ggml_object对象，但是不同的是，**这个object管理的是计算图（ggml_cgraph结构体对象）**，而不是tensor张量。

![](https://raw.githubusercontent.com/KenForever1/CDN/main/ggml-new-graph-custom.png)

+ 通过ggml_graph_nbytes函数获取计算图需要占用的内存大小

+ ggml_new_object根据计算图大小在ggml_context内存区域中分配一块内存，创建ggml_object对象

+ 通过offs偏移获取ggml_object中管理的ggml_cgraph结构体对象指针，完成计算图的构建

现在我们再对细节进行展开介绍：

#### ggml_gral_nbytes计算细节

GGML_DEFAULT_GRAPH_SIZE是一个宏定义，默认值为2048。定义了单个ggml_cgraph中可分配的最大节点树和leaf（叶节点）张量数。然后使用ggml_hash_size函数计算hash表需要的内存大小，乘以2是需要管理nodes和leafs两种类型。

```c++

// 调用ggml_new_graph_custom传递的参数
ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, false);

#define GGML_DEFAULT_GRAPH_SIZE 2048

static size_t ggml_graph_nbytes(size_t size, bool grads) {
    size_t hash_size = ggml_hash_size(size * 2);
    void * p = 0;
    incr_ptr_aligned(&p, sizeof(struct ggml_cgraph), 1);
    incr_ptr_aligned(&p, size * sizeof(struct ggml_tensor *), sizeof(struct ggml_tensor *)); // nodes
    incr_ptr_aligned(&p, size * sizeof(struct ggml_tensor *), sizeof(struct ggml_tensor *)); // leafs
    incr_ptr_aligned(&p, hash_size * sizeof(int32_t), sizeof(int32_t)); // use_counts
    incr_ptr_aligned(&p, hash_size * sizeof(struct ggml_tensor *), sizeof(struct ggml_tensor *)); // hash keys
    if (grads) {
        incr_ptr_aligned(&p, hash_size * sizeof(struct ggml_tensor *), sizeof(struct ggml_tensor *)); // grads
        incr_ptr_aligned(&p, hash_size * sizeof(struct ggml_tensor *), sizeof(struct ggml_tensor *)); // grad_accs
    }
    // 计算hash_size需要多少个ggml_bitset_t表示状态，这些多个ggml_bitset_t构成了bit位图
    incr_ptr_aligned(&p, ggml_bitset_size(hash_size) * sizeof(ggml_bitset_t), sizeof(ggml_bitset_t));

    size_t nbytes = (size_t) p;
    return nbytes;
}
```

ggml_hash_size从其实现来看，通过二分查找找到大于或等于2 * GGML_DEFAULT_GRAPH_SIZE的最小质数，这个质数决定了计算图哈希表的大小。选择质数主要是出于性能考虑：GGML采用了一种简单的开放地址哈希函数，并使用线性探测法。

```c++
// the last 4 bits are always zero due to alignment
Key = (ggml_tensor_pointer_value >> 4) % table_size
```

>>> 使用质数表大小有助于更均匀地分布键，减少聚集、提高查找效率。

ggml_graph的内存布局：

+ ggml_cgraph结构体对象占用空间：sizeof(struct ggml_cgraph)

+ 2048个tensor张量指针，指向nodes

+ 2048个tensor张量指针，指向leafs

+ hash_size个int32_t，用于记录张量的使用次数

+ hash_size个tensor张量指针，用于存储张量的哈希键

+ 梯度相关，simple_ctx例子不涉及

+ 哈希表bit位图，用于记录张量的使用情况

关于bit位图，

```c++
typedef uint32_t ggml_bitset_t;

static_assert(sizeof(ggml_bitset_t) == 4, "bitset_t constants must be updated");
#define BITSET_SHR 5 // log2(sizeof(ggml_bitset_t)*8)
#define BITSET_MASK (sizeof(ggml_bitset_t)*8 - 1)


// >> BITSET_SHR相当于除以32，表示数字n的bit记录位于第几个ggml_bitset_t中
// 一个ggml_bitset_t可以表示32个数的状态，在本文上下文中即表示一个hash位置
static size_t ggml_bitset_size(size_t n) {
    return (n + BITSET_MASK) >> BITSET_SHR;
}

```

根据ggml_graph_nbytes函数计算出来的内存大小，在ctx中分配内存，分配后的内存布局如下：


![](https://raw.githubusercontent.com/KenForever1/CDN/main/ggml_graph_graph1.png)

包括计算图对象、节点指针、叶子指针、使用次数、哈希键、梯度、哈希表bit位图。接下来的几行代码初始化指向已分配内存中不同区域的指针，并将它们存储在ggml_cgraph结构体中。最后，哈希表被重置，所有槽位都被标记为未占用。


![](https://raw.githubusercontent.com/KenForever1/CDN/main/ggml_graph_graph2.png)

### 构建矩阵乘计算图

前面的内容介绍了在ggml_context中分配一个计算图内存，并且初始化了相关成员默认值。

```c++
struct ggml_cgraph  * gf = ggml_new_graph(model.ctx);

// 用gf表示矩阵乘任务
// result = a*b^T
struct ggml_tensor * result = ggml_mul_mat(model.ctx, model.a, model.b);
ggml_build_forward_expand(gf, result);
```

现在这个计算图gf支持添加2048个张量节点和叶节点，但是还没有将矩阵乘这个计算的节点加入计算图。接下来的内容就是介绍如何将矩阵乘任务信息用计算图进行表示。

在ggml_mul_mat函数中，首先检查输入张量的合法性，然后创建一个结果张量，并设置张量的运算类型为矩阵乘。
```c++
struct ggml_tensor * ggml_mul_mat(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_mul_mat(a, b));
    GGML_ASSERT(!ggml_is_transposed(a));

    // 计算结果的shape
    const int64_t ne[4] = { a->ne[1], b->ne[1], b->ne[2], b->ne[3] };
    struct ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    // 将加入graph nodes中的一个node，类型位矩阵乘
    result->op     = GGML_OP_MUL_MAT;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}
```

现在，到了我们构建计算图的最后阶段了，秘密就藏在ggml_build_forward_expand函数中。

函数的输入参数是我们创建的“空图”和矩阵乘任务所返回的矩阵乘结果节点（在复杂的案例中，就是模型的输出节点或者LLM中的logits）。

#### ggml_build_forward_expand 函数细节探索

该函数调用到了ggml_build_forward_impl函数，核心实现在ggml_visit_parents中。

```c++
static void ggml_build_forward_impl(struct ggml_cgraph * cgraph, struct ggml_tensor * tensor, bool expand) {
    const int n0 = cgraph->n_nodes;

    ggml_visit_parents(cgraph, tensor);

    const int n_new = cgraph->n_nodes - n0;
    GGML_PRINT_DEBUG("%s: visited %d new nodes\n", __func__, n_new);

    if (n_new > 0) {
        // the last added node should always be starting point
        GGML_ASSERT(cgraph->nodes[cgraph->n_nodes - 1] == tensor);
    }
}
```

ggml_visit_parents函数通过递归的方式构建了计算图。

![](https://raw.githubusercontent.com/KenForever1/CDN/main/ggml_visit_parents-1.png)

![](https://raw.githubusercontent.com/KenForever1/CDN/main/ggml_visit_parents-2.png)

核心逻辑：

+ 检查当前张量是否已存在于哈希表中。如果存在，则停止执行并返回。

+ 对所有src张量递归调用ggml_visit_parents函数。

+ 如果它是一个叶节点（即常量张量或不由运算生成的输入张量），则将其存储在图的叶数组（leafs数组）中。

+ 否则，将其存储在图的节点数组（nodes数组）中。

所有递归调用返回后，最后一次检查会确保最后记录的节点是结果张量。因为使用了后序遍历，这意味着输入节点（张量）是最后插入的。

采用debug打印gf计算图信息，1个node节点就是mat mul op节点，两个leaf节点就是两个输入张量a、b矩阵。
```c++
> p *gf
(ggml_cgraph) {
  size = 2048
  n_nodes = 1
  n_leafs = 2
  nodes = 0x000055555556ddd8
  grads = nullptr
  grad_accs = nullptr
  leafs = 0x0000555555571dd8
  use_counts = 0x0000555555575dd8
  visited_hash_set = {
    size = 4099
    used = 0x0000555555581e00
    keys = 0x0000555555579de8
  }
  order = GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT
}
```

到目前为止，我们已经构建好了a、b矩阵乘这个任务的计算图，接下来就是看GGML如何执行这个计算图并获取计算结果了。