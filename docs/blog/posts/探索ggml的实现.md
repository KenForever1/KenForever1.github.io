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