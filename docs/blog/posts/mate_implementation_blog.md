---
title: MATE 实现解析：MUSA 平台上的 GenAI 算子兼容层与高性能内核库
date: 2026-05-15
authors: [KenForever1]
categories: 
  - LLM推理
labels: [LLM推理]
pin: true
comments: true
auto_number_title: false
---

<!-- [TOC] -->


MATE，全称 **MUSA AI Tensor Engine**，是摩尔线程为 MUSA 平台构建的一套生成式 AI 算子库。它的目标不是重新发明一套全新的上层 API，而是尽量兼容 CUDA 生态中已经被广泛使用的接口，例如 `flash_attn_3`、`sageattention`、`flash_mla` 和 `deep_gemm`，同时把底层执行替换为 MUSA 原生 kernel。

<!-- more -->

换句话说，MATE 做的是一件“上层兼容、下层重写”的事情：让上层项目继续使用熟悉的 Python 包名和函数签名，而真正执行时走 MUSA Toolkit、TorchMUSA、MUTLASS、TVM-FFI、JIT/AOT 以及预编译 `mubin` kernel。

## 1. MATE 要解决什么问题

在 CUDA 生态里，LLM 和 Diffusion 推理已经形成了一批事实标准组件：

- FlashAttention / FlashAttention-3：高性能 attention kernel。
- FlashMLA：面向 DeepSeek MLA 架构的高效 decode/prefill kernel。
- DeepGEMM：面向 FP8/BF16 GEMM 的高性能实现。
- SageAttention：面向量化 attention 的高性能接口。

这些库原本大多依赖 CUDA、Triton、CUTLASS 或 NVIDIA 特定硬件能力。如果 SGLang、vLLM 或 Diffusion 框架要迁移到 MUSA 平台，直接替换所有上层调用成本很高。

MATE 的思路是：

1. 保留这些库的 Python import path 和主要 API。
2. 用 wrapper 包把调用转发到 MATE。
3. MATE 内部根据参数选择 MUSA kernel。
4. 对热点 shape 使用预编译 `mubin` 二进制。
5. 对可生成的 kernel 使用 JIT/AOT 编译。
6. 通过 TVM-FFI 把 C++/MUSA kernel 暴露给 Python。

因此，MATE 更像是 MUSA 平台的 GenAI kernel runtime 和兼容层，而不仅仅是一个普通 Python 包。

## 2. 项目总体架构

MATE 的核心目录可以按职责分成几层。

### 2.1 Python 公共 API 层：`mate/`

`mate/` 是核心 Python 包，对外暴露 MATE 的主 API。`mate/__init__.py` 采用懒加载方式暴露关键接口，例如：

- `flash_attn_varlen_func`
- `flash_attn_with_kvcache`
- `get_scheduler_metadata`
- `flash_mla_with_kvcache`
- `get_mla_metadata`
- `sage_attn_quantized`
- `sage_attn_quantized_with_kvcache`
- `gated_delta_rule_decode`
- `deep_gemm`

这一层不直接写 kernel，而是负责参数整理、输出分配、调用日志、backend 选择，以及把请求转发到 JIT/AOT 加载出来的 native module。

### 2.2 Wrapper 兼容层：`wrappers/`

`wrappers/` 下有四个主要兼容包：

- `wrappers/flash-attention`：提供 `flash_attn_3` 和 `flash_attn_interface` 兼容接口。
- `wrappers/SageAttention`：提供 `sageattention` 兼容接口。
- `wrappers/FlashMLA`：提供 `flash_mla` 兼容接口。
- `wrappers/DeepGEMM`：提供 `deep_gemm` 兼容接口。

这些 wrapper 的价值是让上游代码少改甚至不改。例如 SGLang 中如果写的是：

```python
from flash_attn_interface import flash_attn_with_kvcache
```

在 MUSA 环境中，这个 import 可以由 MATE 的 flash-attention wrapper 提供，最终调用进入 MATE 的 MUSA 实现。

这是一种典型的“兼容包名 + 替换后端”的迁移策略。

### 2.3 JIT/AOT 编译层：`mate/jit/`

`mate/jit/` 是 MATE 的 kernel 构建和加载系统。核心抽象是 `JitSpec`：

- 记录 kernel 名称。
- 记录源码文件列表。
- 记录编译参数。
- 记录 include 路径。
- 记录 AOT 路径和 JIT cache 路径。
- 生成 ninja build 文件。
- 调用 ninja 编译 `.so`。
- 用 `tvm_ffi.load_module()` 加载编译产物。

JIT/AOT 层的作用是把 Python 侧的一次高层调用连接到 C++/MUSA 编译产物。

以 SageAttention 为例，`mate/jit/sage_attention.py` 会生成名为 `sage_attention` 的 JIT spec，源码包括：

- `csrc/sage_attention_asm.mu`
- `csrc/mubin/mp31/flash_atten/` 下匹配 SageAttention 的预编译 `.cpp` blob

也就是说，SageAttention 的 native module 不是只编译一个 `.mu` 文件，而是把 dispatcher、TVM-FFI 入口和一批 `mubin` 二进制数组一起编译成一个 `.so`。

### 2.4 C++/MUSA kernel 层：`csrc/` 和 `include/`

`csrc/` 和 `include/` 是 MATE 的 native 实现层。

主要内容包括：

- `.mu` 源码：MUSA kernel 或 host-side launch/dispatch 逻辑。
- `.hpp` 头文件：公共工具、descriptor、注册表、kernel wrapper。
- Jinja 模板：用于生成 FMHA/GEMM 等 kernel 代码。
- `mubin` 二进制：预编译好的 MUSA kernel，以 C++ `unsigned char[]` 形式嵌入。

`include/mate/` 下则有 attention、flash MLA、GEMM 等更底层的模板和封装。

### 2.5 预编译二进制层：`csrc/mubin/mp31/`

`mubin` 是 MATE 中非常关键的一层。它存放 MP31 架构上的预编译 kernel blob，包括：

- `flash_atten/`
- `flash_mla/`
- `gemm/`

这些 `.cpp` 文件通常是自动生成的，例如：

```cpp
// Auto-generated binary mubin file
unsigned char e4m3tce_flash_atten_quant_mode_6_512_256x128x128[] = {
  0x7F, 0x45, 0x4C, 0x46, ...
};
unsigned int e4m3tce_flash_atten_quant_mode_6_512_256x128x128_len = ...;
```

`0x7F 0x45 0x4C 0x46` 是 ELF 魔数，说明这里嵌入的是二进制目标文件。它的作用类似 CUDA 生态里的 `cubin` 或 fatbin：不是源码，而是已经为特定硬件架构编译好的设备端 kernel。

## 3. MATE 的调用链：从 Python 到 MUSA kernel

MATE 的典型调用链可以概括为：

```text
用户 Python API
  -> wrapper 兼容层
  -> mate Python interface
  -> TVM-FFI native module
  -> C++/MUSA dispatcher
  -> registry 查找 mubin blob
  -> muModuleLoadData 加载二进制
  -> muModuleGetFunction 获取 kernel 函数
  -> muLaunchKernelEx 发射 kernel
```

这个链路里最关键的是 TVM-FFI 和 `mubin` registry。

TVM-FFI 负责把 C++ 函数导出给 Python。例如 `csrc/sage_attention_asm.mu` 末尾通过：

```cpp
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sage_attn_quantized_asm, sage_attn_quantized_asm);
```

把 C++ 函数 `sage_attn_quantized_asm` 暴露成 Python 侧可调用的 native function。

`mubin` registry 则负责把运行时参数映射到具体的预编译 kernel。

## 4. 以 SageAttention 为例：wrapper 如何工作

SageAttention 是最容易理解的 wrapper，因为它没有 FlashAttention 那么多 varlen、paged KV cache、scheduler metadata 分支，也没有 DeepGEMM 那么复杂的 GEMM 配置空间。

调用入口是：

```python
from sageattention import sageattn
out = sageattn(q, k, v, tensor_layout="HND")
```

实际链路如下：

```text
sageattention.sageattn
  -> sageattn_qk_int8_pv_fp8_cuda_sm90
  -> _run_quantized_sage_attention
  -> mate.sage_attention_interface.sage_attn_quantized
  -> module.sage_attn_quantized_asm
  -> csrc/sage_attention_asm.mu
  -> mubin registry
  -> muLaunchKernelEx
```

### 4.1 wrapper 层做参数兼容

`wrappers/SageAttention/sageattention/interface.py` 主要负责：

1. 校验输入 tensor 是否为 `float16` 或 `bfloat16`。
2. 校验 Q/K/V 是否在同一 MUSA device。
3. 支持 `HND` 和 `NHD` 两种 tensor layout。
4. 把输入统一转成内部使用的 BNHD layout。
5. 解析 `quant_recipe` 和 `qk_quant_dtype`。
6. 对 Q/K/V 做量化，生成 quantized tensor 和 scale tensor。
7. 调用 MATE 的低层接口 `sage_attn_quantized`。
8. 根据 `return_lse`、`fp8_output` 等参数整理输出形式。

它本身不发射 MUSA kernel，而是做兼容、量化和参数整理。

### 4.2 MATE Python interface 分配输出并调用 native module

`mate/sage_attention_interface.py` 中的 `sage_attn_quantized` 是低层接口。它接受已经量化好的 Q/K/V 和 scale tensor，做以下工作：

1. 检查 backend，只支持 `mubin`。
2. 推导 `softmax_scale`。
3. 根据 `quant_recipe` 得到 ASM quant mode。
4. 分配输出 tensor：`out`、可选 `out_scale`、`lse`。
5. 调用 `module.sage_attn_quantized_asm(...)`。

这里的 `module` 来自 `mate.jit.sage_attention.get_sage_attention_module()`，即 JIT/AOT 加载出来的 TVM-FFI module。

### 4.3 C++ 入口组装 Args

`csrc/sage_attention_asm.mu` 中的 `sage_attn_quantized_asm` 是真正的 C++ 入口。

它会检查：

- 是否是 MP31 设备。
- Q/K/V 最后一维是否 contiguous。
- Q/K/V dtype 是否符合 INT8/FP8/BF16 等路径要求。
- scale tensor shape 是否匹配当前 quant mode。
- output shape 是否正确。

然后把所有运行时信息填进 `SageAttenQuantizedASMArgs`，包括：

- `is_causal`
- `is_kv_cache`
- `is_qk_int8`
- `fp8_output`
- `quant_mode`
- Q/K/V dtype
- batch、seqlen、head 数、head dim
- stride 信息
- Q/K/V/output/scale/lse 的 data pointer
- stream

这些 Args 就是后续选择和启动具体 `mubin` kernel 的依据。

## 5. SageAttention 如何调用到 mubin

这是 MATE 中最有代表性的实现机制。

### 5.1 JIT spec 把 mubin 编进 native module

`mate/jit/sage_attention.py` 会扫描：

```text
csrc/mubin/mp31/flash_atten/
```

找出名字匹配 SageAttention quantized attention 的 `.cpp` 文件，然后和：

```text
csrc/sage_attention_asm.mu
```

一起编译成 `sage_attention.so`。

因此，`mubin` 文件不是运行时按路径动态打开，而是在构建 native module 时已经作为 C++ 符号链接进 `.so`。

### 5.2 registry 把 kernel ID 映射到二进制数组

`csrc/sage_attention_asm.mu` include 了：

```cpp
#include "mubin/mp31/mp31_sage_attention_registry.hpp"
```

而这个 registry 又 include：

```cpp
#include "mp31_sage_attention_mubin.hpp"
```

`mp31_sage_attention_mubin.hpp` 声明了大量外部符号：

```cpp
extern unsigned char e4m3tce_flash_atten_qk_int8_quant_mode_6_512_256x128x128[];
extern unsigned int  e4m3tce_flash_atten_qk_int8_quant_mode_6_512_256x128x128_len;
```

这些符号对应 `mubin/mp31/flash_atten/*.cpp` 中的二进制数组。

registry 通过宏：

```cpp
REGISTER_FA_ASM_KERNEL(..., KERN_NAME)
```

把一个 `FlashAttenAsmID` 映射到：

```text
(unsigned char* blob, const char* kernel_name)
```

这个 `FlashAttenAsmID` 包含：

- 是否 causal
- 是否 varlen
- dtype
- head dim bucket
- 是否 KV cache
- quant mode
- QK 是否 INT8
- 是否 FP8 output

这就是 MATE 的预编译 kernel dispatch key。

### 5.3 dispatcher 根据运行时参数查表

`SageAttenQuantizedASMDispatcher::get_kernel_config()` 会把运行时 Args 编码成 `FlashAttenAsmID`，然后：

1. 先查 `kernel_map`，如果之前加载过，就复用 `MUfunction`。
2. 如果没加载过，查 `fa_asm_kern_registry`。
3. 找到后构造 `MateAsmKernel`。

`MateAsmKernel` 的构造函数在 `csrc/asm_common.hpp` 中：

```cpp
MateAsmKernel(const unsigned char* path, const std::string& func_name) {
  muModuleLoadData(&asm_module, path);
  muModuleGetFunction(&asm_func, asm_module, func_name.c_str());
}
```

这里的 `path` 实际不是文件路径，而是 `unsigned char[]` 二进制数组的内存地址。`muModuleLoadData` 会直接从内存中的 ELF/mubin blob 加载 MUSA module，然后 `muModuleGetFunction` 获取具体 kernel 函数句柄。

### 5.4 最后发射 kernel

dispatcher 拿到 `MUfunction` 后，`SageAttentionQuantizedAsmKernel` 会组装 kernel 参数和 launch config，最后调用：

```cpp
muLaunchKernelEx(&launch_config, *config.asm_func, kernel_params, nullptr);
```

这样，一次 Python 层的 `sageattn(q, k, v)` 最终就变成了一个预编译 MUSA kernel 的 launch。

## 6. FlashAttention、FlashMLA、DeepGEMM 的实现模式

MATE 的其他 wrapper 也大体遵循同样模式，只是复杂度不同。

### 6.1 FlashAttention / FA3

FlashAttention wrapper 提供 `flash_attn_3` 和 `flash_attn_interface` 兼容 API。

底层由 `mate/mha_interface.py` 和 `mate/jit/attention/fmha/` 负责：

- 支持 dense attention。
- 支持 varlen attention。
- 支持 KV cache decode。
- 支持 scheduler metadata。
- 支持 mutlass 和 mubin backend 选择。
- 对满足条件的输入走 `mubin` 快路径。
- 其他路径可走 MUTLASS/JIT 生成 kernel。

SGLang 的 MUSA FA3 backend 正是通过 `flash_attn_interface` 接入这一层。

### 6.2 FlashMLA

FlashMLA wrapper 提供 `flash_mla` 兼容接口，重点服务 DeepSeek MLA 架构。

它支持：

- MLA metadata 生成。
- dense MLA decode。
- sparse MLA decode。
- sparse prefill。
- FP8 KV cache。
- paged KV cache。

相比 SageAttention，FlashMLA 更关注 DeepSeek 风格的 compressed KV 和 MLA decode 调度。

### 6.3 DeepGEMM

DeepGEMM wrapper 提供 `deep_gemm` 兼容接口。

底层 GEMM 实现会使用：

- MUTLASS 模板。
- MUSA 编译器。
- `csrc/mubin/mp31/gemm/` 下的预编译 GEMM kernel。
- JIT/AOT 编译出的 native module。

GEMM 这类算子 shape 空间巨大，因此 MATE 同时保留了模板化生成和预编译热点 kernel 两种方式。

## 7. mubin 是什么：MUSA 版 cubin/fatbin

`mubin` 可以理解成 MUSA 平台上的预编译设备端二进制。它在项目里以 C++ 数组形式出现：

```cpp
unsigned char some_kernel[] = {
  0x7F, 0x45, 0x4C, 0x46, ...
};
unsigned int some_kernel_len = ...;
```

这和 CUDA 中的 cubin/fatbin 很相似：

- 都是特定硬件架构上的设备端二进制。
- 都不是可读源码。
- 都用于避免运行时重新生成/编译热点 kernel。
- 都可以通过 driver API 加载 module 和获取 function。

MATE 中加载 `mubin` 的关键 API 是：

```cpp
muModuleLoadData(&asm_module, path);
muModuleGetFunction(&asm_func, asm_module, func_name.c_str());
muLaunchKernelEx(...);
```

所以从实现角度看，`mubin` 是 MATE 性能路径里的黑盒 kernel 库。

## 8. TVM-FFI 的作用

MATE 使用 TVM-FFI，而不是单纯的 PyTorch C++ extension。

它的作用包括：

1. 把 C++ 函数导出成 Python 可调用函数。
2. 提供 tensor view 的类型封装。
3. 让同一套 native kernel 有机会服务 Python、C++、Rust 等多语言绑定。
4. 降低 MATE 对 PyTorch extension ABI 的直接耦合。

例如：

```cpp
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sage_attn_quantized_asm, sage_attn_quantized_asm);
```

这行会把 `sage_attn_quantized_asm` 注册为 TVM-FFI 函数。Python 侧加载 `.so` 后，就能通过：

```python
module.sage_attn_quantized_asm(...)
```

直接调用 C++ 实现。

## 9. JIT、AOT 与预编译 mubin 的关系

MATE 同时使用三类 kernel 交付方式：

### 9.1 JIT

运行时根据配置生成源码、生成 ninja 文件、编译 `.so`，然后加载。

优点是灵活，可以覆盖更多参数组合。缺点是首次调用有编译开销。

### 9.2 AOT

提前用：

```bash
MATE_MUSA_ARCH_LIST=3.1 python -m mate.aot
```

预构建常用 kernel，然后打包进 wheel。

优点是部署时减少首次编译成本。

### 9.3 mubin

预编译设备二进制，以 `unsigned char[]` 的形式嵌入 C++。

优点是启动快、性能路径固定、适合热点 shape。缺点是不可读、不可轻易修改，扩展 shape 需要重新生成对应二进制。

三者的关系可以理解为：

```text
JIT/AOT 负责生成和加载 native module
mubin 负责提供 native module 内部可调用的预编译设备 kernel
MUTLASS/.mu/Jinja 负责可生成或可维护的 kernel 实现路径
```

## 10. MATE 对 SGLang 的价值

在 SGLang 中，MUSA 支持需要解决两类问题：

1. 上层推理框架如何少改代码接入 MUSA。
2. 底层 attention、MLA、GEMM、sampling 等热点算子如何达到可用性能。

MATE 主要解决第一类和第二类之间的断层。

例如：

- SGLang LLM 推理中的 FA3 backend 可以继续使用 `flash_attn_interface`。
- Diffusion 推理中的 SageAttention backend 可以继续 `from sageattention import sageattn`。
- DeepSeek MLA 相关路径可以通过 FlashMLA 风格接口调用 MATE MLA kernel。
- GEMM 热点可以通过 DeepGEMM 兼容接口映射到 MUSA/MUTLASS 实现。

这让 SGLang 在支持 MUSA 时，不必把每个第三方 CUDA kernel API 都在业务层重新适配一遍。

## 11. 总结

MATE 的实现可以总结成三句话：

1. **上层兼容 CUDA 生态 API**：通过 wrapper 保留 `flash_attn_3`、`sageattention`、`flash_mla`、`deep_gemm` 等包名和函数入口。
2. **中层负责 dispatch、JIT/AOT、TVM-FFI 和参数适配**：Python interface 分配输出、整理参数，TVM-FFI 把调用转进 native module。
3. **底层使用 MUSA 原生 kernel**：包括 `.mu`、MUTLASS 模板、Jinja 生成代码，以及 MP31 上的预编译 `mubin` 二进制。

从工程角度看，MATE 是一个“迁移层 + runtime + kernel 库”的组合体。它既服务于兼容性，也服务于性能。兼容性来自 wrapper 和 TVM-FFI，性能则来自 MUTLASS、AOT/JIT 和 `mubin` 预编译 kernel。

如果用一句话概括：**MATE 是把 CUDA GenAI 算子生态搬到 MUSA 平台上的桥梁；上层看起来还是 FlashAttention、SageAttention、FlashMLA、DeepGEMM，底层实际已经换成 MUSA 原生实现。**
