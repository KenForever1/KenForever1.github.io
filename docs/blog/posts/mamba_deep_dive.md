---
title: TensorRT-Edge-LLM Mamba 实现深度解析
date: 2026-05-10
authors: [KenForever1]
categories: 
  - LLM推理
labels: [LLM推理]
pin: true
comments: true
---

<!-- [TOC] -->

本文结合 `cpp/runtime/mambaCacheManager.h`、`cpp/plugins/mamba/mambaPlugin.h`，以及项目中 Python 导出、ONNX custom op、TensorRT plugin、C++ Runtime cache 管理等链路，系统说明 TensorRT-Edge-LLM 中 Mamba 的设计与实现。

<!-- more -->

## Mamba 在项目中的位置

TensorRT-Edge-LLM 的主体推理链路是：

```text
Checkpoint / HuggingFace config
        ↓
experimental/llm_loader 构造本地模型实现
        ↓
torch.onnx.export(dynamo=True) 导出 ONNX
        ↓
C++ Builder 将 ONNX 编译为 TensorRT engine
        ↓
C++ Runtime 执行 prefill / decode
```

Mamba 不是独立的完整模型类型，而是作为 hybrid decoder 中的一类 layer 出现。项目里典型支持对象是 Nemotron-H / Nemotron-Omni 这类混合架构：

```text
Attention layer + Mamba layer + MLP layer + MoE layer
```

其中：

- Attention layer 使用 KV cache；
- Mamba layer 使用 recurrent SSM state 和 causal conv state；
- MLP / MoE layer 通常无跨 token 状态；
- Runtime 通过 `HybridCacheManager` 统一路由 attention cache 和 Mamba state。

关键文件：

- `experimental/llm_loader/config.py`：解析 hybrid / Mamba 配置。
- `experimental/llm_loader/models/nemotron_h/modeling_nemotron_h.py`：Python 侧 Mamba2 layer 建模。
- `experimental/llm_loader/models/ops.py`：定义 `causal_conv1d`、`update_ssm_state` trace-time custom op stub。
- `experimental/llm_loader/onnx/dynamo_translations.py`：把 custom op 翻译为 `trt_edgellm` ONNX custom op。
- `cpp/plugins/mamba/mambaPlugin.{h,cpp}`：TensorRT plugin，实现 Mamba selective state update。
- `cpp/plugins/mamba/causalConv1dPlugin.{h,cpp}`：TensorRT plugin，实现 causal conv1d。
- `cpp/runtime/mambaCacheManager.{h,cpp}`：运行时管理 Mamba recurrent / conv states。
- `cpp/runtime/hybridCacheManager.{h,cpp}`：统一管理 Attention KV cache 与 Mamba state。
- `cpp/runtime/llmEngineRunner.{h,cpp}`：加载 engine、绑定 cache/state、执行 prefill / decode。

## Mamba 的核心思想

传统 Transformer attention 依赖 KV cache：每生成一个 token，都要让当前 token attend 到历史 KV。Mamba / State Space Model 的思路不同：它把历史信息压缩进一个固定大小的 recurrent state，解码时只需要更新这个状态，而不是保存所有历史 token 的 K/V。

项目中的 Mamba2 核心计算可以抽象为：

```text
输入 hidden_states
    ↓
in_proj 投影
    ↓
拆分为 gate、conv path、dt
    ↓
causal conv1d 更新局部卷积状态
    ↓
拆分出 SSM 输入 x、B、C
    ↓
selective state update 更新 SSM recurrent state
    ↓
gated RMSNorm
    ↓
out_proj 回到 hidden_size
```

在 `MambaMixer.forward()` 中，这条路径对应：

```text
hidden_states
  -> in_proj
  -> split(gate, hidden_states_for_conv, dt)
  -> causal_conv1d(..., conv_state)
  -> SiLU
  -> split(ssm_input, ssm_b, ssm_c)
  -> update_ssm_state(..., ssm_state)
  -> gated RMSNorm
  -> out_proj
```

这说明项目并不是把整个 Mamba block 写成一个大 plugin，而是拆成两个主要 runtime custom op：

1. `causal_conv1d`：负责局部卷积和 conv state 更新；
2. `update_ssm_state`：负责 selective SSM recurrent state 更新。

这种拆分的好处是：

- 与 Python 模型结构一致，导出更自然；
- TensorRT plugin 边界清晰；
- conv 和 SSM 可以分别优化；
- MTP / speculative decoding 可以分别收集中间 conv state 和 recurrent state。

## Python 模型侧：如何表达 Mamba

### MambaConfig

`experimental/llm_loader/config.py` 中的 `MambaConfig` 保存 Mamba layer 的关键超参数：

```text
num_heads          -> mamba_num_heads
head_dim           -> mamba_head_dim
ssm_state_size     -> SSM state size / dstate
conv_dim           -> causal conv channel count
conv_kernel        -> conv1d kernel size
n_groups           -> SSM groups
intermediate_size  -> num_heads * head_dim
```

这些参数后续会写入 runtime `config.json`，C++ Runtime 再据此初始化 `MambaCacheManager`。

### MambaMixer

`experimental/llm_loader/models/nemotron_h/modeling_nemotron_h.py` 的 `MambaMixer` 是项目里的 Mamba2 计算模块。

其参数命名直接对齐 checkpoint key：

```text
backbone.layers.N.mixer.in_proj.weight
backbone.layers.N.mixer.out_proj.weight
backbone.layers.N.mixer.conv1d.weight
backbone.layers.N.mixer.conv1d.bias
backbone.layers.N.mixer.A_log
backbone.layers.N.mixer.D
backbone.layers.N.mixer.dt_bias
backbone.layers.N.mixer.norm.weight
```

`in_proj` 的输出维度为：

```text
d_inner + conv_dim + num_heads
```

其中：

- `d_inner = num_heads * head_dim`，用于 gate；
- `conv_dim`，进入 causal conv1d；
- `num_heads`，作为时间步长参数 `dt`。

forward 中的关键 reshape：

```text
ssm_input: [batch, seq_len, d_inner]
    -> [batch, seq_len, num_heads, head_dim]

ssm_b / ssm_c:
    [batch, seq_len, n_groups * ssm_state_size]
    -> [batch, seq_len, n_groups, ssm_state_size]

ssm_state:
    [batch, num_heads, head_dim, ssm_state_size]
```

这与 C++ `MambaPlugin` 的输入 shape 完全对应。

## ONNX 导出侧：Mamba 如何穿过 torch.export

Mamba 中的 selective scan / state update 对标准 ONNX 来说并不友好，因此项目采用 custom op 策略。

### trace-time stub

`experimental/llm_loader/models/ops.py` 定义了两个 `torch.library.custom_op`：

```text
trt_edgellm::causal_conv1d
trt_edgellm::update_ssm_state
```

它们在 Python eager / trace 阶段只返回形状和 dtype 正确的 dummy tensor：

```text
causal_conv1d -> output, conv_state_out
update_ssm_state -> output, state_out
```

这样 `torch.onnx.export(dynamo=True)` 可以完成图捕获和 shape propagation，不需要在 Python 里实现真正的高性能 kernel。

### ONNX translation

`experimental/llm_loader/onnx/dynamo_translations.py` 把上述 PyTorch custom op 翻译成 ONNX custom domain：

```text
_trt_edgellm.causal_conv1d(...)
_trt_edgellm.update_ssm_state(...)
```

`update_ssm_state` 的 ONNX 输入大致是：

```text
hidden_states     [batch, seq_len, num_heads, head_dim] FP16
ssm_a             [num_heads] FP32
ssm_b             [batch, seq_len, n_groups, state_size] FP16
ssm_c             [batch, seq_len, n_groups, state_size] FP16
ssm_d             [num_heads] FP16
dt                [batch, seq_len, num_heads] FP16
dt_bias           [num_heads] FP16
state             [batch, num_heads, head_dim, state_size] FP16
context_lengths   [batch] INT32
```

输出是：

```text
output            [batch, seq_len, num_heads, head_dim] FP16
state_out         [batch, num_heads, head_dim, state_size] FP16
```

### 导出后的 dtype 修正

`experimental/llm_loader/onnx/export.py` 里有一段专门处理 plugin FP32 输入的逻辑：

```text
Mamba2 update_ssm_state: input[1] = ssm_A must stay FP32
```

也就是说，虽然多数权重/激活以 FP16 运行，但 `ssm_A` 被要求保持 FP32。这是因为 SSM 离散化中 `exp(A * dt)` 对数值稳定性更敏感，项目通过 ONNX initializer dtype fixup 保证 TensorRT plugin 获得 FP32 的 `A`。

## TensorRT Plugin：`MambaPlugin`

`cpp/plugins/mamba/mambaPlugin.h` 中的 `MambaPlugin` 是 `update_ssm_state` 的 TensorRT plugin。它注册为：

```text
plugin name: update_ssm_state
domain: trt_edgellm
version: 1
```

### 数学语义

文件注释给出的核心公式是：

```text
new_state = state * exp(A * dt) + B * dt * x
output    = sum_i(new_state_i * C_i) + D * x
```

含义：

- `state`：历史压缩状态；
- `A`：状态转移参数；
- `dt`：动态时间步长；
- `B`：输入写入 state 的系数；
- `C`：从 state 读出 output 的系数；
- `D`：skip connection / direct term；
- `x`：当前输入。

这个 plugin 只做 selective state update。SiLU gate / gated RMSNorm 在 ONNX 图中由其他算子表达，不在该 plugin 内部处理。

### 输入输出约定

`MambaPlugin` 的输入顺序在 `mambaPlugin.cpp` 中固定：

```text
[0] x                [batch, (seq_len,) nheads, dim] FP16
[1] A                [nheads] FP32
[2] B                [batch, (seq_len,) ngroups, dstate] FP16
[3] C                [batch, (seq_len,) ngroups, dstate] FP16
[4] D                [nheads] FP16
[5] dt               [batch, (seq_len,) nheads] FP16
[6] dt_bias          [nheads] FP16
[7] state            [batch, nheads, dim, dstate] FP16
[8] context_lengths  [batch] INT32
```

输出：

```text
[0] output           same shape as x
[1] state_out        same shape as state
```

注意 `x` 支持两种形态：

```text
Decode:  [batch, nheads, dim]
Prefill: [batch, seq_len, nheads, dim]
```

这对应 runtime 的双阶段执行：

- decode 阶段一次处理一个 token；
- prefill 阶段处理一段 prompt。

### dtype 与 format 限制

`supportsFormatCombination()` 中限制：

- 主要数据输入为 FP16；
- `A` 为 FP32；
- `context_lengths` 为 INT32；
- tensor format 为 linear。

这体现了一个明确取舍：

- 大部分数据走 FP16，提高吞吐和降低显存；
- 对敏感的状态转移参数 `A` 保留 FP32，提升数值稳定性；
- plugin 接口保持简单，便于 TensorRT shape 推理和 kernel 调度。

### build 阶段参数推导

`configurePlugin()` 会从输入 shape 推导：

```text
dim      <- x 最后一维
nheads   <- x 倒数第二维
dstate   <- B 最后一维
ngroups  <- B 倒数第二维
```

如果 ONNX node 没显式提供这些属性，build 阶段也可以从 profile shape 中得到。序列化时则通过 `getFieldsToSerialize()` 保存：

```text
dim, dstate, nheads, ngroups, dt_softplus
```

这样 engine 反序列化后 runtime 不需要重新猜测这些结构参数。

### enqueue 执行路径

`enqueue()` 的关键步骤：

1. 读取 batch、seq_len、shape；
2. 将输入 state copy 到输出 state；
3. 把 TensorRT raw pointer 包成项目内部 `rt::Tensor` view；
4. 根据 `x` 是否有 seq_len 维度选择 prefill 或 decode kernel；
5. 调用底层 CUDA kernel：
   - `invokeSelectiveStateUpdate()`：decode / 单步；
   - `invokeSelectiveStateUpdatePrefill()`：prefill / 多步；
   - 若启用 `CUTE_DSL_SSD_ENABLED` 且满足条件，可走 CuTe DSL chunked SSD path。

这里有一个重要实现细节：plugin 先把 input state copy 到 output state，然后 kernel 在 output state 上原地更新。这样 TensorRT 图语义仍是函数式的：

```text
state_in -> state_out
```

但 kernel 内部可以用 in-place 更新减少额外临时 buffer。

### prefill 性能路径

`mambaPlugin.h` 注释指出：当 `seq_len > 1` 时，当前默认实现会在 plugin 内部循环调用单步 kernel：

```text
for t in seq_len:
    update state
```

这对 decode 最优，因为 decode 本来就是 `seq_len=1`。但对长 prompt prefill，串行 step loop 会比并行 scan 慢。

项目中已经预留了 CuTe DSL SSD 优化路径：

```text
CUTE_DSL_SSD_ENABLED
CuteDslSSDRunner::canImplement(...)
seqLen >= 128
```

满足条件时可以走 chunked SSD prefill kernel，以降低长序列 prefill 的串行开销。当前 plugin creator 仍对 `chunk_size > 1` 做限制，说明通用 chunk scan 支持还在演进中。

## Runtime 状态管理：`MambaCacheManager`

Attention 的历史状态是 KV cache；Mamba 的历史状态是两类 state：

```text
recurrent state: [maxBatchSize, recurrentStateNumHeads, recurrentStateHeadDim, recurrentStateSize]
conv state:      [maxBatchSize, convDim, convKernel]
```

`cpp/runtime/mambaCacheManager.h` 的 `MambaCacheManager` 就是专门管理这些 per-layer state 的组件。

### Config

`MambaCacheManager::Config` 包含：

```text
numRecurrentLayers
maxBatchSize
recurrentStateNumHeads
recurrentStateHeadDim
recurrentStateSize
convDim
convKernel
maxIntermediateSeqLen
recurrentStateType
convStateType
```

其中：

- `numRecurrentLayers == 0` 时 manager 是 no-op，不分配显存；
- `maxIntermediateSeqLen > 0` 表示启用 MTP intermediate state buffer；
- dtype 默认是 FP16。

### 显存布局

构造函数中，每个 Mamba layer 分配两块主状态：

```text
mRecurrentStates[layer]
mConvStates[layer]
```

shape 分别是：

```text
recurrent: [maxBatchSize, numHeads, headDim, stateSize]
conv:      [maxBatchSize, convDim, convKernel]
```

分配后立即 `cudaMemsetAsync(..., 0)` 清零。这样每个 batch slot 的初始历史状态为空，符合新请求的语义。

### 为什么按 layer 分 vector 管理

`mRecurrentStates` 和 `mConvStates` 是 `std::vector<rt::Tensor>`，每层一份 tensor。这与 ONNX / TensorRT engine 的 binding 方式一致：每个 recurrent layer 在 graph 中有自己的 state input/output。

这样做的优点：

- 绑定 engine input/output 简单；
- 每层 shape 相同但生命周期独立；
- 与 `HybridCacheManager` 的 absolute-layer routing 容易对应；
- prompt cache capture / restore 可以按层操作。

缺点是：

- layer 数较多时会有多个 Tensor 对象；
- 若要做极致显存连续化，还可以进一步将所有 layer state pack 到一个大 buffer 中。但当前设计更清晰，且便于 TensorRT binding。

### 清空、捕获与恢复

`clearStates()` 清零所有 recurrent 和 conv state。注释说明它会在 warmup inference 后、CUDA graph capture 前调用，确保 capture 的起始状态干净。

`captureRecurrentStates()` 和 `captureConvStates()` 会把某个 batch slot 的状态复制到新 tensor 中，用于 system prompt cache：

```text
batch slot N 的 Mamba states
        ↓
保存为 shape [1, ...] 的 device tensors
```

这与 Attention KV cache 的 system prompt cache 是同一类需求：如果系统提示词相同，可以复用已经计算出的历史状态。

对 Attention 来说复用 KV；对 Mamba 来说复用 recurrent / conv state。

## HybridCacheManager：Attention 与 Mamba 的统一路由

`cpp/runtime/hybridCacheManager.h` 体现了项目对 hybrid model 的核心抽象：

```text
absolute decoder layer index
        ↓
LayerType::kAttention -> KVCacheManager
LayerType::kMamba     -> MambaCacheManager
```

`HybridCacheManager::Config` 中有：

```text
layerTypes
kvConfig
mambaConfig
maxBatchSize
```

这使 runtime 不需要假设模型是纯 attention，也不需要把 Mamba 当成特殊 case 到处散落。上层只按 layer index 请求状态：

```text
getCombinedKVCache(absLayerIdx)
getRecurrentState(absLayerIdx)
getConvState(absLayerIdx)
```

实际由 `HybridCacheManager` 进行 absolute index 到 local index 的映射。

这种设计对扩展非常关键：

- pure Transformer：`numRecurrentLayers = 0`，Mamba manager no-op；
- hybrid Mamba：attention layer 和 Mamba layer 混排；
- GatedDeltaNet：也可复用 recurrent-state 管理框架；
- future linear attention：可以继续纳入类似 routing 结构。

## MTP / speculative decoding 下的 Mamba state

`MambaCacheManager` 里还有 intermediate states：

```text
mIntermediateRecurrentStates
mIntermediateConvStates
```

shape：

```text
intermediate recurrent:
[maxBatchSize, maxIntermediateSeqLen, numHeads, headDim, stateSize]

intermediate conv:
[maxBatchSize, maxIntermediateSeqLen, convDim, convKernel]
```

它们用于 MTP / speculative decoding 场景：一次验证多个候选 token 时，每个候选 token 都可能产生一份中间 recurrent / conv state。最终只有被接受的 token 对应的 state 应该进入主 state pool。

流程可以理解为：

```text
base verify tree / MTP verify
        ↓
每个候选位置产生 intermediate Mamba states
        ↓
acceptLengths 决定每个 batch 接受多少 token
        ↓
scatterMtpStates() 将被接受位置的 state 写回主 state
```

`scatterMtpStates()` 调用：

```text
mtpScatterRecurrentStates(...)
mtpScatterConvStates(...)
```

这说明项目在 speculative decoding 中不只处理 Attention KV cache，也完整处理 Mamba recurrent / conv state 的接受与回滚问题。

这是 hybrid 架构里非常关键的一点：

- Attention speculative verify 后要更新 KV；
- Mamba speculative verify 后要更新 recurrent state 和 conv state；
- 两者必须根据同一组 `acceptLengths` 保持一致。

否则模型后续 decode 的历史状态会错位。

## Runtime 配置如何从导出阶段传到 C++

`experimental/llm_loader/checkpoint/checkpoint_utils.py` 会把 hybrid 信息写到 runtime `config.json`：

```json
{
  "model_type": "hybrid_mamba",
  "num_linear_attn_layers": ...,
  "num_attention_layers": ...,
  "recurrent_state_num_heads": ...,
  "recurrent_state_head_dim": ...,
  "recurrent_state_size": ...,
  "conv_dim": ...,
  "conv_kernel": ...,
  "layer_types": ["attention", "mamba", ...],
  "kv_layer_configs": [...]
}
```

C++ `LLMEngineRunnerConfig` 中也有对应字段：

```text
numLinearAttnLayers
numAttentionLayers
recurrentStateNumHeads
recurrentStateHeadDim
recurrentStateSize
convDim
convKernel
layerTypes
kvLayerConfigs
```

这构成了 Python exporter 与 C++ runtime 的契约：

- Python 知道 checkpoint 结构和 layer 类型；
- 导出时写出 normalized config；
- C++ 不需要读 HF 原始 config，只消费 runtime config；
- Runtime 根据这些字段分配 KV cache 和 Mamba state。

## Prefill 与 Decode 的状态语义

Mamba 在 prefill / decode 中与 Attention 类似，都需要维护历史状态，但状态形式不同。

### Prefill

Prefill 输入：

```text
x: [batch, seq_len, nheads, dim]
state: [batch, nheads, dim, dstate]
context_lengths: [batch]
```

输出：

```text
output: [batch, seq_len, nheads, dim]
state_out: [batch, nheads, dim, dstate]
```

对于每个 batch，plugin 需要根据有效 context length 更新状态，避免 padding token 污染 state。

### Decode

Decode 输入通常是：

```text
x: [batch, nheads, dim]
```

或者某些图里仍保留 4D 但 `seq_len=1`。

此时 plugin 不应使用 cumulative `context_lengths` 做多步 scan，否则会越界或重复更新。代码里也明确区分：

```text
seqLen > 1 时才使用 context_lengths
seqLen == 1 decode 时不使用 context_lengths 做 prefill scan
```

## 与 Attention KV Cache 的对比

| 维度 | Attention | Mamba |
|---|---|---|
| 历史状态 | KV cache | recurrent state + conv state |
| 状态大小 | 随序列长度增长 | 固定大小，和 state_size / conv_kernel 有关 |
| Decode 代价 | 依赖历史 KV 读 | 更新固定 recurrent state |
| Prefill 优化 | 并行 attention / fused attention | 需要 scan；当前可串行，也预留 chunked SSD |
| Runtime manager | `KVCacheManager` | `MambaCacheManager` |
| Hybrid 统一入口 | `HybridCacheManager` | `HybridCacheManager` |

Mamba 的理论优势主要在长序列 decode 时状态固定，不需要随着上下文长度无限增长 KV。但 prefill 阶段如果没有高效并行 scan，可能成为瓶颈。因此项目中特别在 plugin 注释和实现里提到未来 chunked scan / CuTe DSL SSD 优化路径。

## 项目实现的设计取舍

### 不把 Mamba 当成 Attention 的变种

项目没有强行把 Mamba state 塞进 KV cache，而是单独设计 `MambaCacheManager`。这是正确的抽象，因为二者状态结构、更新规则、capture/restore、speculative accept 都不同。

### 用 HybridCacheManager 做统一门面

虽然底层状态不同，上层 Runtime 不希望到处写：

```text
if attention layer...
if mamba layer...
```

因此 `HybridCacheManager` 负责路由。这样 `LLMEngineRunner` 可以面向统一 cache manager 编程。

### Python 侧保留模型语义，C++ 侧实现性能 kernel

Python 侧 `MambaMixer` 保留清晰的模型结构：projection、conv、SSM、norm、projection。真正高性能和 TensorRT 兼容性由 C++ plugin 负责。

这是 TensorRT-Edge-LLM 一贯的设计模式：

```text
Python: checkpoint parsing + graph construction + ONNX custom op
C++: TensorRT plugin + CUDA kernel + runtime state management
```

### 显式 state I/O

ONNX graph 中 Mamba states 是显式输入输出，而不是 plugin 内部隐藏状态。这一点非常重要：

- TensorRT engine 是纯函数式执行；
- Runtime 可以控制 state 生命周期；
- batch eviction / compaction 可以由 Runtime 管；
- system prompt cache 可以 capture/restore；
- CUDA graph capture 前可以清空状态；
- speculative decoding 可以控制 accept/reject 后的状态提交。

### dtype 保守策略

大部分数据 FP16；`A` 保持 FP32。这是性能与数值稳定性的折中。

## 当前限制与潜在优化

从代码注释和实现可以看出当前限制：

1. `MambaPlugin` 主要支持 FP16 data path；
2. `A` 必须是 FP32；
3. 默认长 prefill 可能走 step loop，存在串行开销；
4. `chunk_size > 1` 在 plugin creator 中仍被限制；
5. 非 trivial `time_step_limit` 暂未支持；
6. CuTe DSL SSD path 需要编译宏和硬件/shape 条件满足。

潜在优化方向：

- 完善 chunked parallel scan；
- 扩展更多 dtype / quantized state 支持；
- 减少 state copy 或让 TensorRT binding 支持更直接的 ping-pong buffer；
- 将多层 state 做更连续的 packed allocation；
- 对 prefill 和 decode 分别建立更细粒度 TensorRT profile / plugin specialization。

## 一句话总结

TensorRT-Edge-LLM 的 Mamba 实现本质上是一个跨 Python exporter、ONNX custom op、TensorRT plugin、CUDA kernel、C++ runtime cache manager 的完整工程化闭环：Python 侧保留 Mamba2 模型语义，ONNX 侧显式暴露 conv / recurrent state，TensorRT plugin 执行 selective SSM 更新，C++ Runtime 用 `MambaCacheManager` 管理每层每 batch 的状态，并通过 `HybridCacheManager` 与 Attention KV cache 统一到 hybrid decoder 推理框架中。
