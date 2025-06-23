---
title: sglang投机推理实现
date: 2025-06-08
authors: [KenForever1]
categories: 
  - LLM推理
labels: [LLM推理]
comments: true
---

阅读[sglang/pull/3582](github.com/sgl-project/sglang/pull/3582)。

## deepseek_nextn.py
实现DeepSeek NextN推测解码。
通过forward_batch.spec_info.hidden_states获取推测解码的隐藏状态
使用eh_proj合并当前和推测的隐藏状态
支持分布式并行计算(通过enable_dp_attention标志)
<!-- more -->

```python
# hidden_states: 当前模型的嵌入输出
# forward_batch.spec_info.hidden_states: 来自推测模型(draft model)的隐藏状态
hidden_states = self.eh_proj(
    torch.cat(
        (
            self.enorm(hidden_states),  # 当前模型归一化输出
            self.hnorm(forward_batch.spec_info.hidden_states)  # 推测模型归一化输出
        ),
        dim=-1  # 在特征维度拼接
    )
)
```
通过合并两个模型的隐藏状态，可以同时利用：
当前模型的精确表示能力
推测模型的快速生成能力

## deepseek_v2.py

### DeepseekV2MoE (混合专家层)
实现稀疏激活的混合专家系统，核心功能：动态路由机制、专家并行计算、支持共享专家。
路由机制：
使用门控网络(MoEGate)计算每个token的路由logits
支持分组TopK选择专家(use_grouped_topk)
可选的专家分数校正(correction_bias)
工作流程：
计算路由logits
选择top-k专家
分发tokens到对应专家
并行执行专家计算
聚合专家输出

DeepseekV2MoE 中的 use_grouped_topk 是一种高效的分组专家选择机制，其工作原理如下：
分组策略：
将专家划分为 num_expert_group 个组
每个组包含 num_experts/num_expert_group 个专家
```python
scores.view(num_token, num_expert_group, -1)  # 将专家分数按组划分
```
两阶段选择：
组级选择：
```python 
group_scores = scores.view(...).max(dim=-1).values  # 取每组最高分
group_idx = torch.topk(group_scores, k=topk_group)  # 选择topk_group个最优组
```
组内选择：
```python
tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # 屏蔽未选中组的专家
topk_weights, topk_ids = torch.topk(tmp_scores, k=topk)  # 在选中组内选topk专家
```
topk_idx 作用：在MoE层中标识被选中的专家索引，通过路由算法生成：topk_weights, topk_idx = select_experts(...)，用于后续的专家并行计算和数据分发

### DeepseekV2DecoderLayer (解码器层)的实现

```python
class DeepseekV2DecoderLayer(nn.Module):
    def __init__(self, config, layer_id, quant_config=None, is_nextn=False):
        self.self_attn = DeepseekV2AttentionMLA(...)  # 多头注意力
        self.mlp = DeepseekV2MoE(...) if config.n_routed_experts else DeepseekV2MLP(...)
        self.input_layernorm = RMSNorm(...)
        self.post_attention_layernorm = RMSNorm(...)
```
输入归一化 → 自注意力 → 残差连接
中间归一化 → MoE/MLP → 残差连接

## export_deepseek_nextn.py

用于从 DeepSeek-V3/R1 模型中导出 NextN 层的工具，主要用于支持推测解码(Speculative Decoding)。以下是核心功能解析：

主要作用：
从完整模型中提取 NextN 预测层（用于推测解码的草稿模型）

生成独立的轻量级模型文件

修改配置文件适配单层结构

```
1. 定位NextN层位置 (get_nextn_layer_id)
2. 复制非参数文件 (copy_non_safetensors_files)
3. 更新配置文件 (update_and_save_config)
4. 提取指定层参数 (export_nextn_layer_parameters)
```

参数处理逻辑：
筛选原始模型中 model.layers.{nextn_layer_id} 开头的参数
重命名为 model.layers.0 以适应单层结构
排除 embedding 和 head 层参数

## speculative
DeepSeek的推测解码(Speculative Decoding)系统采用模块化设计，主要组件如下：
``` mermaid
graph TD
    A[EagleWorker] --> B[EagleUtils]
    A --> C[BuildEagleTree]
    A --> D[CudaGraphRunner]
    D --> E[DraftRunner]
    D --> F[ExtendRunner]
```

spec_info.py
定义枚举类型SpeculativeAlgorithm
支持EAGLE/EAGLE3两种加速算法
提供类型检查方法(is_eagle/is_eagle3)

eagle_worker.py (主控模块)
管理草稿模型和目标模型的协同执行
实现推测解码的状态机转换
处理token验证和回退逻辑

cuda_graph_runner.py
使用CUDA Graph优化计算流程
包含两个子运行器：
DraftRunner: 快速生成候选序列
ExtendRunner: 精确验证候选

工作流程：
构建阶段：通过build_eagle_tree.py预处理模型
草稿阶段：用轻量级模型生成N个候选token
验证阶段：用原模型并行验证所有候选
接受阶段：确定最长有效前缀

性能优化
使用CUDA Graph减少内核启动开销
树状验证结构(EAGLE3)提升吞吐量
动态批处理管理

## GPU执行算子间隙优化思路
• 核心解决问题：如何做到GPU不停歇，一直运行
• 解決方案：核心做到CPU卡GPU异步化（cuda-graph，去除同步逻辑mask，算子融合，异步化调度….）
