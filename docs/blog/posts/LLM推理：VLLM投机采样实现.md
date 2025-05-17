---
title: LLM推理：VLLM投机采样实现
date: 2025-05-17
authors: [KenForever1]
categories: 
  - LLM推理
labels: []
comments: true
---

[vllm源码7464a](https://sourcegraph.com/github.com/vllm-project/vllm@37464a0f745a0204da7443d2a6ef4b8f65e5af12/-/blob/vllm/spec_decode/spec_decode_worker.py)。

### 架构

SpecDecodeWorker
+ Proposers (ngram, draft model)
+ Scorer (top-1 scoring)
+ Verifier (rejection sampling)

<!-- more -->

![](https://raw.githubusercontent.com/KenForever1/CDN/main/spec_arch.png)

参考[A Hacker’s Guide to Speculative Decoding in vLLM](https://docs.google.com/presentation/d/1p1xE-EbSAnXpTSiSI0gmy_wdwxN5XaULO3AnCWWoRe4/edit?pli=1&slide=id.g272bde77b90_0_250#slide=id.g272bde77b90_0_250)。


### 调用流程
+ 创建一个SpecDecodeWorker

```
Create a SpecDecodeWorker.

Args:
    proposer_worker: A worker that can produce speculative tokens for
        sequences.
    scorer_worker: A worker that produces probabilities of speculative
        tokens according to some base model. Typically a vanilla vLLM
        Worker.
    rejection_sampler: A Torch module used to perform modified rejection
        sampling for speculative decoding.
```

核心逻辑：
```python

    @nvtx_range("spec_decode_worker._run_speculative_decoding_step")
    def _run_speculative_decoding_step(
            self, execute_model_req: ExecuteModelRequest,
            num_lookahead_slots: int) -> List[SamplerOutput]:
        """Execute a single step of speculative decoding.

        This invokes the proposer worker to get k speculative tokens for each
        sequence, then scores each speculative token using the scoring worker.

        Returns a list of SamplerOutput, each containing a single token per
        sequence.
        """
        assert num_lookahead_slots == execute_model_req.num_lookahead_slots

        # Generate proposals using draft worker.
        proposals = self.proposer_worker.get_spec_proposals(execute_model_req)

        proposal_scores = self.scorer.score_proposals(
            execute_model_req,
            proposals,
        )

        # 根据提议者模型和打分者模型，利用每个词元的概率来确定哪些推测词元会被接受
        accepted_token_ids, target_logprobs = self._verify_tokens(
            execute_model_req.seq_group_metadata_list, proposal_scores,
            proposals, execute_model_req.num_lookahead_slots)

        return self._create_output_sampler_list(
            execute_model_req.seq_group_metadata_list,
            accepted_token_ids,
            target_logprobs=target_logprobs,
            k=execute_model_req.num_lookahead_slots)
```

proposer_worker阅读分支有两个实现，分别是n_gram_worker和multi_step_worker。multi_step_worker也就是draft model proposer。

![](https://raw.githubusercontent.com/KenForever1/CDN/main/spec_proposer.png)

get_spec_proposals函数调用的就是Top1Proposer的get_proposals方法。


_verify_tokens会调用

```python
@nvtx_range("spec_decode_worker._verify_tokens")
    def _verify_tokens(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        proposal_scores: SpeculativeScores,
        proposals: SpeculativeProposals,
        max_proposal_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
      ...
      # rejection_sampler是一个用于为推测解码执行改进的拒绝采样的PyTorch模块。
      accepted_token_ids = self.rejection_sampler(
          target_probs=proposal_verifier_probs,
          bonus_token_ids=bonus_token_ids,
          draft_probs=proposal_probs,
          draft_token_ids=proposal_token_ids,
      )
      ...
```
见vllm/model_executor/layers/rejection_sampler.py。

### spec 采样算法实现

![](https://raw.githubusercontent.com/KenForever1/CDN/main/spec_sampling.png)

[来源于](https://arxiv.org/pdf/2302.01318.pdf.)。

```python
def _get_accepted(
            self,
            target_probs: torch.Tensor,  # [batch_size, k, vocab_size]
            draft_probs: torch.Tensor,  # [batch_size, k, vocab_size]
            draft_token_ids: torch.Tensor,  # [batch_size, k]
    ) -> torch.Tensor:
        r"""Create bool matrix over the proposed draft tokens. If
        True, then a token can be accepted, else it should be
        rejected.

        Given :math:`q(\hat{x}_{n+1}|x_1, \dots, x_n)`, the probability of
        :math:`\hat{x}_{n+1}` given context :math:`x_1, \dots, x_n` according
        to the target model, and :math:`p(\hat{x}_{n+1}|x_1, \dots, x_n)`, the
        same conditional probability according to the draft model, the token
        is accepted with probability:

        .. math::
            \min\left(1, \frac{q(\hat{x}_{n+1}|x_1, \dots, x_n)}
                           {p(\hat{x}_{n+1}|x_1, \dots, x_n)}\right)

        This implementation does not apply causality. When using the output,
        if a token is rejected, subsequent tokens should not be used.

        Returns a bool tensor of shape [batch_size, k] specifying which tokens
        are accepted.
        """
        batch_size, k, _ = draft_probs.shape
        batch_indices = torch.arange(batch_size,
                                     device=target_probs.device)[:, None]
        probs_indicies = torch.arange(k, device=target_probs.device)

        # shape [batch_size, k]
        selected_draft_probs = draft_probs[batch_indices, probs_indicies,
                                           draft_token_ids]

        # shape [batch_size, k]
        selected_target_probs = target_probs[batch_indices, probs_indicies,
                                             draft_token_ids]

        uniform_rand = torch.rand(batch_size,
                                  k,
                                  dtype=self.probs_dtype,
                                  device=target_probs.device)
        capped_ratio = torch.minimum(
            selected_target_probs / selected_draft_probs,
            torch.full((1, ), 1, device=target_probs.device))
        # 对应论文中算法公式，如果概率比小于比值，则接受该 draft token
        accepted = uniform_rand < capped_ratio

        return accepted
```


### 问题

+ top-1 proposal and scoring 和 Tree-attention 的区别？
‌top-1 proposal and scoring‌：基于贪婪搜索(Greedy Search)策略，每一步只选择概率最高的token作为输出，属于单路径解码方式。
‌Tree-attention‌：通过多解码头(Medusa heads)生成多个候选token，构建树状结构路径，并利用树注意力机制并行验证候选序列，属于多路径解码方式。