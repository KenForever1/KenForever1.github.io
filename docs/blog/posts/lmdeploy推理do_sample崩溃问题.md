---
title: lmdeploy推理do_sample崩溃问题
date: 2024-12-17
authors: [KenForever1]
categories: 
  - LLM推理
labels: [LLM推理]
comments: true
---

采用server方式推理一个模型，会崩溃。

采用pipeline方式推理一个模型，不会崩溃。

不管是server还是pipeline，都调用了同样的generate()方法，但是为什么会产生不同的结果？传递的参数不同。
<!-- more -->

比如server调用的位置如下：

```python
~/lmdeploy/serve/openai/api_server.py
@router.post('/v1/completions', dependencies=[Depends(check_api_key)])
async def completions_v1(request: CompletionRequest,
                         raw_request: Request = None):

                         ...
    gen_config = GenerationConfig(
        max_new_tokens=request.max_tokens if request.max_tokens else 512,
        do_sample=True,
        logprobs=request.logprobs,
        top_k=request.top_k,
        top_p=request.top_p,
        temperature=request.temperature,
        repetition_penalty=request.repetition_penalty,
        ignore_eos=request.ignore_eos,
        stop_words=request.stop,
        skip_special_tokens=request.skip_special_tokens,
        random_seed=random_seed,
        spaces_between_special_tokens=request.spaces_between_special_tokens)
    generators = []
    for i in range(len(request.prompt)):
        result_generator = VariableInterface.async_engine.generate(
            request.prompt[i],
            request.session_id + i,
            gen_config=gen_config,
            stream_response=True,  # always use stream to enable batching
            sequence_start=True,
            sequence_end=True,
            do_preprocess=False,
            adapter_name=adapter_name)
        generators.append(result_generator)
                         ...
```
可以看到do_sample=True。

```python
async def _async_infer(self, requests: AsyncIterator[Dict],
                        **kwargs) -> AsyncIterator[AsyncIterator[Response]]:
    async for req in requests:
        gen = self.generate(**req, **kwargs)
        yield gen

async def generate(
        self,
        messages,
        session_id: int,
        gen_config: Optional[GenerationConfig] = None,
        tools: Optional[List[object]] = None,
        stream_response: bool = True,
        sequence_start: bool = True,
        sequence_end: bool = True,  # no interactive mode by default
        step: int = 0,
        do_preprocess: bool = True,
        adapter_name: Optional[str] = None,
        skip_stop_tokens: bool = True,
        rewind_stop_tokens: bool = False,
        input_ids: Optional[List] = None,
        **kwargs):
        """Generate responses.

        Args:
            messages (str | List): chat history or prompt
            session_id (int): the session id
            gen_config (GenerationConfig | None): a instance of
                GenerationConfig. Default to None.
            stream_response (bool): whether return responses streamingly
            sequence_start (bool): indicator for starting a sequence
            sequence_end (bool): indicator for ending a sequence
            step (int): the offset of the k/v cache
            do_preprocess (bool): whether pre-process the messages. Default to
                True, which means chat_template will be applied.
        """

```

断点对比了generate()方法中的gen_config参数，

```python
# 断点可以使用:
import pdb
pdb.set_trace()
```

发现server中默认do_sapmle=True。pipelie中do_sample=False。

do_sample=True时，会崩溃。do_sample会影响top_k、top_p、temperature参数。
代码如下：
```python
        if not gen_config.do_sample:
            logger.warning(f'GenerationConfig: {gen_config}')
            logger.warning(
                'Since v0.6.0, lmdeploy add `do_sample` in '
                'GenerationConfig. It defaults to False, meaning greedy '
                'decoding. Please set `do_sample=True` if sampling '
                ' decoding is needed')
            # greedy decode
            gen_config.top_k = 1
            # avoid unnecessary process
            gen_config.temperature = 1.0
            gen_config.repetition_penalty = 1.0
```

如果do_sample=False, 采用的greedy 贪婪搜索(主要是top_k = 1)，不会崩溃。
do_sample=True时，采用的是采样方法(top_k > 1)，比如top_k = 40, 会崩溃。

```python
~/lmdeploy/pytorch/engine/logits_process.py
class FusedLogitsProcessor(LogitsWarper):
    """Custom logits processor."""
    @torch.inference_mode()
    def sampling(self, logits: torch.Tensor):
        """sampling."""
        
        sampling_inputs = self.sampling_inputs
        
        # 不同的top_k值，会调用不同的采样方法。
        if sampling_inputs.max_top_k == 1:
            return logits.argmax(-1)
        else:
            # sort logits is too slow. and we only need topk logits
            max_topk = sampling_inputs.max_top_k
            if max_topk <= 0:
                scores, indices = logits.sort(1, descending=True)
            else:
                scores, indices = logits.topk(max_topk, dim=1)
            # 发生错误的分支逻辑
            return __random_sampling(scores, indices)


```
执行scores.softmax(1)，将inf变成了nan，导致崩溃。
```bash
(Pdb) p logits
tensor([[-inf, -inf, -inf,  ..., inf, -inf, inf]], device='npu:0',
       dtype=torch.float16)
(Pdb) p logits.shape
torch.Size([1, 64007])
(Pdb) p logits.argmax(-1)
tensor([7], device='npu:0')
(Pdb) p logits[0][7]
tensor(inf, device='npu:0', dtype=torch.float16)
(Pdb) p logits.topk(40, dim=1)
torch.return_types.topk(
values=tensor([[inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf,
         inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf]],
       device='npu:0', dtype=torch.float16),
indices=tensor([[ 7, 18, 32, 35, 37, 38, 41, 47, 48, 49, 50, 52, 53, 54, 55, 56, 58, 59,
         61, 62, 65, 67, 68, 70, 71, 73, 74, 76, 77, 78, 79, 80, 81, 82, 83, 84,
         85, 86, 88, 89]], device='npu:0'))
(Pdb) p scores.softmax(1)
tensor([[nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
         nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]],
       device='npu:0', dtype=torch.float16)
```


由于是使用的pytorch的eager模式，采用dlinfer调用的Ascend npu推理。跟踪到了下面的算子multinomial_sampling崩溃了。

```python
~/lmdeploy/pytorch/engine/logits_process.py
def _multinomial_sampling(scores: torch.Tensor,
                          seeds: torch.LongTensor,
                          offsets: torch.LongTensor,
                          indices: torch.LongTensor = None):
    """sampling."""
    from lmdeploy.pytorch.nn.multinomial_sampling import multinomial_sampling
    return multinomial_sampling(scores, seeds, offsets, indices)
```

根据不同的backend，比如Ascend npu、GPU等，multinomial_sampling的实现不同。
```python
~/lmdeploy/pytorch/nn/multinomial_sampling.py
import torch

from ..backends import OpType, get_backend


def multinomial_sampling(scores: torch.Tensor,
                         seeds: torch.LongTensor,
                         offsets: torch.LongTensor,
                         indices: torch.Tensor = None):
    """multinomial sampling op."""
    impl_builder = get_backend().get_layer_impl_builder(
        OpType.MultinomialSampling)
    return impl_builder.build().forward(scores, seeds, offsets, indices)

```

搜索OpType.MultinomialSampling，可以找到default实现。
```python
~/lmdeploy/pytorch/backends/default/multinomial_sampling.py
```

```
(Pdb) p scores
tensor([[nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]],device='npu:0', dtype=torch.float16)
(Pdb) p seeds
tensor([2281162171], device='npu:0')
(Pdb) p offsets
tensor([0], device='npu:0')
(Pdb) p indices
tensor([[ 7, 18, 32, 35, 37, 38, 41, 47, 48, 49, 50, 52, 53, 54, 55, 56, 58, 59,
         61, 62, 65, 67, 68, 70, 71, 73, 74, 76, 77, 78, 79, 80, 81, 82, 83, 84,
         85, 86, 88, 89]], device='npu:0')
```

写个例子测试下：
参考https://github.com/Ascend/pytorch
```python

import torch
import torch_npu

# 创建一个包含NaN的张量
nan_tensor = torch.full((1, 40), float('nan'), dtype=torch.float16)

# 将张量移动到NPU设备
nan_tensor = nan_tensor.npu()

print(nan_tensor)

sampled_index = torch.multinomial(nan_tensor,
                                num_samples=1,
                                replacement=True)
print(sampled_index)

```
上面的例子就是复现问题的最小demo。

```
torch-npu 2.1.0.post6
torch 2.1.0
```

另一种修改方式是，判断前面如果是inf，就不能做softmax。

或者将inf换成一个比较大的数，比如1e5。
```python
# 替换 `inf` 为一个大数值, 例如：最大有限值
tensor = torch.where(torch.isinf(tensor), torch.tensor(1e10), tensor)
```

```python
import torch

# 示例 scores 张量
scores = torch.tensor([[1.0, float('inf'), 3.0], [float('-inf'), 1.0, 2.0]], dtype=torch.float32)

# 检查并处理 inf 值
if torch.isinf(scores).any():
    # 获取 scores 的数据类型
    dtype = scores.dtype
    
    # 根据数据类型选择替换 inf 的值
    if dtype in [torch.float16, torch.float32, torch.float64]:
        max_finite_value = torch.finfo(dtype).max
        min_finite_value = torch.finfo(dtype).min
    elif dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
        max_finite_value = torch.iinfo(dtype).max
        min_finite_value = torch.iinfo(dtype).min
    else:
        raise TypeError("Unsupported data type")
    
    # 替换正 inf 为最大有限值，负 inf 为最小有限值
    scores = torch.where(scores == float('inf'), torch.tensor(max_finite_value, dtype=dtype), scores)
    scores = torch.where(scores == float('-inf'), torch.tensor(min_finite_value, dtype=dtype), scores)

# 计算 softmax
softmax_scores = scores.softmax(dim=1)

print(softmax_scores)
```

```python
# if score has inf, replace it with max or min finite value, then do softmax
def _softmax_scores(scores: torch.Tensor):
    """softmax scores."""
    # 检查并处理 inf 值
    if torch.isinf(scores).any():
        # 获取 scores 的数据类型
        dtype = scores.dtype
        
        # 根据数据类型选择替换 inf 的值
        if dtype in [torch.float16, torch.float32, torch.float64]:
            max_finite_value = torch.finfo(dtype).max
            min_finite_value = torch.finfo(dtype).min
        elif dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            max_finite_value = torch.iinfo(dtype).max
            min_finite_value = torch.iinfo(dtype).min
        else:
            raise TypeError("Unsupported data type")

        # 获取张量所在的设备
        device = scores.device
        
        # 替换正 inf 为最大有限值，负 inf 为最小有限值
        scores = torch.where(scores == float('inf'), torch.tensor(max_finite_value, dtype=dtype, device=device), scores)
        scores = torch.where(scores == float('-inf'), torch.tensor(min_finite_value, dtype=dtype, device=device), scores)
    softmax_scores = scores.softmax(dim=1)
    return softmax_scores
```

example:
```python
import torch
import torch_npu

#inf_tensor = torch.full((1, 10), float('inf'), dtype=torch.float16)
# or
inf_tensor = torch.tensor([[1, 2, 3, 4, float('inf')]], dtype=torch.float16)

inf_tensor = inf_tensor.npu()
print(inf_tensor)

#res_nan = inf_tensor.softmax(1)
#print(res_nan)

# fix buy replacing inf with max value
res = _softmax_scores(nan_tensor)
print(res)

# error occurred
#sampled_index = torch.multinomial(res_nan,
#                                num_samples=1,
#                                replacement=True)
#print(sampled_index)
```

# 参数支持top_k设置
可以看到 api不支持top_k的设置，需要修改源代码。

```python
~/lmdeploy/serve/openai/api_client.py
def chat_completions_v1(self,
                        model: str,
                        messages: Union[str, List[Dict[str, str]]],
                        temperature: Optional[float] = 0.7,
                        top_p: Optional[float] = 1.0,
                        logprobs: Optional[bool] = False,
                        top_logprobs: Optional[int] = 0,
                        n: Optional[int] = 1,
                        max_tokens: Optional[int] = None,
                        stop: Optional[Union[str, List[str]]] = None,
                        stream: Optional[bool] = False,
                        presence_penalty: Optional[float] = 0.0,
                        frequency_penalty: Optional[float] = 0.0,
                        user: Optional[str] = None,
                        repetition_penalty: Optional[float] = 1.0,
                        session_id: Optional[int] = -1,
                        ignore_eos: Optional[bool] = False,
                        skip_special_tokens: Optional[bool] = True,
                        **kwargs):
    """Chat completion v1.

```

从上面的代码分析，可以知道do_sample = False, 就是采用greedy策略采样。等价于设置下面的参数：
```
gen_config.top_k = 1
# avoid unnecessary process
gen_config.temperature = 1.0
gen_config.repetition_penalty = 1.0
```

completions_v1 接口目前支持top_k的设置，但是和chat_completions_v1还是有区别的。直接替换调用，并不能成功。
+ 参数messages需要替换成prompt

还有一种改法，将temperature设置为一个非0值，temperature设置为0时，下面的temperature=1e-6，所以scores会变成无穷大。

```python

def _process_temperature_(scores: torch.Tensor, temperature: torch.Tensor):
    """process temperature."""
    temperature = temperature.to(scores.dtype)
    scores.div_(temperature[:, None])
    return scores
```