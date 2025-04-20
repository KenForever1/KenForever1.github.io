---
title: sglang推理使用之环境配置
date: 2024-12-17
authors: [KenForever1]
categories: 
  - LLM推理
labels: [LLM推理]
comments: true
---
本文记录了sglang推理使用之环境配置。

<!-- more -->

### Docker环境
https://docs.sglang.ai/backend/send_request.html

镜像网站：https://docker.aityp.com/i/search?search=nvidia%2Fcuda

```
docker run --gpus all \
    --privileged=true --network=host --workdir=/workspace  \
    -v /ssd3/swift/:/workspace --name sglang-dev-test \
    --shm-size 32g \
    --ipc=host \
    -it \
    swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/lmsysorg/sglang /bin/bash

docker run --gpus all \
    --privileged=true --network=host --workdir=/workspace  \
    -v /ssd3/swift/:/workspace --name cuda-dev-test \
    --shm-size 32g \
    --ipc=host \
    -it swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/nvidia/cuda:12.2.2-devel-ubuntu22.04 /bin/bash
```

```
python3 -m sglang.launch_server --model-path /workspace/models--OpenGVLab--InternVL2-8B/snapshots/c527cd3717f4bb8339002b342bc9476ec0485004 --trust-remote-code --served-model-name OpenGVLab--InternVL2-8B

python3 -m sglang.launch_server --model-path /workspace/models--Qwen--Qwen2-VL-2B-Instruct/snapshots/aca78372505e6cb469c4fa6a35c60265b00ff5a4 --trust-remote-code --served-model-name Qwen--Qwen2-VL-2B
```
### 调试

sglang是多进程的，不能直接用pdb.set_trace()调试，https://github.com/sgl-project/sglang/discussions/2480。

### 推理报错

推理intervl报错：
```
[2025-03-18 03:29:57 TP0] Traceback (most recent call last):
  File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 1134, in run_scheduler_process
    scheduler = Scheduler(server_args, port_args, gpu_id, tp_rank, dp_rank)
  File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 123, in __init__
    self.model_config = ModelConfig(
  File "/sgl-workspace/sglang/python/sglang/srt/configs/model_config.py", line 82, in __init__
    self.hf_text_config.hidden_size // self.hf_text_config.num_attention_heads,
  File "/usr/local/lib/python3.10/dist-packages/transformers/configuration_utils.py", line 205, in __getattribute__
    return super().__getattribute__(key)
AttributeError: 'InternVLChatConfig' object has no attribute 'hidden_size'

Killed
```

internvl2暂不支持。从ModelConfig类来看，不能正确处理internvl2的参数。
https://github.com/sgl-project/sglang/issues/1042

```
sglang                            0.3.4.post2          /sgl-workspace/sglang/python
```

### 如何升级到最新版

```
python3 -m pip install sglang==0.4.4 -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
python3 -m pip install sgl-kernel -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
#查看torch版本
python3 -m pip list | grep torch
#https://github.com/flashinfer-ai/flashinfer/issues/794
pip install flashinfer-python -i https://flashinfer.ai/whl/cu121/torch2.4/ 

```