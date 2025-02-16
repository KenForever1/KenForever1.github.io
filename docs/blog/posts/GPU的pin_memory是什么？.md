---
title: GPU的pin_memory是什么？
date: 2024-12-09
authors: [KenForever1]
categories: 
  - LLM推理
labels: []
comments: true
---

## gpu的pin_memory

pin_memory就是在RAM上固定了一块内存，这个内存范围是被锁住的。pin这个单词很形象，很像rust中pin含义，用钉子把钉住，这个内存就不会释放，是安全的意思。GPU在传递数据的时候，就可以用DMA的方式，高效传输数据。否则，普通的cpu_memory，就会swap掉，然后访问的时候缺页中断，这样速度肯定就慢了很多。

<!-- more -->

> CUDA Driver checks, if the memory range is locked or not and then it will use a different codepath. Locked memory is stored in the physical memory (RAM), so device can fetch it w/o help from CPU (DMA, aka Async copy; device only need list of physical pages). Not-locked memory can generate a page fault on access, and it is stored not only in memory (e.g. it can be in swap), so driver need to access every page of non-locked memory, copy it into pinned buffer and pass it to DMA (Syncronious, page-by-page copy).

参考：[why-is-cuda-pinned-memory-so-fast](https://stackoverflow.com/questions/5736968/why-is-cuda-pinned-memory-so-fast)


## 推理库中的使用

### vllm中相关code

在vllm中就有根据GPU平台和环境的不同，判断pin_memory是否可用。
比如：Pinning memory in WSL is not supported.
```python
@lru_cache(maxsize=None)
def is_pin_memory_available() -> bool:

    if in_wsl():
        # Pinning memory in WSL is not supported.
        # https://docs.nvidia.com/cuda/wsl-user-guide/index.html#known-limitations-for-linux-cuda-applications
        print_warning_once("Using 'pin_memory=False' as WSL is detected. "
                           "This may slow down the performance.")
        return False
    elif current_platform.is_xpu():
        print_warning_once("Pin memory is not supported on XPU.")
        return False
    elif current_platform.is_neuron():
        print_warning_once("Pin memory is not supported on Neuron.")
        return False
    elif current_platform.is_hpu():
        print_warning_once("Pin memory is not supported on HPU.")
        return False
    elif current_platform.is_cpu() or current_platform.is_openvino():
        return False
    return True
```

https://github.com/vllm-project/vllm/issues/188

### 在lmdeploy使用

在lmdeploy中，同样有关于pin_memory的判断。
```
lmdeploy-0.6.1.2/lmdeploy/pytorch/engine/cache_engine.py
```
