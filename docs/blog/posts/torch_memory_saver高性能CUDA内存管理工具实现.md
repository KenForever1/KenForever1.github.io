---
title: torch_memory_saver高性能CUDA内存管理工具实现
date: 2025-06-21
authors: [KenForever1]
categories: 
  - LLM推理
labels: []
comments: true
---

[torch_memory_saver](https://github.com/fzyzcjy/torch_memory_saver/tree/master
)是一个开源的高性能CUDA内存管理工具，主要功能是允许暂停和恢复PyTorch张量的CUDA内存占用。保持用户使用的虚拟地址不变，暂停后释放显存，恢复重新分配显存，绑定到虚拟地址上。

本文会介绍核心原理，以及拦截CUDA runtime API的实现。你还可以看到如何实现一个python c++扩展。在sglang大模型推理库中也有使用到这个torch_memory_saver库。

<!-- more -->

## 核心原理和使用示例

内存暂停/恢复功能的核心实现原理:
使用CUDA虚拟内存管理API（cuMemCreate/cuMemMap等）替代常规cudaMalloc, 通过LD_PRELOAD拦截cudaMalloc/cudaFree调用, 在特定region内（例如：在python中可以用with memory_saver.region()语法）分配的内存会被特殊管理。

使用例子：
```python
import torch_memory_saver

memory_saver = torch_memory_saver.memory_saver

# 1. For tensors that wants to be paused, create them within `region`
with memory_saver.region():
    pauseable_tensor = torch.full((1_000_000_000,), 100, dtype=torch.uint8, device='cuda')

# 2. After `pause`, CUDA memory is released for those tensors.
# For example, check `nvidia-smi`'s memory usage to verify.
memory_saver.pause()

# 3. After `resume`, CUDA memory is re-occupied for those tensors.
memory_saver.resume()
```

## 暂停和恢复实现原理

下面介绍pause和resume的实现方式：
```cpp
void pause() {
    // 遍历所有已分配内存
    for (auto it = allocation_metadata_.begin(); ...) {
        // 1. 取消内存映射 (cuMemUnmap)
        // 2. 释放底层内存句柄 (cuMemRelease)
        // 保留虚拟地址空间不释放
        CURESULT_CHECK(cuMemUnmap((CUdeviceptr) ptr, metadata.size));
        CURESULT_CHECK(cuMemRelease(metadata.allocHandle));
    }
}
```

```cpp
void resume() {
    // 遍历所有已分配内存
    for (auto it = allocation_metadata_.begin(); ...) {
        // 1. 创建新的内存句柄 (cuMemCreate)
        // 2. 重新映射到原来的虚拟地址 (cuMemMap)
        // 3. 设置访问权限 (cuMemSetAccess)

        CUmemGenericAllocationHandle newAllocHandle;
        CUDAUtils::cu_mem_create(&newAllocHandle, metadata.size, metadata.device);
        CURESULT_CHECK(cuMemMap((CUdeviceptr) ptr, metadata.size, 0, newAllocHandle, 0));
        CUDAUtils::cu_mem_set_access(ptr, metadata.size, metadata.device);
        metadata.allocHandle = newAllocHandle;
    }
}
```
使用CUDA虚拟内存API保持虚拟地址不变，因此暂停时只释放物理内存，保留虚拟地址空间。恢复时重新分配物理内存并映射到原虚拟地址。

+ 调用cuMemAddressReserve保留虚拟地址空间；
+ 通过cuMemCreate分配物理内存；
+ 使用cuMemMap将物理内存映射到虚拟地址；
+ 最后通过cuMemSetAccess设置访问权限

这样应用程序无需修改指针引用，特别适合需要临时释放显存的大模型场景。

再看看malloc和free的实现：
```c++
cudaError_t malloc(void **ptr, size_t size, const std::string& tag) {
    CUdevice device;
    CURESULT_CHECK(cuCtxGetDevice(&device));

    CUmemGenericAllocationHandle allocHandle;
    CUDAUtils::cu_mem_create(&allocHandle, size, device);
    // ptr就是保留重复使用的虚拟地址
    CURESULT_CHECK(cuMemAddressReserve((CUdeviceptr *) ptr, size, 0, 0, 0));
    CURESULT_CHECK(cuMemMap((CUdeviceptr) * ptr, size, 0, allocHandle, 0));
    CUDAUtils::cu_mem_set_access(*ptr, size, device);

    {
        const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
        allocation_metadata_.emplace(*ptr, _AllocationMetadata{size, device, allocHandle, tag});
    }
}

cudaError_t free(void *ptr) {
    _AllocationMetadata metadata;
    {
        const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);
        SIMPLE_CHECK(allocation_metadata_.count(ptr), "Trying to free a pointer not allocated here");
        metadata = allocation_metadata_[ptr];
        allocation_metadata_.erase(ptr);
    }

    CURESULT_CHECK(cuMemUnmap((CUdeviceptr) ptr, metadata.size));
    CURESULT_CHECK(cuMemRelease(metadata.allocHandle));
    CURESULT_CHECK(cuMemAddressFree((CUdeviceptr) ptr, metadata.size));
}
```

通过allocation_metadata_保存了虚拟指针ptr和每次分配的metadata，metadata包括了真实使用的内存allocHandle和size。

在pause的时候释放allocHandle，但是不释放ptr。在assume的时候申请新的allocHandle，绑定ptr和allocHandle的关系，即更新allocation_metadata_保存的metadata信息。

## 通过Region进行管理

通过Region进行管理，不在Region内的内存不受影响，正常使用CudaMalloc/CudaFree。在特定Region内分配的内存会被特殊管理，暂停时只释放物理内存，保留虚拟地址空间，恢复时重新分配物理内存并映射到原虚拟地址。

对应python：
```python
with memory_saver.region():
    x = torch.full((1_000_000_000,), 100, dtype=torch.uint8, device='cuda')
```

Region的实现：

```cpp
namespace RegionManager {
    static thread_local bool is_interesting_region_ = false;

    void enter() {
#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] tms_region_enter" << std::endl;
#endif
        is_interesting_region_ = true;
    }

    void leave() {
#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] tms_region_leave" << std::endl;
#endif
        is_interesting_region_ = false;
    }

    bool is_interesting_region() {
        return is_interesting_region_;
    }
}
```
is_interesting_region函数，会在下节拦截中cudaMalloc/cudaFree调用中区分不同的调用实现。

## LD_PRELOAD拦截cudaMalloc/cudaFree调用

pytorch python库做了大量的封装，更加友好使用。比如创建一个tensor只需要下面一句，指定datatype、shape、device为cuda。实际上底层会调用cudaMalloc/cudaFree（cudaruntime的api）去分配和释放显存。
```
x = torch.full((1_000_000_000,), 100, dtype=torch.uint8, device='cuda')
```
为了能够使用上我们自定义的分配逻辑，那么就需要拦截cudaruntime中实现的cudaMalloc和cudaFree。比如在Region区域内调用自定义分配释放逻辑，不在Region内的内存，正常调用cudaruntime的CudaMalloc/CudaFree。

一起看看如何拦截！

自定义cudaMalloc和cudaFree函数。分条件调用不同的逻辑。
```c++
cudaError_t cudaMalloc(void **ptr, size_t size) {
    if (RegionManager::is_interesting_region()) {
        return TorchMemorySaver::instance().malloc(ptr, size, RegionManager::get_current_tag());
    } else {
        return APIForwarder::call_real_cuda_malloc(ptr, size);
    }
}

cudaError_t cudaFree(void *ptr) {
    if (RegionManager::is_interesting_region()) {
        return TorchMemorySaver::instance().free(ptr);
    } else {
        return APIForwarder::call_real_cuda_free(ptr);
    }
}
```
call_real_cuda_malloc和call_real_cuda_free就是调用cudaruntime的标准实现。

可以看到，和我们上篇将拦截的文章一样，用到了dlsym RTLD_NEXT方式和LD_PRELOAD方式获取cudaruntime的实现。
```c++
    static cudaError_t call_real_cuda_malloc(void **ptr, size_t size) {
        if (C10_UNLIKELY(nullptr == real_cudaMalloc)) {
            real_cudaMalloc = (CudaMallocFunc) check_dlsym(dlsym(RTLD_NEXT, "cudaMalloc"));
        }
        cudaError_t ret = real_cudaMalloc(ptr, size);
        return ret;
    }
```

## 额外学习到的知识

除此之外，由于核心逻辑是c++实现的，因此你还可以通过本项目学会如何实现python c/c++扩展（setup ext_modules）。

和pytorch一样，最终使用torch_memory_saver是python包形式。如何通过（from contextlib import contextmanager）自定义python with语法糖。如何调用封装c++扩展。形成一个用户友好的python库。