---
title: VLLM推理框架中的sleep_mode如何实现
date: 2025-07-06
authors: [KenForever1]
categories: 
  - LLM推理
labels: [LLM推理]
pin: true
comments: true
---

[TOC]

## vllm sleep_model 简介

前文[torch_memory_saver 高性能 CUDA 内存管理工具实现](https://zhuanlan.zhihu.com/p/1919901725592093271 "torch_memory_saver 高性能 CUDA 内存管理工具实现")，介绍了 sglang 中利用了该库将保存 kv_cache 和权重的显存释放出来。

在 VLLM 中也有同样的功能实现，在 VLLM 中的直接应用是“sleep mode”。将模型权重从显存（或者 NPU 内存中）卸载，并丢弃其中的 KV 缓存。

<!-- more -->

两种 level 不同策略：

```bash
Level 1 Sleep

Action: Offloads model weights and discards the KV cache.

Memory: Model weights are moved to CPU memory; KV cache is forgotten.

Use Case: Suitable when reusing the same model later.

Note: Ensure sufficient CPU memory is available to hold the model weights.

Level 2 Sleep

Action: Discards both model weights and KV cache.

Memory: The content of both the model weights and kv cache is forgotten.

Use Case: Ideal when switching to a different model or updating the current one.
```

## 如何使用 sleep 和 wake_up?

在 VLLM 中通过 CuMemAllocator.get_instance()使用 sleep 和 wake_up 方法，实现释放显存和恢复显存。

```python
def load_model(self) -> None:
    if self.vllm_config.model_config.enable_sleep_mode:
        allocator = CuMemAllocator.get_instance()
        assert allocator.get_current_usage() == 0, (
            "Sleep mode can only be "
            "used for one instance per process.")
        context = allocator.use_memory_pool(tag="weights")
    else:
        from contextlib import nullcontext
        context = nullcontext()
    with context:
        self.model_runner.load_model()

class Worker():

    def sleep(self, level: int = 1) -> None:
        free_bytes_before_sleep = torch.cuda.mem_get_info()[0]

        allocator = CuMemAllocator.get_instance()
        allocator.sleep(offload_tags=("weights", ) if level == 1 else tuple())
        free_bytes_after_sleep, total = torch.cuda.mem_get_info()
        freed_bytes = free_bytes_after_sleep - free_bytes_before_sleep
        used_bytes = total - free_bytes_after_sleep
        assert freed_bytes >= 0, "Memory usage increased after sleeping."
        logger.info(
            "Sleep mode freed %.2f GiB memory, "
            "%.2f GiB memory is still in use.", freed_bytes / GiB_bytes,
            used_bytes / GiB_bytes)

    def wake_up(self, tags: Optional[list[str]] = None) -> None:
        allocator = CuMemAllocator.get_instance()
        allocator.wake_up(tags)
```

当调用 sleep 方法时，所有具有指定标签的张量都将被卸载(offload)到 CPU 内存中，其余张量将被丢弃(discard)。
当我们调用 wake_up 时，之前卸载的所有张量都将被加载回 GPU 内存，其余张量为空。

## sleep 和 wake_up 实现

cuda 实现针对 NVIDIA GPU, npu 实现针对华为 Ascend NPU。

### cuda 实现

cuda 实现采用 [CuMemAllocator 类实现](https://github.com/vllm-project/vllm/blob/main/vllm/device_allocator/cumem.py#L106-L107 "CuMemAllocator 类实现")。pointer_to_data 中保存了 ptr 和元数据。

sleep 方法就是将offload_tags中的数据先offload到cpu，即调用 cudaMemcpy DeviceToHost 拷贝到 cpu 内存中。并且记录下 cpu_backup_tensor，供 wakeup 的时候，将 cpu 内存中的数据拷贝回 gpu 内存中。

对于没有在 offload_tags 中的数据，sleep 方法会直接丢弃数据，释放显存。调用 unmap_and_release实现。主要是保持用户使用的虚拟地址不变，暂停后释放显存，恢复重新分配显存，绑定到虚拟地址上。

```python
def sleep(
            self,
            offload_tags: Optional[Union[tuple[str, ...],
                                         str]] = None) -> None:
    if offload_tags is None:
        # by default, allocated tensors are offloaded
        # when the allocator sleeps
        offload_tags = (CuMemAllocator.default_tag, )
    elif isinstance(offload_tags, str):
        offload_tags = (offload_tags, )

    assert isinstance(offload_tags, tuple)

    for ptr, data in self.pointer_to_data.items():
        handle = data.handle
        if data.tag in offload_tags:
            size_in_bytes = handle[1]
            cpu_backup_tensor = torch.empty(
                size_in_bytes,
                dtype=torch.uint8,
                device='cpu',
                pin_memory=is_pin_memory_available())
            cpu_ptr = cpu_backup_tensor.data_ptr()
            libcudart.cudaMemcpy(cpu_ptr, ptr, size_in_bytes)
            data.cpu_backup_tensor = cpu_backup_tensor
        unmap_and_release(handle)

    gc.collect()
    torch.cuda.empty_cache()
```

wake_up 方法则是调用 create_and_map(handle)实现重新分配显存，并绑定到虚拟地址上。将 offload 的数据从 cpu 拷贝到 gpu 中。

```python
def wake_up(self, tags: Optional[list[str]] = None) -> None:
    for ptr, data in self.pointer_to_data.items():
        if tags is None or data.tag in tags:
            handle = data.handle
            create_and_map(handle)
            if data.cpu_backup_tensor is not None:
                cpu_backup_tensor = data.cpu_backup_tensor
                if cpu_backup_tensor is not None:
                    size_in_bytes = cpu_backup_tensor.numel(
                    ) * cpu_backup_tensor.element_size()
                    cpu_ptr = cpu_backup_tensor.data_ptr()
                    libcudart.cudaMemcpy(ptr, cpu_ptr, size_in_bytes)
                    data.cpu_backup_tensor = None
```

pointer_to_data 元数据：

```python
# py_device, py_alignedSize, py_d_mem (保存的用户使用的指针), py_p_memHandle（内存mmap的文件描述符）
HandleType = tuple[int, int, int, int]

@dataclasses.dataclass
class AllocationData:
    handle: HandleType
    tag: str
    cpu_backup_tensor: Optional[torch.Tensor] = None # offload到cpu的tensor
```

python 调用底层 c++实现，和**torch_memory_saver**原理完全一致。

```python
def create_and_map(allocation_handle: HandleType) -> None:
    python_create_and_map(*allocation_handle)


def unmap_and_release(allocation_handle: HandleType) -> None:
    python_unmap_and_release(*allocation_handle)
```

c++实现查看[cumem_allocator.cpp](https://github.com/vllm-project/vllm/blob/main/csrc/cumem_allocator.cpp#L297 "cumem_allocator.cpp")。

### Ascend NPU 的实现

主要通过[CaMemAllocator 类实现](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/device_allocator/camem.py "CaMemAllocator 类的实现")。python 调用底层 c++实现。和 cuda 实现的区别就是 sleep 中的 unmap_and_release 函数、wake_up 中的 create_and_map 函数实现不一样。

具体实现可以参考[camem_allocator.cpp](https://github.com/vllm-project/vllm-ascend/blob/main/csrc/camem_allocator.cpp "camem_allocator.cpp")，使用 Ascend NPU 的 aclrt 实现了保存指针 aclrtReserveMemAddress、内存映射等 API。

```c++
void create_and_map(unsigned long long device, ssize_t size, void* d_mem,
                    aclrtDrvMemHandle* p_memHandle) {
  ensure_context(device);
  // Define memory allocation properties
  aclrtPhysicalMemProp prop = {};
  prop.handleType = ACL_MEM_HANDLE_TYPE_NONE ;
  prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
  prop.memAttr = ACL_HBM_MEM_HUGE;
  prop.location.id = device;
  prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
  prop.reserve = 0;

  // Allocate memory using aclrtMallocPhysical
  aclError error_code = aclrtMallocPhysical(p_memHandle, size, &prop, 0);
  if (error_code != 0) {
    std::cerr << "acl Error, code: " << error_code << " at " << __FILE__ << ":" \
            << __LINE__ << std::endl;
    return;
  }
  error_code = aclrtMapMem(d_mem, size, 0, *p_memHandle, 0);
  if (error_code != 0) {
    std::cerr << "acl Error, code: " << error_code << " at " << __FILE__ << ":" \
            << __LINE__ << std::endl;
    return;
  }
}

void unmap_and_release(unsigned long long device, ssize_t size,
                       void* d_mem,
                       aclrtDrvMemHandle* p_memHandle) {
  // std::cout << "unmap_and_release: device=" << device << ", size=" << size <<
  // ", d_mem=" << d_mem << ", p_memHandle=" << p_memHandle << std::endl;
  ensure_context(device);
  aclError error_code = aclrtUnmapMem(d_mem);
  if (error_code != 0) {
    std::cerr << "acl Error, code: " << error_code << " at " << __FILE__ << ":" \
            << __LINE__ << std::endl;
    return;
  }
  error_code = aclrtFreePhysical(*p_memHandle);
  if (error_code != 0) {
    std::cerr << "acl Error, code: " << error_code << " at " << __FILE__ << ":" \
            << __LINE__ << std::endl;
    return;
  }
}
```

通过 pytorch 插件注册了 NPU 的自定义 my_malloc 和 my_free 函数。

```python
def get_pluggable_allocator(
    python_malloc_fn: Callable[[tuple[int, int, int, int]], None],
    python_free_func: Callable[[int], tuple[int, int, int, int]]
) -> torch.npu.memory.NPUPluggableAllocator:
    init_module(python_malloc_fn, python_free_func)
    new_alloc = torch.npu.memory.NPUPluggableAllocator(lib_name, 'my_malloc',
                                                       'my_free')
    return new_alloc

@contextmanager
def use_memory_pool_with_allocator(
        python_malloc_fn: Callable[[tuple[int, int, int, int]], None],
        python_free_func: Callable[[int], tuple[int, int, int, int]]):
    new_alloc = get_pluggable_allocator(python_malloc_fn, python_free_func)
    mem_pool = torch.npu.memory.MemPool(new_alloc._allocator)
    with torch.npu.memory.use_mem_pool(mem_pool):
        yield mem_pool, new_alloc
```

torch_memory_saver 的实现和 vllm 中 sleep_mode 用到的 CuMemAllocator 实现是一样的，除此之外 vllm 针对 npu 实现了 CaMemAllocator，调用的 aclrt 实现。
vllm 的 sleep_mode 除了将 gpu 内存释放，还实现了根据 offload_tags offload 到 cpu（主要是权重）。
