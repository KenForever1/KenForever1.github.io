---
title: 给llama-cpp提交一些PR
date: 2024-11-17
authors: [KenForever1]
categories: 
  - 每周杂谈
labels: []
comments: true
---

## llama-cpp中计算用于线性代数计算的线程数

### 看看llama-cpp中的实现

一般我们是不是想着将threads设置越大越好，最大设置为cpu cores的个数。这里的用法告诉了我们并不是这样的，因为超线程（hyper-threads）对线性代数（linear algebra）计算没有用。线性代数计算，比如深度学习中的矩阵乘法等，有一些著名的BLAS库，用于加速计算。

<!-- more -->

```cpp
/**
 * Returns number of CPUs on system that are useful for math.
 */
int32_t cpu_get_num_math() {
#if defined(__x86_64__) && defined(__linux__) && !defined(__ANDROID__)
    //_SC_NPROCESSORS_ONLN 返回系统中实际可用的核心数
    int n_cpu = sysconf(_SC_NPROCESSORS_ONLN);
    if (n_cpu < 1) {
        return cpu_get_num_physical_cores();
    }
    if (is_hybrid_cpu()) {
        cpu_set_t affinity;
        if (!pthread_getaffinity_np(pthread_self(), sizeof(affinity), &affinity)) {
            int result = cpu_count_math_cpus(n_cpu);
            pthread_setaffinity_np(pthread_self(), sizeof(affinity), &affinity);
            if (result > 0) {
                return result;
            }
        }
    }
#endif
    return cpu_get_num_physical_cores();
}

static int cpu_count_math_cpus(int n_cpu) {
    int result = 0;
    for (int cpu = 0; cpu < n_cpu; ++cpu) {
        if (pin_cpu(cpu)) {
            return -1;
        }
        if (is_running_on_efficiency_core()) {
            continue; // efficiency cores harm lockstep threading
        }
        ++cpu; // hyperthreading isn't useful for linear algebra // 超线程对线性代数没有用，算出就是physic cores / 2
        ++result;
    }
    return result;
}

```

```cpp
int32_t cpu_get_num_physical_cores() {
    std::cout << "call cpu_get_num_physical_cores" << std::endl;
#ifdef __linux__
    // enumerate the set of thread siblings, num entries is num cores
    std::unordered_set<std::string> siblings;
    for (uint32_t cpu=0; cpu < UINT32_MAX; ++cpu) {
        std::ifstream thread_siblings("/sys/devices/system/cpu/cpu"
            + std::to_string(cpu) + "/topology/thread_siblings");
        if (!thread_siblings.is_open()) {
            break; // no more cpus
        }
        std::string line;
        if (std::getline(thread_siblings, line)) {
            std::cout << "line :"  << line << std::endl; 
            siblings.insert(line);
        }
    }
    if (!siblings.empty()) {
        return static_cast<int32_t>(siblings.size());
    }
#endif
    unsigned int n_threads = std::thread::hardware_concurrency();
    return n_threads > 0 ? (n_threads <= 4 ? n_threads : n_threads / 2) : 4;
}

```

### 什么是hybrid_cpu？
>   Interesting Question

    A hybrid CPU refers to a processor that combines two different types of processing units in a single Chip:

    a conventional CPU (Central Processing Unit) and a specialized accelerator, such as a GPU(Graphics Processing Unit) or an FPGA (Field Programmable Gate Array)

    Examples Intel’s Lakefield and Alder Lake Processor.

    Another Example of Hybrid CPU AMD’s Ryzen APUs.
https://www.quora.com/What-is-a-hybrid-CPU

### 关于thread_siblings 和 cpu_cores?
通过cat /proc/cpuinfo, flags中可以查看是否支持HT（超线程）。如果thread_siblings个数等于cpu_cores就不是超线程。
>   On Linux I think you can read /proc/cpuinfo, but after that you have to do a bit of     thinking to see whether we have multicore cpu, or HT enabled cpu etc.
First, flags will give you supported features, and ht there will indicate hyperthreading support.
Then you have to check whether sibling count matches core count on each CPU, so look for cpu id, and deduct from there. (So if sibling count matches core count -> no HT) 

参考：[hyper-threading-by-which-test-can-i-check-if-it-is-enabled-or-disabled](https://stackoverflow.com/questions/18863646/hyper-threading-by-which-test-can-i-check-if-it-is-enabled-or-disabled)


### 关于hyperthreading，如何在compute中决定thread num？

https://www.juliabloggers.com/tag/hyperthreading/

> Because BLAS (and LINPACK, Linear Algebra Package, for other linear algebra routines) is so optimized, people say you should always make sure that it knows exactly how many “real” processors it has to work with. So in my case, with a Core i7 with 4 physical cores and 4 from hyperthreading, forget the hyperthreading and thus there are 4. With the FX8350, there are only 4 processors for doing math, so 4 threads. Check to make sure this is best.

## llama-cpp 编译

解决编译中的openmp undefined问题：
```bash
CMAKE_ARGS="-DCMAKE_CXX_FLAGS=-fopenmp" pip install llama-cpp-python
https://github.com/abetlen/llama-cpp-python/issues/1573
```

如何编译动态链接库，build shared lib:
```bash
cmake -DLLAMA_STATIC=Off -DBUILD_SHARED_LIBS=On -B build -S .
cmake --build build
```


## 配置本地python包搜索路径

在本地调试python项目时很实用, clone项目不用安装，直接设置一个本地包搜索路径。或者采用pip install -e . 安装，这样之后安装一个链接,而不会将项目安装到python site-packages下。这里有个需要注意的，如果采用pip install -e .安装以后, 如果项目文件在你映射的目录中，你打包一个docker分享给人家不会包括整个包。需要pip install ., 不要-e。

如何查看site-packages目录：

```bash
$ python
>> import toml # 任意安装了的包
>> print(toml.__file__)

```

写入本地包的路径，自动搜索本地包了。

```bash
touch /home/ken/miniconda3/lib/python3.10/site-packages/mypkpath.pth
```

参考：[python-package-development](https://www.liangye.site/2018/05/11/python-package-development/)

## c++ cmake add_library设置不同选项

cmake add_library如何不设置shared、static等，就会根据BUILD_SHARED_LIBS设置shared或者static。

```
add_library(${TARGET} 
    xxx.cpp)

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(${TARGET} PRIVATE LLAMA_BUILD)
    target_compile_definitions(${TARGET} PUBLIC  LLAMA_SHARED)
endif()
```

## other

+ [spawning-processes-on-linux](https://maelstrom-software.com/blog/spawning-processes-on-linux)

+ OpenBMB开源的MiniCPM-V系列模型，尝试了用llama-cpu推理gguf量化模型，使用llama-cpp-python库部署服务，支持openai chat api格式调用。
[OpenBMB/MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V/blob/main/README_zh.md#gradio-demo-)

+ [libcurl使用example](https://curl.se/libcurl/c/getinmemory.html)

+ swift大模型训推一体框架

......
