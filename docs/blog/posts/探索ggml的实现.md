---
title: 探索ggml的实现
date: 2025-09-01
authors: [KenForever1]
categories: 
  - cpp
labels: [cpp]
pin: true
comments: true
---

<!-- more -->

本文主要介绍ggml的调试入门环境搭建、ggml的核心数据结构、内存管理与布局、计算图构建与执行、以及gguf文件格式构成，会分成多个小节介绍。

本文起源于作者ggml学习过程中了解的资料，包括[xsxszab的ggml-deep-dive系列文章](https://xsxszab.github.io/posts/ggml-deep-dive-i/)、以及阅读源码，记录和分享自己的理解过程。

ggml是采用c/c++编写高度优化的tensor张量计算库，没有外部依赖。存在不同的backend实现，从mac的metal、x86的avx实现以及arm的neon指令实现、gpu的cuda、hip、opencl等实现。llama.cpp项目就是使用ggml进行模型加载、推理的。

## 探索ggml的实现--vscode调试环境的搭建

在阅读源码的时候，找一个最简单的example例子跑通，然后跟着源码进行调试，是很快理解原理的一种方法。 

在linux、或者mac下都方便编译。

编译依赖：

+ gcc/g++或者 clang编译
+ cmake： 用于管理构建项目
+ ccache：可选项，加快编译速度用

```bash
git clone https://github.com/ggml-org/ggml.git
```

上面的依赖安装很简单，以ubuntu为例：
```bash
apt update
apt install -y gcc g++ cmake ccache
```

### 配置调试目标

ggml的example中包括了很多例子，先从简单的例子看起。

```bash
examples/simple/simple-ctx.cpp
```

修改项目的cmake配置，即添加-g方便debug调试。找到examples/simple/CMakeLists.txt配置文件，为simple-ctx这个可执行文件添加-g参数。
```bash
set(TEST_TARGET simple-ctx)
add_executable(${TEST_TARGET} simple-ctx.cpp)
target_compile_options(${TEST_TARGET} PRIVATE -g)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)
```
上面只是为可执行文件添加了-g调试信息，如果要调试ggml.so这个库，则需要编译ggml.so库时指定Debug类型。

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=DEBUG
cmake --build build
```
查看编译输出，可以看到ggml.so库被编译成with debug_info。
```bash
file build/src/libggml.so 
build/src/libggml.so: ELF 64-bit LSB shared object, x86-64, version 1 (SYSV), dynamically linked, BuildID[sha1]=db376e303daeef672c8002ed6cebed1da303a706, with debug_info, not stripped
```

### vscode配置debug

点击右边的Run and Debug按钮，create a launch.json file，创建一个配置文件，把下面的内容粘贴进去。

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "debug simple-ctx",
            "type": "lldb",
            "request": "launch",
            "program": "${workspaceFolder}/build/bin/simple-ctx",
            "cwd": "${workspaceFolder}"
        }
    ]
}
```
接下来就可以按照vscode的界面button调试了，如果要调试其它可执行程序，修改launch.json的program字段即可。

可以看到，通过上面的编译debug模式和添加-g参数，可以进入可执行程序和ggml.so库的源码进行调试了了。

![](https://raw.githubusercontent.com/KenForever1/CDN/main/ggml-debug.png)

![](https://raw.githubusercontent.com/KenForever1/CDN/main/ggml-debug1.png)