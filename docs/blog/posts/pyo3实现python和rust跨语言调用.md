---
title: pyo3实现python和rust跨语言调用
date: 2025-09-20
authors: [KenForever1]
categories: 
  - rust
  - python
labels: []
comments: true
---

<!-- more -->

## pyo3是什么？

pyo3是一个Rust库，可以实现python和rust之间的相互调用。可以使用Rust为Python编写扩展。
教程可以参考[pyo3 getting-started](https://pyo3.rs/main/getting-started.html)。

就行，pybind11一样，pybind11是一个c++库，提供了c++和python之间的相互调用。

## FFI需要考虑什么问题？

python和rust之间的相互调用，需要考虑的问题有：
线程安全，线程间竞争。pyo3针对不同的python版本实现，实现了GIL和no GIL。
数据结构的生命周期，pyo3封装了智能指针，通过过程宏生成函数

## python的no GIL

GIL就是Python的全局解释器锁，因此python的多线程实际是伪多线程，不能像c++等其它语言的多线程一样真正的实现并发。

python的no GIL是CPython 3.14提出的不再依赖全局解释器锁的多线程实现，提升并发。

> CPython 3.14 declared support for the "free-threaded" build of CPython that does not rely on the global interpreter lock (often referred to as the GIL) for thread safety. 


## 如何编译一个多个版本都可以使用的wheel包？与什么特性有关？

我们平时使用python依赖安装包时可能会注意到，有些wheel文件命名上是指明了python版本要求的，比如python3.8的wheel包和python3.10等是不同的文件。

标准Python ABI通常要求模块针对特定Python版本编译，而ABI3通过冻结关键二进制接口实现跨版本兼容，降低了维护成本。

默认情况下，Python扩展模块只能与编译它们时所针对的Python版本一起使用。例如，为Python 3.5构建的扩展模块无法在Python 3.8中导入。[PEP 384](https://www.python.org/dev/peps/pep-0384/)引入了有限Python API的概念，该API将具有稳定的ABI，使使用它构建的扩展模块能够在多个Python版本中使用。这也被称为abi3。

其缺点是，PyO3无法使用那些依赖于针对已知确切Python版本进行编译的优化。pyo3仅调用属于稳定API的Python C-API函数。

由于单个abi3轮包可用于多个不同的Python版本，PyO3提供了abi3-py37、abi3-py38、abi3-py39等功能标志，用于为您的abi3轮包设置最低所需的Python版本。例如，如果您设置了abi3-py37功能，您的扩展轮包可用于从Python 3.7及更高版本的所有Python 3版本。maturin和setuptools-rust会给轮包命名为类似my-extension-1.0-cp37-abi3-manylinux2020_x86_64.whl的名称。


## 一个python包，如何测试多个版本？

[nox](https://nox.thea.codes/en/stable/index.html)，可以自动化的在多个python版本下运行测试。通过py脚本配置。

