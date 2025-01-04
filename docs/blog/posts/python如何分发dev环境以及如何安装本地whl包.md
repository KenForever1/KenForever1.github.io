---
title: python如何分发dev环境以及如何安装本地whl包
date: 2025-01-05
authors: [KenForever1]
categories: 
  - Python
labels: []
---
用本地的pip配置文件，配置文件中指定了使用本地的pip源
<!-- more -->
```
export PIP_CONFIG_FILE=$PWD/pip.conf
```

pip.conf文件
```
[global]
no-index = true
find-links =
    /workspace/packages
```
/workspace/packages目录下就是你存放whl文件的地方。

通过下面的方式安装python whl：
```
python3 -m pip install xxx.whl --force-reinstall --no-cache-dir --no-index --no-deps
```
这个命令将强制重新安装当前目录下的所有 `.whl` 文件，不使用缓存，也不从 PyPI 下载任何包，同时忽略这些包的依赖项。

+ `--force-reinstall`：这个选项强制重新安装指定的包
+ `--no-cache-dir`：此选项告诉 `pip` 不要使用缓存目录，也就是在安装时不使用之前下载的包。
+ `--no-index`：这个选项禁止 `pip` 从 Python Package Index (PyPI) 下载包，仅使用本地提供的文件。
+ `--no-deps`：此选项告诉 `pip` 不安装包的依赖项。通常情况下，`pip` 会自动查找并安装所需的依赖项，但如果你明确指定了这个选项，它将只安装你提供的包，而不处理依赖关系。

简单的场景用上面分发你的python环境和依赖就可以了。还有个conda pack工具。

`conda-pack` 是一个工具，用于将 Conda 环境打包成一个独立的归档文件（如 `.tar.gz`），从而可以在不依赖 Conda 的情况下在其他计算机上使用该环境。这对于需要在不同机器之间移动环境或在没有 Conda 的系统上运行代码非常有用。

```
https://conda.github.io/conda-pack/
```


另外的方式，就是通过容器导出一个docker镜像分发环境了。