---
title: git_sparse_checkout检出特定文件或目录
date: 2025-02-09
authors: [KenForever1]
categories: 
  - linux
labels: []
comments: true
---

Git的sparse checkout功能允许用户仅检出仓库中的特定目录或文件，而非整个仓库内容，这对于处理大型仓库或只需关注部分内容的场景非常有用。

<!-- more -->

## Download the add_sub model
1. Create a new directory and enter it
```bash
mkdir <new_dir> && cd <new_dir>
```
2. Start a git repository

```bash
git init && git remote add -f origin https://github.com/triton-inference-server/model_analyzer.git
```
3. Enable sparse checkout, and download the examples directory, which contains the add_sub model

```bash
git config core.sparseCheckout true && \
echo 'examples' >> .git/info/sparse-checkout && \
git pull origin main
```
[参考使用例子](https://github.com/triton-inference-server/model_analyzer/blob/main/docs/quick_start.md)。

## 命令方式
现代Git版本提供了更简洁的命令：

```bash
git sparse-checkout init --cone
git sparse-checkout set dir1 dir2
```