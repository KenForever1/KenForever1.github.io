---
title: 优雅开发篇：git worktree并行分支开发，及一个测量二进程膨胀的rust工具应用
date: 2025-07-01
authors: [KenForever1]
categories: 
  - linux
labels: []
comments: true
---

## git worktree 是什么？

Git Worktree 是 Git 提供的一个强大功能，允许你在同一个仓库中**创建多个独立的工作目录**，每个目录可以关联不同的分支（或者 commit），从而实现并行开发而**无需频繁切换分支或依赖 git stash 暂存代码 ‌**。

<!-- more -->

![](https://raw.githubusercontent.com/KenForever1/CDN/main/git-worktree.png)

### 可以解决什么问题？

没有使用 worktree 的同学，假设在 dev 分支上开发，开发了一半，需要切换到 release 分支修复 bug，那么需要：

1. 保存当前分支的状态，比如 git commit 或者 git stash。
2. 切换到 release 分支，切一个 hotfix 分支，修复 bug，提交。
3. 切换回来继续开发。

或者：

1. git clone repo 到另一个目录，修复 bug。
2. 切换回原目录，继续开发。

缺点就是：

第一，这样很麻烦，切来切去。

第二，占地方，多个 repo 拷贝占磁盘空间。

第三，如果 2 个版本分支的工程依赖环境差异较大，导致每次切换分支后，工程都都需要重新安装依赖以及做全量编译——这无疑增加了编译时间，导致开发效率下降。

### 有什么优势？

针对这个痛点， Git Worktree 出现了, 它具有下面的优势：

1. 不需要提交或者暂存代码，就可以切换分支。面对多个分支的代码不需要频繁切换。
2. 每个工作区共享同一个版本仓库信息，更节省硬盘空间。
3. 各个工作区之间的更新同步更快，git clone 方式下，A 工作区和 B 工作区需要 A commit-> A push -> B pull。git worktree 方式下，A 工作区只要本地提交更新后，其他工作区就能立即收到（因为它们共享同一个版本仓库）。

## 通过一个例子学习使用 git worktree

### git worktree如何使用

假设我们有个 helloworld 的仓库：https://github.com/KenForever1/helloworld。

当前工作在 dev 分支上，想要在 release 分支修复 bug，那么就可以使用 git worktree，而无需切换分支。

为release分支增加一个worktree：

```bash
cd /workspace/helloworld
git worktree add ../release-branch release
```

进入release分支的worktree目录，修复bug:
```bash
cd ../release-branch
// fix your bug here
```

查看当前 worktree 列表：

```bash
git worktree list
/workspace/helloworld [main]
/workspace/release-branch [release-branch]
```

切换回 dev 分支，删除 release 分支：

```bash
cd ../dev-branch
git worktree remove release-branch
```

### 其他命令

1. 删除除了 remove 命令，还可以手动删除目录，然后 git worktree 会自动清理

```bash
rm -rf ../release-branch
git worktree prune
```

2. 锁定与解锁 ‌

```bash
git worktree lock releas-branch --reason "维护中"
# 解锁‌
git worktree unlock releas-branch
```

锁定了以后，git worktree remove 会报错，需要先解锁。

## 一个测量二进制膨胀的 rust 小工具

### 设计原理

下面介绍的一个 rust 工具测量二进制膨胀（Measures bloat）的小工具：
[facet-rs/limpid](https://github.com/facet-rs/limpid/blob/HEAD/limpid/src/main.rs "facet-rs/limpid")

一个是当前 HEAD（目前开发的新特性），一个是 main 分支（已经发布的版本）。目的是想对比分析两个目录下的编译构建耗时，二进制大小差异等。
为了实现这个功能，针对 HEAD 开发目录，创建了一个 worktree 目录，命名为 main。对比分析完成后，删除 worktree 目录。

因为对比的仓库[facet-rs/facet](https://github.com/facet-rs/facet "facet-rs/facet")，是一个 rust 序列化库 crate，可以理解和 serde 差不多。
在 limpid 工具中，包括了创建 worktree 和编译统计信息的代码，另一个就是使用 facet 库的例子，采用相对目录引用 facet。所以只要保持层级一致，就可以使用不同分支的 facet 库。

工作目录的结构如下：

```bash
facet/ # facet库HEAD分支
  facet/
    Cargo.toml
  facet-core/
    Cargo.toml
  facet-reflect/
    Cargo.toml
limpid/ # limpid工具
  limpid/ # limpid工具
    Cargo.toml # this tool
  kitchensink/ # 使用facet库的例子，可以编译成二进制程序
    ks-facet/
      Cargo.toml # 相对目录方式，引用../facet库
    ks-serde/
      Cargo.toml
    ks-facet-json-read/
    # etc.
```

对比的 main worktree 目录。

```bash
/tmp/
  limpid-main-workspace/
    facet/       # worktree from facet repo at main branch
      facet/
      facet-core/
      facet-reflect/
    limpid/      # worktree from limpid repo at current HEAD
      limpid/
      kitchensink/
        ks-facet/
        ks-serde/
        ks-facet-json-read/
        # etc.
```

通过，这个工具我们可以学习 worktree 的用法，也可以学习到一些工具的设计理念。与语言无关，你也可以用 python 或 c++以及其他语言实现。

### 杂谈

这个例子如果你想学习 rust 也可以参考，里面有很多常用的不错的 crate 使用，比如：

```bash
anyhow = "1.0" # 用于错误处理
substance = "0.7.1" # https://github.com/fasterthanlime/substance.git，检查二进制文件的符号、分析二进制文件的大小构成
owo-colors = "4.0" # 用于命令行输出彩色字
indicatif = "0.17" # 进度条
camino = "1.1.10" # Utf8路径支持，Path和PathBuf更好的包装
pico-args = "0.5.0" # 命令行参数解析，比clap更轻量
```

顺便提一句，针对刚刚提到的 substance rust 工具，google 也有个 c++工具[Bloaty](https://github.com/google/bloaty "Bloaty")。可以解决是什么让你的二进制文件变得这么大？**Bloaty 会为你展示二进制文件的大小分布**，这样你就能了解其中是什么占用了空间。
