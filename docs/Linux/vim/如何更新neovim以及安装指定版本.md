---
title: 如何更新neovim以及安装指定版本.md
date: 2021-08-09 10:32:14
tags:
---

本文介绍如何更新neovim到最新版本（dev），以及安装指定版本的neovim。由于neovim的dev版本有很多api和特性与stable版本不一样，如neovim 0.5版本以后的undo file 和neovim 0.4版本（以及vim）不兼容，可能会报错（ E824: Incompatible undo file: *path-to-undo-file*.）[pull 13973](https://github.com/neovim/neovim/pull/13973)。通过本文可以方便我们体验dev版本，也可以重新安装到指定的stable版本。

### 1. 安装最新版本的neovim

+ 添加源

```
# stable version
sudo add-apt-repository ppa:neovim-ppa/stable

# dev version
sudo add-apt-repository ppa:neovim-ppa/unstable

```

+ 更新源

```
sudo apt update

# 直接安装不会成功，会提示
# "The following packages have been kept back: ... "
sudo apt upgrade
```

+ 安装

```
sudo apt-get install neovim
```

[solve for kept back](https://askubuntu.com/questions/601/the-following-packages-have-been-kept-back-why-and-how-do-i-solve-it)

### 2. 安装指定版本的neovim

我之前的neovim版本是0.4.4, 我自己的配置使用了[vim-startify](https://github.com/mhinz/vim-startify)插件，当我更新到dev版本0.6以后，出现了undo file的报错。为了稳定性，我重新安装了0.4.4版本。

+ 查看有哪些neovim版本

```
sudo apt install apt-show-versions 

apt-show-versions -a neovim
```

+ 安装查询的版本

```
sudo apt-get install neovim=0.4.4-1~ubuntu20.04.1~ppa1
```

以上就是如何安装指定的neovim版本的方法，感谢您的阅读。
