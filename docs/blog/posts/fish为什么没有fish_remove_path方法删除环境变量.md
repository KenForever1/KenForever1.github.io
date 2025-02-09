---
title: fish如何增加和删除环境变量PATH？为什么没有fish_remove_path方法删除环境变量
date: 2025-02-09
authors: [KenForever1]
categories: 
  - linux
labels: []
---

fish如何增加和删除环境变量PATH？fish为什么没有fish_remove_path方法删除环境变量。

<!-- more -->

## fish如何设置环境变量PATH

答案是：**fish_add_path命令**。

默认是universal variable。将变量增加到通用变量fish_user_paths中。除非指定-P或者--path采用修改$PATH环境变量。

```bash
$ fish_add_path
-g or --global
Use a global fish_user_paths.

-U or --universal
Use a universal fish_user_paths - this is the default if it doesn’t already exist.

-P or --path
Manipulate PATH directly.
```

```bash
ken@LAPTOP-44OC4FG2 ~ [1]> echo $fish_user_paths
/opt/nvim-linux64/bin /opt/helix-24.07-x86_64-linux

ken@LAPTOP-44OC4FG2 ~> fish_add_path /home/ken/tmp/

ken@LAPTOP-44OC4FG2 ~> echo $fish_user_paths
/home/ken/tmp /opt/nvim-linux64/bin /opt/helix-24.07-x86_64-linux
```

## fish变量的作用范围。

fish变量有四种，分别是通用变量(universal variable)、全局变量(global variable)、函数变量(function)和局部变量(local)。

+ 通用变量在用户在一台计算机上运行的**所有fish session之间共享**。**它们存储在磁盘上，即使在重新启动后也会保留**。可以通过“-U”或“--universal”指定。

+ 全局变量(global variable)特定于当前的fish session。可以通过执行“set -e”命令来清除它们，e就是erase的意思。使用“-g”或“--global”设置为全局变量。

+ 函数变量特定于当前正在执行的函数。当函数结束时，超出作用域将被清除。在函数外部，它们不会超出作用域。使用“-f”或“--function”设置为函数作用域变量。

+ 局部变量特定于当前的命令块，并且当特定的块超出作用域时会自动被清除。命令块是一系列以“for”、“while”、“if”、“function”、“begin”或“switch”中的一个命令开头，并以“end”命令结尾的命令。在块外部，这与函数作用域相同。使用“-l”或“--local”设置为当前块的局部变量。

可以有很多名称相同但作用域不同的变量。当你“使用变量”时，将使用具有该名称的作用域最小的变量。比如同时存在局部变量、全局变量、通用变量名称都叫"XX"，将会使用局部变量"XX"。

### universal(大家通用的变量)

什么是universal variables？
universal variables就是你作为用户，你的**所有的fish session会话之间共享的变量**。Fish 将其许多配置选项存储为通用变量。这意味着为了更改 fish 设置，你只需要更改一次变量值，它将自动为所有会话更新，并在**计算机重新启动以及登录/注销时保留**。

而我们这篇文章谈到的fish_user_paths就是universal variable。
```bash
fish_user_paths
a list of directories that are prepended to PATH. This can be a universal variable.
```

## fish如何删除path呢？

我们上面提到了，默认是加到fish_user_paths通用变量中了对吧？那么我们怎么删除呢？
我们可以通过查找fish_user_paths这个list中的下标位置，然后进行erase移除，就是下面的方法：
```bash
set -l index (contains -i -- /my/path/to/remove $PATH)
and set -e PATH[$index]
# or replace PATH with fish_user_paths
```

当然也可以自定义一个方法：
```bash
function remove_path
  if set -l index (contains -i "$argv" $fish_user_paths)
    set -e fish_user_paths[$index]
    echo "Removed $argv from the path"
  end
end
```

## 为啥fish不添加一个fish_remove_path方法呢？

那么为什么不能有fish_remove_path？因为对初学者，用一个命令行不是更加简单吗？为啥还要这么麻烦去写一个脚本，或者两句语句。

首先，$PATH 是从父进程继承的，没有办法永久地从其中删除某个东西，因为每次它都会被设置为一个新值。
作者认为添加一个fish_remove_path命令会很混乱。

原文是：
我们可以做一个 fish_remove_path，它可以从 $fish_user_paths 中删除你的路径，但如果路径不是在这里添加的，那就会很混乱，因为它看起来什么也没做。
或者我们可以添加一个从 $PATH 中删除它的，但如果稍后再次添加它，那就会很混乱。
唯一真正的解决方案是一开始就不要添加这个路径。
> We could do a fish_remove_path that would remove your path from $fish_user_paths, but that would be confusing if that's not where the path is added, because it would appear to do nothing.Or we could add one that removes it from $PATH, but that would be confusing if it was added again later.

参考:[fish_remove_path or equivalent, or documentation update](https://github.com/fish-shell/fish-shell/issues/8604).

有个commit提交给，fish_add_path增加了一个-r/--remove选项，指定删除path。

作者否定了:
> 这有一些我认为不可接受的权衡取舍。
> This has trade-offs that I find unacceptable.

参考：[Add --remove to fish_add_path](https://github.com/fish-shell/fish-shell/pull/9744)

通过，本文我们学会了fish如何设置环境变量$PATH, 以及变量的作用范围。为什么没有fish_remove_path命令，作者的考虑是什么。

