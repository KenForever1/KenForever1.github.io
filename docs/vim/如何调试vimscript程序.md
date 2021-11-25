---
title: 如何调试vimscript程序
date: 2021-08-09 10:56:44
tags:
---

当我们为vim或者nvim写插件时，需要用到vimscript语言，这里我介绍一下调试.vim文件的方法。

### 创建.vim 文件
首先，创建一个 my.vim 文件， 并且尝试写一些vimscript代码。
例如：

```
function Show(start, ...)
  echohl Title
  echo "start is " . a:start
  echohl None
  let index = 1
  while index <= a:0
    echo "  Arg " . index . " is " . a:{index}
    let index = index + 1
  endwhile
end
```

这里，我写了一个函数 Show，这个函数最少要传一个参数start，也可以动态传入多个参数(动态参数用‘...’表示)。a:0表示有几个动态参数，a:1表示第一个动态参数，a:2表示第二个动态参数，以次类推。

### 执行以及进入debug模式

vim或者neovim并不知道这个Show函数，所以我们在调试之前，应该先source这个文件。在vim下命令行模式（以":"进入），输入”source my.vim“。然后就可以执行Show函数了。


+ 正常执行

```
：call Show("hello", "aaa", "ccc")

```
+ debug执行

```
: debug call Show("hello", "aaa", "ccc")

```

使用debug模式的方法类似于gdb的使用，但是也有一些区别。

### debug模式的使用方法

cont : 一直执行到断点处，如果没有断点则执行完程序
quit : 停止当前执行，但仍会在下一个断点处停止
step : 执行当前命令并在完成后返回调试模式
next : 和step相似，但是它会跳过函数调用，不进入函数
interrupt : 类似于quit，但返回到下一个命令的调试模式
finish : 完成当前脚本或函数并返回到下一个命令的调试模式 

+ 添加断点 breakadd

```
# 为函数添加断点
breakadd func [lineNumber] functionName
# 为文件添加断点
breakadd file [lineNumber] fileName
# 为当前文件的当前行添加断点
breakadd here
```

+ 删除断点 breakdel

```
# 从断点列表中删除指定number的断点
breakdel number
# 删除所有的断点
breakdel *
# 删除函数中的断点
breakdel func [lineNumber] functionName
# 删除文件中的断点
breakdel file [lineNumber] fileName
# 删除当前文件的当前行的断点
breakdel here
```

以上就是debug的使用方法，详细参考：
[vim debug文档](http://vimdoc.sourceforge.net/htmldoc/repeat.html#:debug)
[vim learn](https://github.com/kktao/vim-learn)
