---
title: C++常见错误：窥探std::Sort的cmp函数用错，导致的程序abort分析
date: 2024-03-31
authors: [KenForever1]
categories: 
  - cpp
labels: []
pin: true
comments: true
---

## std::sort 函数的Compare函数
c++ std::sort函数是经常被使用到的，但是不知道大家注意没有，定义的Compare函数是需要满足一定条件的。这个条件就是：strict weak ordering。

<!-- more -->
[cppreference](https://en.cppreference.com/w/cpp/algorithm/sort) 的英文原文：

> comparison function object (i.e. an object that satisfies the requirements of Compare) which returns ​true if the first argument is less than (i.e. is ordered before) the second.

strict weak ordering描述了以下三条：第一条描述的是，如果a，b作为cmp的输入，当a和b相等时，return的值必须是false。

![](https://picx.zhimg.com/v2-ade2b319b508ad3f4772788aa5157cc5_1440w.jpg)


## 一个容易发生且错误使用std::sort的例子

在一段业务代码中，发现了一个不满足cmp函数**strict weak ordering**的例子，引起了内存相关的问题。以下面的代码为例：

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

void func()
{
    std::vector<int> v1;

    for (auto i = 0; i < 18; i++)
    {
        v1.push_back(1);
    }
    std::cout << "v1 addr : " << &v1[0] << std::endl;
    std::cout << "befor sort" << std::endl;
    std::sort(v1.begin(), v1.end(), [](int a, int b) { return a >= b; });
    std::cout << "after sort" << std::endl;
}

int main(int, char **)
{
    func();
}
```

```bash
v1 addr : 0x55e0ab6d1f30
befor sort
after sort
double free or corruption (out)
fish: Job 1, './hello/…' terminated by signal SIGABRT (Abort)
```
## 崩溃原因分析

执行上面的程序Abort了，原因就是cmp函数中的=等号。为什么会引起这个问题呢，通过gdb查看vector变量附近的内存。

```bash
(gdb) b 14
(gdb) b 16
(gdb) r
(gdb) x/26wd 0x55555576df20
0x55555576df20: 0       0       145     0
0x55555576df30: 1       1       1       1
0x55555576df40: 1       1       1       1
0x55555576df50: 1       1       1       1
0x55555576df60: 1       1       1       1
0x55555576df70: 1       1       0       0
0x55555576df80: 0       0
(gdb) until 2
(gdb) x/26wd 0x55555576df20
0x55555576df20: 0       0       145     1
0x55555576df30: 1       0       1       1
0x55555576df40: 1       1       1       1
0x55555576df50: 1       1       1       1
0x55555576df60: 1       1       1       1
0x55555576df70: 1       1       0       0
0x55555576df80: 0       0
```

可以看到内存数据被改了， 比如0x55555576df20一行地址的145，0被改成了145，1。比如0x55555576df30一行地址的1，1，1，1被改成了1，0，1，1。

当把set *(0x55555576df2c)=0，就不会报错了。这里与vector的析构函数调用有关。

在项目代码中，如果vector中是一个class或者std::string，那么报错现象可能会是std::bad_alloc，析构std::string出错，或者析构class出错，导致问题不易察觉。但本质是因为std::sort的cmp函数定义不正确，导致内存数据被更改，所以代码执行出错了。

这里找到了一篇古老的文章，从源码上进行了分析：[一次stl sort调用导致的进程崩溃](https://blog.sina.com.cn/s/blog_532f6e8f01014c7y.html)，讲解得很细致清楚。