---
title: c++从一次静态库链接引发的double free聊起
date: 2025-03-22
authors: [KenForever1]
categories: 
  - cpp
labels: [quickcheck]
pin: true
comments: true
---

## 从一个例子说起
在这个例子中，我们将创建一个可执行程序 `A`，它使用动态链接库 `C`，而 `C` 本身又依赖于另一个动态链接库 `D`。

库 `C` 和 `D` 都链接了库 `E`，其中库 `E` 包含一个使用 `extern` 声明和实现的 `const std::string` 全局变量。

<!-- more -->

### 文件内容

文件的目录结构如下

```bash
/project
    /src
        main.cpp             // 可执行程序 A 的源文件
        libC.cpp             // 动态库 C 的源文件
        libD.cpp             // 动态库 D 的源文件
        libE.cpp             // 动态库 E 的源文件
        libE.h               // 动态库 E 的头文件
    /build
```

#### libE.h

```cpp
#ifndef LIBE_H
#define LIBE_H

#include <string>

extern const std::string global_message;

void printMessage();

#endif // LIBE_H
```

#### libE.cpp

```cpp
#include "libE.h"
#include <iostream>

const std::string global_message = "Hello from library E!";

void printMessage() {
    std::cout << global_message << " (Address: " << &global_message << ")" << std::endl;
}
```

#### libD.cpp

```cpp
#include "libE.h"

void callPrintMessageFromD() {
    printMessage();
}
```

#### libC.cpp

```cpp
#include "libE.h"

extern void callPrintMessageFromD();

void callPrintMessageFromC() {
    printMessage();
    callPrintMessageFromD();
}
```

#### main.cpp

```cpp
extern void callPrintMessageFromC();

int main() {
    callPrintMessageFromC();
    return 0;
}
```

### 采用cmake构建

CMakeLists.txt文件如下：

```c++
cmake_minimum_required(VERSION 3.10)
project(a_demo)

add_library(e STATIC libe.cpp)
target_compile_options(e PUBLIC -fPIC)
add_library(c SHARED libc.cpp)

target_link_libraries(c 
e
)

add_library(d SHARED libd.cpp)
target_link_libraries(d
e
)

add_executable(main 
    main.cpp
)
target_link_libraries(main
    c
    d
)
```


```bash
$ nm -CD libc.so | grep global_message
00000000000040a0 B global_message[abi:cxx11]
$ ~/a_demo/build# nm -CD libd.so | grep global_message
00000000000040a0 B global_message[abi:cxx11]
```

### 运行

按照正常的逻辑，程序运行时，输出应该是下面这样的：

```bash
Hello from library E!
Hello from library E!
```

然而，当你运行程序时，输出却是。报错了！！！你知道为什么会这样吗？你有想到这个结局吗！

```bash
Hello from library E! (Address: 0x7fc90c3740a0)
Hello from library E! (Address: 0x7fc90c3740a0)
free(): double free detected in tcache 2
Aborted (core dumped)
```

### 一起看一看
`global_message` 是一个在库 `E` 中定义的 `const std::string`，并在库 `C` 和 `D` 中使用。

如果将 libE 编译为静态库而不是动态库，并且 libC 和 libD 都静态链接 libE，都包含一份global_message 实例，但是观察发现这两个实例的地址是一样的，析构两次，导致 global_message 的 double_free 问题。

检查符号表

```bash
# 查看动态库中的 global_message 符号
nm -CD libc.so | grep global_message
nm -CD libd.so | grep global_message
```
显示两个动态库均包含 global_message
打印地址，在 libe.cpp 中打印的global_message地址是相同的：
```c++
void printMessage() {
    std::cout << global_message << " (Address: " << &global_message << ")" << std::endl;
}
```

为什么静态库会导致重复定义？

静态库的本质‌：一组 .o 文件的集合。链接时，链接器仅提取被引用的目标文件。

动态库链接静态库‌：每个动态库独立链接静态库时，会将所需的 .o 文件复制到自身，导致多份数据副本。

## 如何解决呢？

有三种方法可以解决这个问题，下面一起来看一下：

### ‌将 libe 改为动态库‌

编译 libe 为动态库：
```bash
g++ -shared -fPIC libe.cpp -o libe.so
```

或者将上述CMakeLists.txt文件中的STATIC改为SHARED。让 libc.so 和 libd.so 链接动态库 libe.so，而非静态库 libe.a。效果‌：全局变量仅有一份实例，避免了重复析构。

### 使用单例模式重构全局变量‌

修改 libe.h 和 libe.cpp：

```c++
// libe.h
#include <string>
const std::string& get_global_message();  // 返回引用而非 extern 变量
void printMessage();

// libe.cpp
#include "libe.h"
#include <iostream>

const std::string& get_global_message() {
    static const std::string instance = "Hello from library E!";  // 局部静态变量
    return instance;
}

void printMessage() {
    std::cout << get_global_message() << std::endl;
}
```

原理‌：利用局部静态变量的线程安全初始化（C++11 起），确保全局唯一实例。

### 采用inline

```c++
inline const std::string global_message = "Hello from library E!";
```

添加inline也可以解决。添加inline后重新编译，查看符号表：

```bash
$ nm -CD libd.so | grep global_message
00000000000040a0 u global_message[abi:cxx11]
0000000000004088 u guard variable for global_message[abi:cxx11]
```

guard variable在多编译单元场景下仅被初始化一次，避免重复构造或竞争条件

> Inline const variables at namespace scope have external linkage by default (unlike the non-inline non-volatile const-qualified variables). (since C++17) 
> 
> https://en.cppreference.com/w/cpp/language/inline
