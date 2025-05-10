---
title: C++常见陷阱：一个例子了解const使用
date: 2025-02-03
authors: [KenForever1]
categories: 
  - cpp
labels: []
pin: true
comments: true
---

## 一个例子了解：cpp 常量常见陷阱

下面通过这个例子，一起了解一下 cpp 常量使用的常见陷阱。

<!-- more -->

```c++
// do_s.h
#pragma once
constexpr char kSpecial[] = "special";

// 比较字符串函数
void DoSomething(const char* value);
```

```c++
// do_s.cpp
#include "do_s.h"
#include <iostream>

void DoSomething(const char* value) {
    std::cout << "addr in func: " << &kSpecial << std::endl;

  // 比较字符串，👇👇👇👇注意下面这行代码👇👇👇👇
  if (value == kSpecial) {
    // do something special
    std::cout << "it's special!" << std::endl;
  } else {
    // do something boring
    std::cout << "it's boring!" << std::endl;
  }
}
```

```c++
// main.cpp
#include "do_s.h"
#include <iostream>

int main(){
    std::cout << "addr in main: " << &kSpecial << std::endl;
    DoSomething(kSpecial);
    return 0;
}

```

猜一下运行结果会是什么呢？会打印"it's special!"吗？

```bash
$ g++ do_s.cpp main.cpp
$ ./a.out
addr in main: 0x562e0650d040
addr in func: 0x562e0650d008
it's boring!
```

## 原因分析

和预期的执行结果一致怎么不一致呢？当我们调用 DoSomething(kSpecial)的时候，执行 do_s.cpp 中"if (value == kSpecial)"代码哪个分支的行为是不确定的，也就是结果是未定义的（undefined behavior）!

kSpecial 对象在编译过程中会产生一组对象，这是 C++标准定义的：编译时每个引用了 do_s.h 文件的源代码文件会有一个独立的编译单元，每个编译单元有一个独立的 kSpecial 对象副本，每个对象的地址都不同。

所以在 do_s.cpp 代码，会出现未定义现象，即在不同的调用位置，kSpecial 的地址不同，导致同样的 DoSomething(kSpecial)代码可能结果不同。

main.cpp 和 do_s.cpp 是两个独立的编译模块，导致了 kSpecial 对象在编译阶段产生了两个独立的实例，每个实例的地址都不同，所以当调用 DoSomething(kSpecial)时，就会导致未定义行为。

如果将 do_s.cpp 中的内容移动到 main.cpp 中，删除 do_s.cpp，编译和运行结果就是预期的结果"it's special!"了。因为只有一个编译单元了。

但是，这不能解决根本问题呀！！！

## 小插曲：const 介绍

### const 代表了只读，不代表不可以修改

在 c++中提供了 mutable 和 const_cast 等手段修改。看一个例子：

```c++
void f(const std::string& s) {
  const int size = s.size();
  std::cout << size << '\n';
}

f("");  // Prints 0
f("foo");  // Prints 3
```

在上述代码中，size 是一个 const 变量，但在程序运行时它持有多个值。它并非常量。

### 非恒定的常量

const 经常和指针一起使用：

```c++
const char* kStr = "foo";
const Thing* kFoo = ...;
```

上述 kFoo 是一个指向常量的指针，但指针本身不是常量。你可以对其赋值、设为 null 等。

```c++
kStr = "bar";   // kStr其实是可以修改的
kFoo = nullptr;     // kFoo同样也是可以修改
```

如果我们想实现一个“不能修改”的常量，应该如下实现

```c++
const char* const kStr = ...;
const Thing* const kFoo = ...;
// C++17之后，可以这样
constexpr const Thing* kFoo = ...;
```

## 如何解决上面的问题呢

### 头文件和源文件分离定义常量

了解链接，链接与程序中一个命名对象有多少实例（或 “副本”）有关。通常，在程序中，具有一个名称的常量最好引用单个对象。对于全局或命名空间作用域的变量，这需要一种称为外部链接的东西。

```c++
// do_s.h
extern const int kMyNumber;
extern const char kSpecial[];
extern const std::string_view kMyStringView;

// 即上面的代码修改为
// extern const char kSpecial[];
```

```c++
// do_s.cpp
constexpr int kMyNumber = 42;
constexpr char kSpecial[] = "special";
constexpr std::string_view kMyStringView = "Hello";

// 常量定义
// constexpr char kSpecial[] = "special";
```

### 仅在头文件或者源文件定义常量

在头文件中通过函数返回常量：

```c++
// constexpr函数，可以调用MyNumber函数来获取常量
constexpr int MyNumber() { return 42; }

// 一个普通函数定义，注意这里的kHello是一个真正的常量，地址不变
// 可以调用MyString()来获取常量对象
inline std::string_view MyString() {
  // 注意一定要static constexpr修饰，否则会有未定义行为发生
  static constexpr char kHello[] = "Hello";
  return kHello;
}
```

或者，如果只需要在 cpp 文件中使用的话，可以定义在源文件中，不要放在头文件中。

```c++
// 只在cpp文件中使用的话，可以如下定义常量
constexpr int kBufferSize = 42;
constexpr char kBufferName[] = "example";
constexpr std::string_view kOtherBufferName = "example";
```

详细内容可以阅读[abseil tips 140](https://abseil.io/tips/140 "abseil tips 140")。
