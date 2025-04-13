---
title: C++开发技法：用PIMPL模式彻底隐藏类实现，编译速度飞升
date: 2025-01-01
authors: [KenForever1]
categories: 
  - cpp
labels: []
pin: true
comments: true
---

# C++开发技法：用PIMPL模式彻底隐藏类实现，编译速度飞升

## 什么是PIMPL技法？
<!-- more -->
[PIMPL‌](https://en.wikibooks.org/wiki/C%2B%2B_Programming/Idioms)（Pointer to IMPLementation，指向实现的指针）（也叫"opaque pointer"）是一种 C++ 编程惯用法，用于‌隐藏类的实现细节‌，减少编译依赖，提升代码封装性和二进制兼容性。

## 哪PIMPL有什么用途呢？

学到一个技法，总要应用到开发实践中，那你肯定会问，我学会了这个有啥用呢？

+ 库开发‌：保持 ABI 兼容性，隐藏实现细节。
比如你的工作是提供sdk给你的用户，那么你肯定不希望接口类增加一个新的成员变量，就要重新导出头文件（并更新一个新的动态链接库so），你的用户可执行程序也要重新编译一次，因为保证二进制兼容性。
而采用PIMPL方式，你不用修改接口，只需要修改接口指向的实现，你就只需要提供一个动态链接库给用户，用户不需要替换头文件，重新编译可执行文件了。

+ ‌大型项目‌：减少编译时间，加速增量构建。
正如前面库开发的例子谈到的，我们是不是只用编译部分so，相比于全部编译，在大型项目中就减少了编译时间。因此PIMPL还有个说法，叫"Compilation Firewall"编译防火墙。

‌+ 接口稳定性要求高‌：公有头文件需要长期保持稳定。

我们总结一下PIMPL的优点：‌减少编译依赖‌，修改 Impl 不会触发依赖主类头文件的代码重新编译。‌接口稳定，公有接口保持不变时，实现可自由修改。‌二进制兼容性‌，保持类布局稳定，避免 ABI 破坏。隐藏第三方依赖‌，避免在头文件中暴露外部库头文件。

当然也有缺点了‌，比如它增加了间接访问开销‌，通过指针访问成员，可能有轻微性能损失。‌相比你直接修改主类，提升了代码复杂度‌，还需要维护一个实现类。由于是动态内存分配‌，Impl 通常存储在堆上，可能影响内存局部性。

## 一起来看看怎么实现

前面讲的都偏概念了，接下来看一看怎么实现的。通过一个例子出发：

```c++
// public.h
class Book
{
public:
  void print();
private:
  std::string  m_Contents;
}
```
我们有个Book的类，提供了一个print函数给用户。试想一下，我们要给Book类增加信息，假如想增加一个书名成员变量，用户并不care。
```c++
// public.h
class Book
{
public:
  void print();
private:
  std::string  m_Contents;
  std::string  m_Title;
}
```
但是我们改动了这个接口，导致可执行文件需要重新编译了。来看看PIMPL技法是如何解决这个问题的？

```c++
/* public.h */
class Book
{
public:
  Book();
  ~Book();
  void print();
private:
  class BookImpl;
  BookImpl* const m_p;
}
```

```c++
/* private.h */
#include "public.h"
#include <iostream>
class Book::BookImpl
{
public:
  void print();
private:
  std::string  m_Contents;
  std::string  m_Title;
}
```

改动了Book类, 通过BookImpl*的方式，拆分了一个内部类。将Book的具体实现细节移动到了BookImpl中，在Book类中只保留了print()函数接口。

```c++
#include "private.h"
// book.cpp
Book::Book(): m_p(new BookImpl())
{
}

Book::~Book()
{
  delete m_p;
}

void Book::print()
{
  m_p->print();
}

/* then BookImpl functions */

void Book::BookImpl::print()
{
  std::cout << "print from BookImpl" << std::endl;
}
```
main函数模拟可执行程序执行：
```c++
#include "public.h"

int main()
{
  Book b;
  b.print();
}
```

具体的代码实现[KenForever1/cpp_idioms/pimpl](https://github.com/KenForever1/cpp_idioms)。

你也可以使用 std::unique_ptr<BookImpl> 或类似的方法来管理内部指针。

## 项目中的应用

+ [NVIDIA/TensorRT-LLM 中executor](https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/executor/executor.cpp)