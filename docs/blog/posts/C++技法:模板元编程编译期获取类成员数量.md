---
title: C++技法:模板元编程编译期获取类成员数量
date: 2025-06-29
authors: [KenForever1]
categories: 
  - C++
labels: []
comments: true
---

C++反射中，有个必要的就是需要获取一个类的成员个数，然后就可以根据个数，将类的成员通过std::tie转换成tuple。继而可以实现equal、hash、serialize等功能。
<!-- more -->

### 基本原理

本文介绍分析如何获取一个类的成员个数的方法，通过模板元编程实现，基本原理：
+ **编译时计算**结构体成员数量

+ 使用**SFINAE**技术，通过**递归实例化arity_impl模板来探测**结构体能接受的最大参数数量

+ 通过一个instance类提供**到任意类型的隐式转换**

### 核心代码实现

核心代码如下：

```c++
#pragma once
#include <type_traits>
namespace detail {

// instance类的作用：定义了operator函数，提供到任意类型的隐式转换操作符，用于模拟构造Aggregate类型时所需的任意类型参数
struct instance {
  template <typename Type>
  operator Type() const;
};

template <typename Aggregate, typename IndexSequence = std::index_sequence<>,
          typename = void>
struct arity_impl : IndexSequence {};

// 特化版本
template <typename Aggregate, std::size_t... Indices>
struct arity_impl<Aggregate, std::index_sequence<Indices...>,
                  std::void_t<decltype(Aggregate{
                      (static_cast<void>(Indices), std::declval<instance>())...,
                      std::declval<instance>()})>>
    : arity_impl<Aggregate,
                 std::index_sequence<Indices..., sizeof...(Indices)>> {};

}  // namespace detail

template <typename T>
constexpr std::size_t arity() noexcept {
  // 使用decay_t去除类型修饰（如const/volatile/引用）
  return detail::arity_impl<decay_t<T>>().size();
}
```

首先，对上面的代码进行拆解，分细节进行讨论：

#### 细节1: 构造Aggregate对象+逗号表达式
```c++
Aggregate{
    (static_cast<void>(Indices), std::declval<instance>())...,
    std::declval<instance>()}
```
这段代码尝试构造Aggregate对象，通过不断增加参数数量直到编译失败，从而确定最大有效参数数量。

```c++
(static_cast<void>(Indices), std::declval<instance>())
```
这段代码通过std::declval\<instance\>()生成一个instance对象，并通过static_cast\<void\>(Indices)来避免编译器警告。这里还用到了一个技术，就是c++中的逗号表达式。

C++中的逗号表达式是一种特殊的运算符，它可以将多个表达式连接起来并按顺序求值。逗号表达式的一般形式为：表达式1, 表达式2, ..., 表达式n。其求值过程是从左到右依次计算每个子表达式，最终整个表达式的值为最后一个表达式（表达式n）的值。例如：
```c++
int a = (1, 2, 3); // a的值为3
```
结合上面的代码，就只会输入std::declval\<instance\>()作为参数构造Aggregate对象。

参数包(Indices, instance)...生成N个instance，额外添加的instance用于探测边界条件。也就是indices为(0, 1)时，传递3个参数，当indices为(0, 1, 2)时，传递4个参数。

#### 细节2：SFINAE技术+void_t表达式合法性检查

```c++
std::void_t<decltype(Aggregate{
                      (static_cast<void>(Indices), std::declval<instance>())...,
                      std::declval<instance>()})>>
```

说明一下这里std::void_t，void_t通常结合SFINAE技术进行元编程的类型诊断与表达式的合法性检查，void_t本身定义非常简单，对于任意类型，乃至可变的参数类型，都重定义为void。

看个例子你就明白了：

+ 检测类型成员是否存在
```c++
// 泛化版本（默认返回 false）
template<typename T, typename = void>
struct has_type_member : std::false_type {};

// 特化版本（当 T::type 存在时匹配）
template<typename T>
struct has_type_member<T, std::void_t<typename T::type>> : std::true_type {};
```

若 T 包含 type 成员类型，则特化版本生效，返回 true。
+ 检测函数是否存在

```c++
template<typename T, typename = void>
struct has_hello_func : std::false_type {};

template<typename T>
struct has_hello_func<T, std::void_t<decltype(std::declval<T>().hello())>> 
    : std::true_type {};
```

#### 细节3：index_sequence的使用

std::index_sequence是C++14引入的编译期整数序列工具，主要用于模板元编程中处理参数包和索引操作。

例如：
```c++
template<size_t... I>
constexpr auto make_squares(std::index_sequence<I...>) {
    return std::array{I*I...};
}
auto arr = make_squares(std::make_index_sequence<5>{}); // {0,1,4,9,16}

```

#### 细节4: 递归探测的机制

```c++
template <typename Aggregate, std::size_t... Indices>
struct arity_impl<Aggregate, std::index_sequence<Indices...>,
                  // 本次探测的参数列表，用于探测Aggregate能接受的参数数量，参数数量为sizeof...(Indices)+ 1
                  std::void_t<decltype(Aggregate{
                      (static_cast<void>(Indices), std::declval<instance>())...,
                      std::declval<instance>()})>>
    // 递归探测，每次递归增加一个参数，增加的参数为sizeof...(Indices)
    : arity_impl<Aggregate,
                 std::index_sequence<Indices..., sizeof...(Indices)>> {};
```

递归探测机制：
+ 从空参数列表开始(std::index_sequence\<\>)

+ 每次递归增加一个参数(std::index_sequence\<Indices..., sizeof...(Indices)\>)，比如原来是：std::index_sequence\<\>，下一次递归变为std::index_sequence\<0\>，再下一次递归变为std::index_sequence\<0, 1\>，以此类推。

+ 当参数数量超过Aggregate类型成员数量时，SFINAE使特化版本失效

### 通过一个例子理解计算过程

arity\<T\>()函数模板可以返回任意类型T的成员数量，完全在编译期计算，零运行时开销。

```c++
struct Point { int x; double y; };

// 尝试构造：
Point{
    instance{},  // 匹配x
    instance{},  // 匹配y 
    instance{}   // 触发SFINAE (Point只有2个成员)
}
```

当参数数量=3时构造失败，递归终止，确定arity=2。

以结构体Point为例，逐步解释这个模板特化的执行过程：

+ 第一步, 尝试匹配特化版本（带std::void_t的版本）

```c++
arity_impl<Point, std::index_sequence<>>
```
此时Indices包为空，尝试构造：
```c++
Point{instance{}}
```
构造成功（匹配1个成员），触发递归：

+ 第一次递归：

```c++
arity_impl<Point, std::index_sequence<0>>
```
尝试构造：
```c++
Point{instance{}, instance{}}
```
构造成功（匹配2个成员）, 触发递归

+ 第二次递归：

```c++
arity_impl<Point, std::index_sequence<0,1>>
```

尝试构造：

```c++
Point{instance{}, instance{}, instance{}}
```

构造失败（超过2个参数）, 递归终止。
最终继承arity_impl\<Point, std::index_sequence\<0,1\>\>, size()返回2。

```c++
Point p1 = {};

Point p2 = {1};

Point p3 = {1, 2};

Point p4 = {1, 2, 3}; // error, 超过2个参数
```

通过这个例子，我们可以看到，通过递归探测，可以确定任意类型的成员数量。