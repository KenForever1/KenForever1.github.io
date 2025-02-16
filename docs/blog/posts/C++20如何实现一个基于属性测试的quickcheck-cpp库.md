---
title: C++20如何实现一个基于属性测试的quickcheck-cpp库
date: 2025-01-01
authors: [KenForever1]
categories: 
  - cpp
labels: [quickcheck]
pin: true
---

[TOC]

`quickcheck` 是一个非常强大的工具！`quickcheck` 是一个基于属性测试（Property-based Testing）的 Rust 库，灵感来自于 Haskell 的 QuickCheck 库。它的核心思想是通过自动生成大量随机输入来测试代码的属性，而不是手动编写具体的测试用例。

<!-- more -->

## Property-based Testing方法

### Property-based Testing核心概念

1. **属性测试**：
   - 属性测试是一种测试方法，它关注代码的“属性”或“行为”，而不是具体的输入输出。
   - 例如，对于一个排序函数，属性可以是“排序后的数组应该是非递减的”。

2. **随机输入生成**：
   - `quickcheck` 会自动生成大量随机输入来测试代码。
   - 它支持生成各种类型的随机数据，包括基本类型（如整数、字符串）和复杂类型（如结构体、枚举）。

3. **缩小失败用例**：
   - 如果测试失败，`quickcheck` 会尝试缩小失败的输入，找到一个最小的、仍然能触发错误的输入，方便调试。

### 一个简单的例子

用一个例子来展示一下使用 `quickcheck` 测试一个函数的属性的方式。`reverse` 函数是一个简单的函数，用于反转一个数组。我们定义了一个属性：`reverse(reverse(xs)) == xs`，即对一个数组反转两次应该得到原数组。创建一个 `QuickCheck` 实例。运行属性测试。`quickcheck` 会自动生成随机 `Vec<i32>` 输入，并验证属性是否成立。

```rust
use quickcheck::QuickCheck;
use quickcheck::TestResult;

fn reverse<T: Clone>(xs: &[T]) -> Vec<T> {
    let mut rev = xs.to_vec();
    rev.reverse();
    rev
}

fn main() {
    QuickCheck::new()
        .quickcheck(
            // 测试属性：反转两次应该得到原数组
            "reverse(reverse(xs)) == xs",
            |xs: Vec<i32>| -> bool {
                let rev = reverse(&xs);
                let rev_rev = reverse(&rev);
                rev_rev == xs
            }
        );
}
```

### Property-based Testing优点

- **覆盖更多边界情况**：随机生成输入可以覆盖手动测试难以想到的边界情况。
- **减少测试代码量**：不需要手动编写大量具体的测试用例。
- **自动缩小失败用例**：方便调试和定位问题。

当你需要测试函数的通用属性时,希望覆盖更多的输入可能性,自动化生成测试用例时候，你都可以采用这种方法。

## 用C++20实现一个quickcheck-cpp库

[quickcheck-cpp](https://github.com/KenForever1/quickcheck-cpp)基于 C++20 实现一个简化版的 quickcheck 一个基本的属性测试框架。使用到了fmt库用于print容器类，比如vector。包括如下功能：

（1）随机输入生成：
使用 C++ 的random生成随机数据。支持基本类型（如 int、double、std::string）和自定义类型。

（2）属性测试：
用户提供一个 Lambda 函数作为属性测试函数。会生成随机输入并验证属性是否成立。

（3）收缩失败用例：
如果测试失败，尝试收缩输入，找到最小的失败用例。

使用示例：

```cpp
#include "quickcheck.hpp"

// 示例：测试 reverse 函数的属性
bool testReverseProperty(const std::vector<int> &xs)
{
    std::vector<int> rev = xs;
    std::reverse(rev.begin(), rev.end());
    std::vector<int> revRev = rev;
    std::reverse(revRev.begin(), revRev.end());
    return revRev == xs;
}

int main()
{
    // 测试 reverse 函数的属性
    QuickCheck::check<std::vector<int>>(
        "reverse(reverse(xs)) == xs",
        testReverseProperty);

    return 0;
}
```

## 包括哪些功能特性

### 生成任意值

通过RandomGenerator 模板，用于生成随机数据。通过特化支持如下类型：

+ 基本类型：

char、int、long long、float、double 等基本类型均支持随机生成。

+ 字符串：

std::string 生成长度为 1 到 10 的随机字符串。

+ 容器：

std::vector<T>：生成随机长度的向量，元素类型 T 可以是任意支持的类型。

std::map<K, V>：生成随机长度的映射，键值对由 RandomGenerator<std::pair<K, V>> 生成。

std::set<T>：生成随机长度的集合，元素类型 T 可以是任意支持的类型。

std::unordered_map<K, V>：生成随机长度的无序映射。

+ 元组和键值对：

std::tuple<Ts...>：生成随机元组，元素类型由 RandomGenerator<Ts> 生成。

std::pair<K, V>：生成随机键值对，键和值分别由 RandomGenerator<K> 和 RandomGenerator<V> 生成。

### 收缩失败用例(shrink)

实现收缩失败用例的功能是 quickcheck 的核心特性之一。当测试失败时，我们需要逐步收缩输入，找到最小的、仍然能触发失败的用例。

+ 收缩策略：

对于数值类型（如 int、double），逐步将值减半。

对于容器类型（如 std::vector、std::map），逐步删除元素。

对于复合类型（如 std::tuple），逐步收缩其组成部分。

+ 递归收缩：

对于复杂类型，递归地收缩其子元素。

+ 停止条件：

当输入无法进一步收缩，或收缩后的输入不再触发失败时，停止收缩。

例如，测试函数错误，输出：

```bash
Testing property: reverse(reverse(xs)) == xs
Test failed for input: [-80, 62, 40, -98, -50, 95, -21]
Shrinking input...Minimal failing input: [-80, 62, 40, -98, -50, 95, -21]
```

### 统计测试覆盖率

CoverageTracker 类用于统计测试覆盖率，记录输入的最小值、最大值和测试次数。支持基本类型（如 int）和容器类型（如 std::vector）等。

#### 核心函数

+ update 方法：

更新统计信息，记录输入的范围和分布。

+ print 方法：

输出统计结果，显示测试覆盖的输入范围。

#### 在 QuickCheck 中使用 CoverageTracker

在每次生成输入后，调用 update 方法更新统计信息。在测试结束后，调用 print 方法输出覆盖率统计。

```bash
Testing property: reverse(reverse(xs)) == xs
All tests passed!
Vector Coverage: minSize = 0, maxSize = 9, sizeCount = 100 
Element Coverage: Coverage: min = -100, max = 100, count = 202
```

## C++20 Concept约束

+ RandomGeneratable Concept

要求类型 T 必须实现 RandomGenerator<T>::generate() 和 RandomGenerator<T>::shrink() 方法。用于约束 RandomGenerator 的模板参数。

+ PropertyTestable Concept

要求类型 Func 必须是一个可调用对象，接受 T 类型参数并返回 bool。用于约束属性测试函数的模板参数。

使用概念的好处，一方面，增强代码安全性，在编译时检查模板参数是否满足要求，避免运行时错误。其二，提高代码可读性，明确模板参数的要求，使代码更易于理解。而且，更好的错误提示，如果模板参数不满足概念约束，编译器会给出清晰的错误信息。

## inspired

除了本文实现的[quickcheck-cpp](https://github.com/KenForever1/quickcheck-cpp)，感兴趣的朋友可以看一下下面的项目。他们分别是rust实现和cpp的实现。cpp的实现相比本文介绍的实现更加复杂，功能更丰富，比如集成了Catch、GoogleTest、Boost Test等。

[BurntSushi/quickcheck](https://github.com/BurntSushi/quickcheck): Automated property based testing for Rust (with shrinking).

[emil-e/rapidcheck](https://github.com/emil-e/rapidcheck): QuickCheck clone for C++ with the goal of being simple to use with as little boilerplate as possible.