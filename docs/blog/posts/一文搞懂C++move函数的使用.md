---
title: 一文搞懂C++move函数的使用
date: 2025-01-25
authors: [KenForever1]
categories: 
  - C++
labels: []
comments: true
---
## C++中move函数是什么？

看一个简单的例子，看看移动构造和移动赋值函数的如何调用的？

<!-- more -->

```c++
#include <iostream>
#include <utility>

class ResourceHolder {
private:
    int* data;
    size_t size;
    
public:
    // 普通构造函数
    ResourceHolder(size_t s) : size(s), data(new int[s]) {
        std::cout << "普通构造函数" << std::endl;
    }
    
    // 析构函数
    ~ResourceHolder() {
        delete[] data;
    }
    
    // 移动构造函数
    ResourceHolder(ResourceHolder&& other) noexcept 
        : data(std::exchange(other.data, nullptr)), 
          size(std::exchange(other.size, 0)) {
        std::cout << "移动构造函数" << std::endl;
    }
    
    // 移动赋值运算符
    ResourceHolder& operator=(ResourceHolder&& other) noexcept {
        if (this != &other) {
            delete[] data;
            data = std::exchange(other.data, nullptr);
            size = std::exchange(other.size, 0);
            std::cout << "移动赋值运算符" << std::endl;
        }
        return *this;
    }
    
    // 禁用拷贝构造和拷贝赋值
    ResourceHolder(const ResourceHolder&) = delete;
    ResourceHolder& operator=(const ResourceHolder&) = delete;
};

int main() {
    ResourceHolder a(10);  // 普通构造
    ResourceHolder b(std::move(a));  // 移动构造
    
    ResourceHolder c(5);  // 普通构造
    c = std::move(b);  // 移动赋值
    
    return 0;
}

// 普通构造函数
// 移动构造函数
// 普通构造函数
// 移动赋值运算符
```
我们看到通过std::move()函数，就可以调用到移动构造函数和移动赋值运算符。那std::move()函数是什么呢？为啥经过它就可以调用到移动构造函数和移动赋值运算符呢？

```c++
template <typename T>
constexpr typename std::remove_reference<T>::type&& move(T&& arg) noexcept {
    return static_cast<typename std::remove_reference<T>::type&&>(arg);
}
```
std::move本质只是进行了一次static_cast，即类型转换。因此，前面std::move(b)，通过std::move改变了返回的数据类型，导致传入ResourceHolder的operator=的参数类型变成了ResourceHolder&&，因此根据参数匹配规则，最终会调用operator=(ResourceHolder&&) 这个移动赋值运算符，在函数内部可以将b的所有权转移了。

## 如何使用move函数？

什么时候使用move？
* 调用函数要有对应的移动语义的函数版本。也就是上面的移动构造函数和移动赋值运算符。
* 函数调用完成后，原对象就不再拥有资源所有权了，不应该用这个对象再访问资源了。
* 移动语义可以提升性能，避免拷贝资源

什么时候不用move？
1. 基本数据类型（如int、float），不要用std::move，没有性能收益，反而可能让代码更难理解
2. 编译器已优化返回值的场景（RVO）不用std::move


## std::make_move_iterator的使用

std::make_move_iterator是C++11引入的迭代器适配器工具函数，用于将普通迭代器转换为移动迭代器，从而在算法中实现元素的移动语义而非拷贝。

```c++
#include <iostream>
#include <list>
#include <vector>
#include <iterator>
#include <algorithm>

int main() {
    std::list<std::string> src{"one", "two", "three"};
    std::vector<std::string> dst;

    // 使用移动迭代器范围构造
    dst.assign(std::make_move_iterator(src.begin()),
               std::make_move_iterator(src.end()));

    std::cout << "size : " << src.size() << std::endl;
    std::cout << "移动后src元素: ";
    for (const auto& s : src) std::cout << "\"" << s << "\" ";  // 为空字符串

    std::cout << "\nsize : " << dst.size() << std::endl;
    std::cout << "\n移动后dst元素: ";
    for (const auto& s : dst) std::cout << "\"" << s << "\" ";  // 正常字符串

    return 0;
}
// size : 3
// 移动后src元素: "" "" "" 
// size : 3
// 移动后dst元素: "one" "two" "three"
```
通过包装输入迭代器，使解引用操作返回右值引用（等效于对元素调用std::move），容器间高效转移数据（如vector<string>迁移时避免字符串拷贝）。
```c++
typedef typename conditional<is_reference<__base_ref>::value,
                          typename remove_reference<__base_ref>::type&&,
                          __base_ref>::type              reference;

reference operator*() const
      { return static_cast<reference>(*_M_current); }
```
