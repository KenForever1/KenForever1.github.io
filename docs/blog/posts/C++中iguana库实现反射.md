---
title: C++技法：iguana序列化库中如何实现enum reflection反射
date: 2025-04-13
authors: [KenForever1]
categories: 
  - cpp
labels: []
pin: true
comments: true
---

我们知道通过反射可以在运行时获取类名、类型等一些信息，然而c++语言本身没有提供类似java这种反射机制。在阅读iguana开源库时，看到了一种EnumRefection的实现，分享给大家。

iguana 是C++17 实现的header only、跨平台、高性能易用的序列化库，包括xml/json/yaml 的序列化和反序列化。

<!-- more -->

iguana 的目标是通过编译期间反射将对象序列化到任意数据格式并大幅提升对象的序列化和反序列化的易用性。

通过模板传入类型T，然后获取__PRETTY_FUNCTION__字符串，通过int默认类型找到T类型所在的开始结束位置，然后通过字符串截取，把string信息拿出来，以实现反射。

定义的函数get_raw_name，通过int进行模板实例化以后返回的字符串是：
```c++
std::string_view get_raw_name() [T = int]
```
可以看到这个字符串中就保存了int的类型信息。

## iguana库实现反射

下面看一下iguana库源码中的实现，iguana是一个c++实现的序列化库, 源码地址：[iguana实现的field_reflection](https://github.com/qicosmos/iguana/blob/master/iguana/field_reflection.hpp)

（注：通过在github.com中加入1s，github1s.com，如上面的链接，可以通过这个工具打开网页版的vscode进行源码阅读）

```cpp
#include <iostream>

template <typename T>
constexpr std::string_view get_raw_name() {
    #ifdef _MSC_VER
    return __FUNCSIG__;
    #else
    return __PRETTY_FUNCTION__;
    #endif
}

template <auto T>
constexpr std::string_view get_raw_name() {
    #ifdef _MSC_VER
    return __FUNCSIG__;
    #else
    return __PRETTY_FUNCTION__;
    #endif
}

template <typename T> // 用于类型
inline constexpr std::string_view type_string() {
    constexpr std::string_view sample = get_raw_name<int>();
    constexpr size_t pos = sample.find("int"); // 找到类型名开始位置
    constexpr std::string_view str = get_raw_name<T>();
    constexpr auto next1 = str.rfind(sample[pos + 3]); // 从右往前找类型结束后的字符，找到类型名结束的位置
    #if defined(_MSC_VER)
    constexpr auto s1 = str.substr(pos + 6, next1 - pos - 6);
    #else
    constexpr auto s1 = str.substr(pos, next1 - pos);
    #endif
    return s1;
}

template <auto T> // 用于值
inline constexpr std::string_view enum_string() {
    constexpr std::string_view sample = get_raw_name<int>();
    constexpr size_t pos = sample.find("int");
    constexpr std::string_view str = get_raw_name<T>();
    constexpr auto next1 = str.rfind(sample[pos + 3]);
    #if defined(__clang__) || defined(_MSC_VER)
    constexpr auto s1 = str.substr(pos, next1 - pos);
    #else
    constexpr auto s1 = str.substr(pos + 5, next1 - pos - 5);
    #endif
    return s1;
}
```

## 通过一个例子学会使用

下面通过一个Person结构体，以及enum Status进行简单测试：
```cpp
struct Person{
    std::string name;    
    int age;
};

enum class Status{
    Good,
    Bad
};
int main(){
    
    constexpr std::string_view sample = get_raw_name<int>();
    std::cout << sample << std::endl;
    
    constexpr std::string_view sample1 = get_raw_name<Status::Good>();
    std::cout << sample1 << std::endl;
    
    std::cout << type_string<float>() << std::endl;
    std::cout << enum_string<12>() << std::endl;
    
    std::cout << type_string<Person>() << std::endl;
    std::cout << enum_string<Status::Good>() << std::endl;
}
```
可以看到实现了值和类型的反射：
```cpp
std::string_view get_raw_name() [T = int]
std::string_view get_raw_name() [T = Status::Good]
float
12
Person
Status::Good
```

可以使用在线网站，运行工具测试验证：[https://cpp.sh/](https://cpp.sh/)。