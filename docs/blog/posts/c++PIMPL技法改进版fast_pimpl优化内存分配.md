---
title: Fast PIMPL 又是什么技法？
date: 2025-01-01
authors: [KenForever1]
categories: 
  - cpp
labels: []
pin: true
comments: true
---

## Fast PIMPL 又是什么技法？

前面我们讲解了PIMPL技法，现在又来一个Fast PIMPL，这又是什么东东！从字面意思猜，这肯定是PIMPL的升级版改进版勒，肯定解决了一些PIMPL技法存在的问题。

恭喜你，答对了！！！
<!-- more -->
[‌Fast Pimpl‌](https://en.wikibooks.org/wiki/More_C%2B%2B_Idioms/Fast_Pimpl) 是 PIMPL 模式的一种优化实现方式，是为了解决传统 PIMPL 模式中因动态内存分配（如 std::unique_ptr）导致的性能开销问题。通过这种方式减少堆内存分配次数或优化内存布局，提升访问效率‌。

前面提到的PIMPL方法，无论是原始指针的方式、还是unique_ptr的方式都是动态内存分配的。

要解决这个问题就需要：
+ 避免动态内存分配‌：PIMPL 需要为 Impl 对象动态分配内存，而 Fast Pimpl 可能通过栈分配或内存池技术降低开销‌。
+ ‌内存布局优化‌：将 Impl 对象直接嵌入主类或通过固定大小的缓冲区存储，减少指针间接访问的开销‌。

Fast PIMPL这种模式在高性能或内存受限的环境中经常被使用，用于解耦、隐藏实现细节。

## 如何实现呢？

```c++
// Wrapper.hpp
struct Wrapper {
    Wrapper();
    ~Wrapper();
    
    // deprecated in C++23
    // 分配固定size的内存（这里只是一个例子32，具体类需要指定具体Size）
    // 对其要求，alignof(std::max_align_t) = 16，所有的scalar type都可以按照这个值对其
    std::aligned_storage_t<32, alignof(std::max_align_t)> storage;
    
    struct Wrapped; // forward declaration
    Wrapped* handle;
};
```

```c++
// Wrapper.cpp
struct Wrapper::Wrapped {
};

Wrapper::Wrapper() {
    static_assert(sizeof(Wrapped) <= sizeof(this->storage) , "Object can't fit into local storage");
    // 采用placement new构造Wrapped
    this->handle = new (&this->storage) Wrapped();
}

Wrapper::~Wrapper() {
    // 显示的析构
    handle->~Wrapped();
}
```

请注意，不需要指向包装类实例的句柄。为了减少内存占用，可以通过辅助函数访问包装类。

```c++
static Wrapper::Wrapped* get_wrapped(Wrapper* wrapper) {
    // c++17 compatible
    return std::launder(reinterpret_cast<Wrapper::Wrapped*>(&wrapper->storage));
}
```

完整代码访问[KenForever1/cpp_idioms/fast_pimpl](https://github.com/KenForever1/cpp_idioms/tree/main/pimpl_about/fast_pimpl)

你还可以查看[sqjk/pimpl_ptr](https://github.com/sqjk/pimpl_ptr?tab=readme-ov-file)。

注意的是，与任何其他优化一样，只有在分析和经验证明在你的情况下确实需要额外的性能提升时，才使用这个方法。

## 额外知识扩充

### alignment 对其要求

每个类型在内存中都有对其要求，表示此类型的对象可以被分配的连续地址之间的字节数。比如：

```c++
#include <iostream>
 
// objects of type S can be allocated at any address
// because both S.a and S.b can be allocated at any address
struct S
{
    char a; // size: 1, alignment: 1
    char b; // size: 1, alignment: 1
}; // size: 2, alignment: 1
 
// objects of type X must be allocated at 4-byte boundaries
// because X.n must be allocated at 4-byte boundaries
// because int's alignment requirement is (usually) 4
struct X
{
    int n;  // size: 4, alignment: 4
    char c; // size: 1, alignment: 1
    // three bytes of padding bits
}; // size: 8, alignment: 4 
 
int main()
{
    std::cout << "alignof(S) = " << alignof(S) << '\n'
              << "sizeof(S)  = " << sizeof(S) << '\n'
              << "alignof(X) = " << alignof(X) << '\n'
              << "sizeof(X)  = " << sizeof(X) << '\n';
}
// alignof(S) = 1
// sizeof(S)  = 2
// alignof(X) = 4
// sizeof(X)  = 8
```

### 预先分配静态内存，placement new原地构造

通过前面的std::aligned_storage_t可以预先分配Size和Align要求的一段静态内存。分配的内存要求是Align的约数.

c++的std::aligned_storage_t相当于：
```c++
template<std::size_t Len, std::size_t Align = /* default alignment not implemented */>
struct aligned_storage
{
    struct type
    {
        alignas(Align) unsigned char data[Len];
    };
};
```

```c++
#include <cstddef>
#include <iostream>
#include <new>
#include <string>
#include <type_traits>
 
template<class T, std::size_t N>
class static_vector
{
    // Properly aligned uninitialized storage for N T's
    std::aligned_storage_t<sizeof(T), alignof(T)> data[N];
    std::size_t m_size = 0;
 
public:
    // Create an object in aligned storage
    template<typename ...Args> void emplace_back(Args&&... args)
    {
        if (m_size >= N) // Possible error handling
            throw std::bad_alloc{};
 
        // Construct value in memory of aligned storage using inplace operator new
        ::new(&data[m_size]) T(std::forward<Args>(args)...);
        ++m_size;
    }
 
    // Access an object in aligned storage
    const T& operator[](std::size_t pos) const
    {
        // Note: std::launder is needed after the change of object model in P0137R1
        return *std::launder(reinterpret_cast<const T*>(&data[pos]));
    }
 
    // Destroy objects from aligned storage
    ~static_vector()
    {
        for (std::size_t pos = 0; pos < m_size; ++pos)
            // Note: std::launder is needed after the change of object model in P0137R1
            std::destroy_at(std::launder(reinterpret_cast<T*>(&data[pos])));
    }
};
 
int main()
{
    static_vector<std::string, 10> v1;
    v1.emplace_back(5, '*');
    v1.emplace_back(10, '*');
    std::cout << v1[0] << '\n' << v1[1] << '\n';
}
// output
// *****
// **********
```
在析构函数中，std::launder 是 C++17 引入的低级工具，用于‌显式告知编译器内存中某地址处的对象已合法存在‌，解决因编译器优化或隐式对象生命周期管理导致的未定义行为（UB）问题‌。