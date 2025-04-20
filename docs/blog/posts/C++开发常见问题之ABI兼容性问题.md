---
title: C++开发常见问题之ABI兼容性问题
date: 2024-11-07
authors: [KenForever1]
categories: 
  - C++
labels: []
comments: true
---

## 背景知识

你在C++开发过程中遇到过下面的问题吗？
<!-- more -->
```bash
/usr/bin/ld: libshared_lib_b.so: undefined reference to `A::get_message[abi:cxx11]()'
collect2: error: ld returned 1 exit status
```

这个问题主要是因为ABI不一致造成的。ABI（Application Binary Interface 应用程序二进制接口）‌，确保不同编译器、库或系统编译的二进制模块能正确协同工作‌。


在**GCC 5.1版本**中，libstdc++引入了一种新的库应用程序二进制接口（ABI），其中包含**std::string和std::list的新实现**。这些更改是为了符合2011年C++标准，该标准禁止使用写时复制（Copy-On-Write）字符串，并要求list跟踪其大小。

为了对链接到libstdc++的现有代码保持向后兼容性，该库的soname没有改变，并且旧的实现仍然与新的实现并行支持。这是通过在内联命名空间中定义新的实现来实现的，这样它们在链接时有不同的名称，例如，新版本的std::list<int>实际上被定义为std::__cxx11::list<int>。由于新实现的符号具有不同的名称，因此两个版本的定义可以同时存在于同一个库中。

我们可以通过**宏_GLIBCXX_USE_CXX11_ABI**进行控制，使用新的ABI还是旧的ABI。还要注意的是**操作系统的不同发现版本，宏_GLIBCXX_USE_CXX11_ABI的默认值不一样**，比如Ubuntu默认为1（采用新ABI），而CentOS默认为0（采用旧ABI）。

可以读一下gcc关于[libstdc++ using_dual_abi的介绍](https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_dual_abi.html)。


## 如何判断一个库采用的ABI版本

```bash
nm -C libxxx.so | grep string

# 采用的ABI=1，新版本
std::__cxx11::basic_string

# 采用的ABI=0，老版本
std::string
```

```bash
nm -C ./build_new/libstatic_lib_a.a  | grep get_message
0000000000000000 T A::get_message[abi:cxx11]()

nm -C ./build_old/libstatic_lib_a.a  | grep get_message
0000000000000000 T A::get_message()
```

```bash
nm -C ./build_new/libstatic_lib_a.so  | grep get_message
000000000000249a T A::get_message[abi:cxx11]()

nm -C ./build_old/libstatic_lib_a.so | grep get_message
00000000000013fa T A::get_message()
```

## 通过一个例子来理解ABI不兼容

不管你是采用静态库、或者动态库链接，只要ABI不一致，都会存在这个问题。下面我们通过一个例子来看一下。
下面的代码创建了一个可执行文件Main，链接动态库B，动态库B链接静态库A，静态库A采用不同的ABI。
或者动态库链接动态库A，但是A采用不同的ABI进行编译。
```c++
// static_lib_a/a.cpp
#include "a.h"

std::string A::get_message() {
    return "Hello from static lib (ABI=" 
    #if _GLIBCXX_USE_CXX11_ABI
           "1)";
    #else
           "0)";
    #endif
}

// static_lib_a/a.h
#pragma once
#include <string>

class A {
public:
    static std::string get_message();
};

// shared_lib_b/b.cpp
#include "a.h"
#include <string>

extern "C" {
    const char* get_message_wrapper() {
        static std::string msg = A::get_message();
        return msg.c_str();
    }
}

// main.cpp
#include <iostream>

extern "C" const char* get_message_wrapper();

int main() {
    std::cout << get_message_wrapper() << std::endl;
    return 0;
}
```
编译采用的CMakeLists.txt如下：
```bash
# CMakeLists.txt
cmake_minimum_required(VERSION 3.12)
project(ABIDemo)

option(USE_NEW_ABI "Use C++11 ABI for static library" ON)

# 设置ABI宏定义
if(USE_NEW_ABI)
    set(STATICLIB_ABI_FLAG "-D_GLIBCXX_USE_CXX11_ABI=1")
    message(STATUS "Using C++11 ABI for static library")
else()
    set(STATICLIB_ABI_FLAG "-D_GLIBCXX_USE_CXX11_ABI=0")
    message(STATUS "Using old pre-C++11 ABI for static library")
endif()

# 在编译时可以先解开这部分注释，编译静态库a或者动态库a
# 编译可执行文件和动态库B时，注释这部分
# add_library(static_lib_a STATIC a.cpp a.h)
# target_include_directories(static_lib_a PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# target_compile_definitions(static_lib_a PUBLIC ${STATICLIB_ABI_FLAG})

set(LIBA ${CMAKE_SOURCE_DIR}/build_new/libstatic_lib_a.a)
# set(LIBA ${CMAKE_SOURCE_DIR}/build_old/libstatic_lib_a.a)

add_library(shared_lib_b SHARED b.cpp)
target_include_directories(shared_lib_b PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(shared_lib_b PRIVATE ${LIBA})

add_executable(main main.cpp)
target_link_libraries(main PRIVATE shared_lib_b)
```

```bash
# # 测试新ABI模式
# rm -rf build_new
# mkdir build_new && cd build_new
# cmake .. -DUSE_NEW_ABI=ON
# make
# ./main 

# # 测试旧ABI模式
rm -rf build_old
mkdir build_old && cd build_old
cmake .. -DUSE_NEW_ABI=OFF
make
./main 
```

## 如何解决ABI不兼容

如果您遇到链接器错误，提示对涉及 std::__cxx11 命名空间中类型或 [abi:cxx11] 标签的符号的未定义引用，那么这可能表明您正在尝试链接使用不同的 _GLIBCXX_USE_CXX11_ABI 宏值编译的目标文件。这种情况通常发生在链接到使用较旧版本的 GCC 编译的第三方库时。如果无法使用新的 ABI 重新构建第三方库，那么您将需要使用旧的 ABI 重新编译您的代码。

> 并非所有新应用二进制接口（ABI）的使用都会导致符号名称发生变化。例如，一个包含`std::string`成员变量的类，无论使用旧版还是新版ABI进行编译，其修饰后的名称都相同。为了检测此类问题，新的类型和函数会使用`abi_tag`属性进行标注，以便编译器对使用它们的代码中潜在的ABI不兼容性发出警告。可以通过`-Wabi-tag`选项启用这些警告。

解决办法一般有两种：

+ 采用相同的ABI。

如果你的所有so和可执行文件都是可以从源码自己编译的，你就可以确保它们使用相同的ABI。

+ 将依赖库通过c语言接口导出后，在项目中调用c语言接口。
