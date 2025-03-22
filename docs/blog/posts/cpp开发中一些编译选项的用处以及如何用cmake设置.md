---
title: cpp开发中一些编译选项的用处，以及如何用cmake设置
date: 2025-03-22
authors: [KenForever1]
categories: 
  - cpp
labels: [quickcheck]
pin: true
comments: true
---

## 如何控制静态库和动态库链接行为

你在编译开发一个项目时，有遇到过指定了链接库，但是运行时，却报错：未定义某个符号吗？undefined xxx。

> 在开发中遇到的xxx报错，是经过mangle了的，不方便看，可以采用c++filt工具查看原始符号，命令：
> c++filt xxx 。

<!-- more -->

下面介绍如何控制链接器行为的 ，这里用GCC（GNU Compiler Collection）选项举例，其他编译器可能有细微差别，特别是在使用 `ld` 链接器时对链接过程进行更精细的控制。它们通过 `-Wl,` 前缀传递给链接器。

### `-Wl,--whole-archive`对静态库的效果

在 GCC 编译中，-Wl,--whole-archive 和 -Wl,--no-whole-archive 是用于控制**静态库**链接行为的选项，其核心作用如下：

+ -Wl,--whole-archive‌

强制包含静态库所有符号‌：链接器会将后续指定的**静态库**（.a 文件）中所有目标文件（.o 文件）包含到最终输出文件中，即使这些符号未被显式引用‌。解决因静态库未被完全链接导致的“未定义符号”问题，如构造函数/析构函数未被调用、未使用的函数被优化等。

+ -Wl,--no-whole-archive‌

重置链接行为‌，关闭 --whole-archive 的强制包含效果，恢复链接器默认行为（仅包含被引用的符号）‌。避免后续其他静态库被意外强制包含，导致输出文件体积膨胀‌。

以下 Makefile 片段演示了如何强制链接 libtest.a 的全部内容：

```bash
LDFLAGS += -Wl,--whole-archive -ltest -Wl,--no-whole-archive
```

此配置确保 libtest.a 中所有符号被包含，而后续其他库仍按默认规则链接‌。

-Wl,--whole-archive 和 -Wl,--no-whole-archive 主要用于精细化控制静态库的链接行为，解决符号未包含或依赖问题。‌

### `-Wl,--as-needed`对动态库的效果

+ `-Wl,--as-needed`

这个选项告诉链接器只在实际需要时才链接**动态库**。它可以减少最终可执行文件的依赖项，因为只有那些真正使用到的**共享库**才会被包含。

```bash
gcc -o myapp main.o -Wl,--as-needed -lmylib
```

- `-Wl,--no-as-needed`

使用此选项关闭 `--as-needed` 行为。指定该选项后，链接器会将所有列出的共享库链接到输出文件中，无论是否在程序中实际使用。这可能会导致更大的二进制文件和不必要的依赖。

### 通过真实例子看一看

#### 例如absl是静态库

在CMakeLists.txt文件中就可以这样指定了：

```c++
file(GLOB ABSEIL_LIBS $ENV{THIRD_PARTY}/lib*/libabsl*.a)
# find_package(absl REQUIRED)
add_library(xxx SHARED
    xxx.cpp   
)

target_link_libraries(xxx PRIVATE
    aaa
    -Wl,--whole-archive
    ${ABSEIL_LIBS}
    -Wl,--no-whole-archive
    yyy
    -ldl
    -lpthread
)
```

#### 例如有个动态库

但是编译时不需要指定这个动态库就可以编译通过，但是运行时需要加载它，如果不采用dlopen的方式，那么就可以强制链接这个动态库。

什么样的情景呢？

因为在C++的代码中，很多注册机制，都是通过动态库来实现的。需要加载动态库，在运行时才能完成注册。Register的原理一般是在SO实现中加入了一个静态全局变量，如果so被加载，这个全局变量就会被初始化，调用构造函数中实现的注册逻辑，从而完成注册。

为了解决这个需要强制链接动态库的问题；
```c++
add_library(${PROJECT_NAME} SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/src/xxx.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/yyy.cc
)

target_include_directories(${PROJECT_NAME} PRIVATE
    xx/include
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    xxx-lib
    -Wl,--no-as-needed
    depend-lib
    -Wl,--as-needed
    yyy-lib
    ${GLOG_LIBS}
)
```

## 第三方库版本不一致，头文件接口定义不兼容

在编译可执行文件，以及很多动态链接库时，一定要保证头文件是一致的，包括公共库，比如glog，absl等。

例如存在两份glog，一份在系统/3rdparty下，一份在项目内部，那么编译时，如果不一样。那么so和可执行文件，就会因为二进制不兼容，导致运行时崩溃。

报错现象还不容易排查到，往往是跑到一个随意的位置，触发了段错误或者stack smashing detected等。

如果编译器版本不一样，还可能导致ABI冲突。

## 编译时忽略未定义的符号，运行时找得到就行

如果想要在编译时，忽略未定义的符号，让编译通过，可以使用如下选项：
```c++
target_link_options(${PROJECT_NAME} PRIVATE "-Wl,--unresolved-symbols=ignore-all")
```

## 检查某些函数返回值未没有返回

在有些编译器上可能不会报错，但是有些编译器运行时会报一些内存错误，让人摸不着头脑。很难定位。这时可以添加编译选项：
```c++
add_compile_options(-Wreturn-type -Werror=return-type)
```

## 排查内存问题时使用工具

### 在调试内存问题时可以使用valgrind

使用示例，它可以帮助我们检查内存泄漏，未初始化的内存，越界访问等问题。
```bash
export GLOG_v=1
valgrind --leak-check=full --show-leak-kinds=all --log-file=valgrind_output.txt  ./hello
```

### 编译时加上 -g， asan使用asanitizer工具

cmakelists.txt中加入：
```c++
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}  -fsanitize=address -g -O0 -fno-omit-frame-pointer ")
```