---
title: C++使用abseil-cpp遇到的小问题
date: 2024-11-07
authors: [KenForever1]
categories: 
  - C++
labels: []
comments: true
---

## 如何使用abseil-cpp
<!-- more -->
### 编译abseil-cpp

不推荐使用预编译版本，和ABI冲突有关。参考：
[what-is-abi-and-why-dont-you-recommend-using-a-pre-compiled-version-of-abseil](https://github.com/abseil/abseil-cpp/blob/master/FAQ.md#what-is-abi-and-why-dont-you-recommend-using-a-pre-compiled-version-of-abseil)

```bash
cmake -B build -S . -DABSL_BUILD_TESTING=OFF -DCMAKE_INSTALL_PREFIX=./output/ -DCMAKE_CXX_STANDARD=17  -DCMAKE_POSITION_INDEPENDENT_CODE=ON
cmake --build build -- -j8
cd build
make install
```

## 问题解决相关

### 报错1

如果你的libtarget.so使用了libabsl_xxx.a, 如果编译libtarget.so事，提示编译absl库需要指定-FPIC，指定即可。

### 报错2

如果报错libabsl中的符号和gcc中的multi definition。

+ 检查编译选项，确保ABI相同，比如-std17，参考：[what-is-abi-and-why-dont-you-recommend-using-a-pre-compiled-version-of-abseil](https://github.com/abseil/abseil-cpp/blob/master/FAQ.md#what-is-abi-and-why-dont-you-recommend-using-a-pre-compiled-version-of-abseil)

+ 在cmake使用abseil例子可以参考：
CMakeLists.txt
```bash
file(GLOB ABSEIL_LIBS $ENV{THIRD_PARTY}/lib*/libabsl*.a)
# find_package(absl REQUIRED)
add_library(target SHARED
    ${target_SRCS}
)

target_link_libraries(target PRIVATE
    -Wl,--whole-archive
    ${ABSEIL_LIBS}
    -Wl,--no-whole-archive
    -ldl
    -lpthread
)

```