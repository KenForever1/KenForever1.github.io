---
title: LTO（Link Time Optimization）能够优化静态库吗？动态库呢有何不同
date: 2025-03-30
authors: [KenForever1]
categories: 
  - C++
labels: []
comments: true
---

[LTO](https://llvm.org/docs/LinkTimeOptimization.html)（链接时优化）背后的基本原理是，将 LLVM 的一些优化过程推迟到链接阶段。为什么是链接阶段呢？因为在编译流程中，链接阶段是整个程序（即整套编译单元）能够一次性全部获取的时刻，因此跨编译单元边界的优化成为可能。

<!-- more -->

> The basic principle behind LTO is that some of LLVM's optimization passes are pushed back to the linking stage. 


> Why the linking stage? Because that is the point in the pipeline where the entire program (i.e. the whole set of compilation units) is available at once and thus optimizations across compilation unit boundaries become possible.

我们知道，采用静态库的代码会被完全打包到最终程序中。如果采用静态库，编译器可以优化静态库中的代码吗？比如函数内联，死代码消除等。

答案是可以的，通过添加-flto选项，编译静态库，编译可执行文件。我们可以通过查看汇编，反汇编，查看版本符号，对比可执行文件大小对比优化结果。

## 静态库的例子

```
project/
├── math_lib/          # 静态库目录
│   ├── square.cpp
│   └── square.h
└── main.cpp           # 主程序
```

```c++
// math_lib/square.h
#pragma once
int square(int x);

// math_lib/square.cpp
#include "square.h"
int square(int x) {
    // 待优化的代码示例
    return x * x;
}

// main.cpp
#include <iostream>
#include "math_lib/square.h"
int main() {
    int result = square(5);
    std::cout << "Square: " << result << std::endl;
    return 0;
}
```

预期是，如果LTO优化了静态库，那么在可执行文件中，square()函数的调用将被内联，直接替换为5*5=25的常量计算结果。

## 查看LTO优化结果

### LTO优化编译，同时对比可执行文件大小
```bash
#!/bin/bash
# build.sh
set -e

# 清理旧文件
rm -rf math_lib/*.o *.a main_*

# 编译LTO版本
cd math_lib
g++ -c -flto -O2 square.cpp -o square.o
ar cr libmath.a square.o
cd ..

g++ -flto -O2 main.cpp -Imath_lib -Lmath_lib -lmath -o main_lto

# 编译无LTO版本
cd math_lib
g++ -c -O2 square.cpp -o square_no_lto.o
ar cr libmath_no_lto.a square_no_lto.o
cd ..

g++ -O2 main.cpp -Imath_lib -Lmath_lib -lmath_no_lto -o main_no_lto

echo "对比可执行文件大小:"
ls -lh main_*
```

通过执行build.sh编译，同时对比可执行文件大小：
```bash
对比可执行文件大小:
-rwxr-xr-x 1 ken ken 16K Mar 30 19:37 main_lto
-rwxr-xr-x 1 ken ken 17K Mar 30 19:37 main_no_lto
```

### 生成汇编，查看版本符号

```bash
#!/bin/bash

echo "查看LTO版本符号:"
# 查看LTO版本符号
nm --demangle main_lto | grep square
# 理想结果：无square符号（函数被内联）

echo "查看no_LTO版本符号:"

# 查看无LTO版本符号
nm --demangle main_no_lto | grep square
# 预期输出：存在square符号


echo "生成LTO汇编:"
# 生成带LTO的汇编
g++ -flto -O2 -S main.cpp -Imath_lib -Lmath_lib -lmath -o main_lto.s

echo "生成no_LTO汇编:"
# 生成无LTO的汇编
g++ -O2 -S main.cpp -Imath_lib -Lmath_lib -lmath_no_lto -o main_no_lto.s
```

通过执行show.sh，可以生成汇编，查看版本符号是否带有square 函数。
```bash
bash show.sh
查看LTO版本符号:
查看no_LTO版本符号:
00000000000012b0 T square(int)
生成LTO汇编:
生成no_LTO汇编:
```

### 反汇编查看代码

同时，我们还可以反汇编查看代码，如果被优化了，那么代码中就不会有call square 函数的调用。

```bash
objdump -dC main_no_lto | grep -A10 '<main>'

# 这个是没有优化的结果
0000000000001100 <main>:
    1100:       f3 0f 1e fa             endbr64
    1104:       55                      push   %rbp
    1105:       bf 05 00 00 00          mov    $0x5,%edi
    110a:       53                      push   %rbx
    110b:       48 83 ec 08             sub    $0x8,%rsp
    110f:       e8 9c 01 00 00          call   12b0 <square(int)>
    1114:       48 8d 35 e9 0e 00 00    lea    0xee9(%rip),%rsi        # 2004 <_IO_stdin_used+0x4>
    111b:       48 8d 3d 1e 2f 00 00    lea    0x2f1e(%rip),%rdi        # 4040 <std::cout@GLIBCXX_3.4>
    1122:       89 c3                   mov    %eax,%ebx
```

这个是查看lto优化后的反汇编，发现没有了call square 函数的调用。直接被优化成了mov    $0x19,%esi，也就是直接输出了16进制的0x19对应了十进制数字25。
```bash
objdump -dC main_lto | grep -A10 '<main>'

0000000000001100 <main>:
    1100:       f3 0f 1e fa             endbr64
    1104:       55                      push   %rbp
    1105:       48 8d 35 f8 0e 00 00    lea    0xef8(%rip),%rsi        # 2004 <_IO_stdin_used+0x4>
    110c:       53                      push   %rbx
    110d:       48 83 ec 08             sub    $0x8,%rsp
    1111:       48 8b 3d c8 2e 00 00    mov    0x2ec8(%rip),%rdi        # 3fe0 <std::cout@GLIBCXX_3.4>
    1118:       e8 a3 ff ff ff          call   10c0 <std::basic_ostream<char, std::char_traits<char> >& std::operator<< <std::char_traits<char> >(std::basic_ostream<char, std::char_traits<char> >&, char const*)@plt>
    111d:       be 19 00 00 00          mov    $0x19,%esi
```

使用了lto对静态库也是可以优化的，前提是编译静态库和可执行程序，都需要加上-flto选项。‌

lto可以优化，函数内联‌，LTO允许将静态库中的square()函数内联到调用处，直接替换为5*5=25的常量计算结果。如果主程序未使用某些库函数，LTO可彻底移除相关代码，‌也就是死代码消除。还有很多优化。

### 静态库和动态库LTO优化的区别

+ ‌静态库‌

LTO在链接阶段可对静态库进行全局优化。由于静态库代码在编译时已完全嵌入可执行文件，链接器能跨模块分析代码，实现函数内联、死代码消除等跨文件优化‌。例如，静态库中的未使用函数会被彻底移除，减少最终文件体积。

+ ‌动态库‌
  
动态库本身是独立编译的二进制文件，LTO只能在生成动态库时对其内部代码进行局部优化。主程序链接动态库时，无法跨库边界进行全局优化（如无法内联动态库中的函数）。动态库的优化需依赖其自身的编译参数（如启用-flto）。