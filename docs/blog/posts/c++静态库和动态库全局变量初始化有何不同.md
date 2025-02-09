---
title: c++静态库和动态库全局变量初始化有何不同？
date: 2024-10-26
authors: [KenForever1]
categories: 
  - C++
labels: []
---
采用动态链接库的问题：版本升级，调用动态库的程序需不需要重新编译，需要考虑动态链接库二进制ABI兼容性问题，比如是否更改了头文件中的结构体，更改了接口函数，添加了虚函数。动态链接库变更以后，如果没有热更新so，需要替换so，重启可执行程序。

静态库的问题，考虑是否存在资源加载多次的问题。
<!-- more -->
## 1 静态库初始化两次

```cpp
#include <cstdio>

#include "A.h"

#include "iBackend.h"

ClassA A1;    // 我们在这里，定义了全局变量

ClassA::ClassA() {

    printf("ClassA\n");

}

ClassA::~ClassA() {

    printf("~ClassA\n");

}

void ClassA::test() {}

REGISTER_BACKEND(A) // 我们在这里，定义了全局变量
```

```cpp
#include <cstdio>

#include "A.h"

#include "B.h"

int main() {

    printf("main()\n");

    ClassA::test();    // 本行保证和A够成链接关系

    ClassB b;          // 本行保证和B构成链接关系

    printf("main: END\n");

    return 0;

}
```

```
ClassA

call Register : A

ClassA

call Register : A

main()

ClassB

main: END

~ClassB

~ClassA

~ClassA
```

## 2 静态库不初始化全局变量

链接器并没有生成这个自动初始化的代码，因为链接器觉得这几个“没有”被使用的全局对象不需要，所以就没生成。

[https://www.cppblog.com/kevinlynx/archive/2010/01/17/105885.aspx](https://www.cppblog.com/kevinlynx/archive/2010/01/17/105885.aspx)

```cpp
#include <cstdio>

#include "A.h"

#include "B.h"

int main() {

    printf("main()\n");

    // ClassA::test();    // 本行保证和A够成链接关系

    // ClassB b;          // 本行保证和B构成链接关系

    printf("main: END\n");

    return 0;

}
```

```bash
main()

main: END
```

解决方法：

主要包括两种，定义dummy函数加static局部变量，由外部显示调用。或者通过链接.a时设置-Wl,-whole-archive标志。

[https://litaotju.github.io/c++/2020/07/24/Whole-Archive-in-static-lib/](https://litaotju.github.io/c++/2020/07/24/Whole-Archive-in-static-lib/)﻿

如果可执行程序或者动态链接库加入了-Wl,-whole-archive链接.a文件的时候，报错.a用到的xxx等库未定义，可以考虑交换链接顺序。
```bash
# add_definitions("-Wl,-u,needed_symbol")

# add_definitions("-Wl,--whole-archive")

add_library(app SHARED ${AI_SRC})

target_link_libraries(app PUBLIC

    -Wl,-whole-archive

    xxx.a

    -Wl,-no-whole-archive

    GTest::gtest_main

    yyy

    dl)

install (TARGETS app

    RUNTIME DESTINATION lib    

)
```

## 3 动态库不初始化全局变量

```cpp
#include <cstdio>

#include "A.h"

#include "B.h"

int main() {

    printf("main()\n");

    // ClassA::test();    // 本行保证和A够成链接关系

    // ClassB b;          // 本行保证和B构成链接关系

    printf("main: END\n");

    return 0;

}
```

```
main()

main: END
```

## 4 动态库初始化一次全局变量

```cpp
#include <cstdio>

#include "A.h"

#include "B.h"

int main() {

    printf("main()\n");

    ClassA::test();    // 本行保证和A够成链接关系

    ClassB b;          // 本行保证和B构成链接关系

    printf("main: END\n");

    return 0;

}
```

```
ClassA

call Register : A

main()

ClassB

main: END

~ClassB

~ClassA
```

[记一次BUG调试——静态链接库中全局变量/静态变量被重复初始化](https://zhuanlan.zhihu.com/p/491162924)

[https://cloud.tencent.com/developer/ask/sof/97688](https://cloud.tencent.com/developer/ask/sof/97688)

## 5 两个动态库加载同一个静态库

[那个两个动态库引用同一个静态库问题？ - 知乎](https://www.zhihu.com/question/473920577/answer/2013486516)

如果不走dlopen、dlsym，而是在g++命令里显式链接的话 （g++ -o main main.cpp -L./ -lsingleton_lib -ldyn_test_1 -ldyn_test_2），又是只创建一个singleton_test实例

为什么静态局部变量是共用的一份？

`dlopen`传入的`flag`，`RTLD_LOCAL`切换为`RTLD_GLOBAL`，输出就不一样了。

[https://cloud.tencent.com/developer/article/1179871](https://cloud.tencent.com/developer/article/1179871)


[多个c/c++动态库函数同名冲突解决方法_c++同名函数冲突-CSDN博客](https://blog.csdn.net/giveaname/article/details/103353828)

[https://www.cnblogs.com/oloroso/p/6273295.html](https://www.cnblogs.com/oloroso/p/6273295.html)

不同的编译选项，不同的链接器，采用不同加载so的方式（编译和dlopen），不同的编译器（gcc、clang）可能结果都不一样，

> 在静态库中最好不要去存放全局变量，也不要在这里创建单例对象等。

[https://joydig.com/research-global-variables-desctruction-behavior-in-cpp/](https://joydig.com/research-global-variables-desctruction-behavior-in-cpp/)

## 6 交叉编译中忽略so中的undefined xxx

```
target_link_options(xxx PRIVATE "-Wl,--unresolved-symbols=ignore-in-shared-libs")
```
## 其它
```
# 为了使x86平台上so中的static变量和demo使用同一个
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -rdynamic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -rdynamic")
```