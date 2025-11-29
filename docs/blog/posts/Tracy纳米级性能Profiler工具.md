---
title: Tracy纳米级性能Profiler工具，手动插桩、可视化分析
date: 2025-11-29
authors: [KenForever1]
categories: 
  - LLM推理
labels: []
comments: true
---

Tracy是最近了解到的一个性能分析工具，有一些特性值得关注：开源免费、实时分析、纳秒级精度、跨平台支持（Linux、Win、Macos）、功能丰富CPU、MEM、GPU等、集成方便。

在C++开发中，包括游戏领域一帧一帧分析、推理中每次推理耗时分析、多个线程中关联同一个Context等。目前还在学习中，先简单介绍一下基础使用。

<!-- more -->

## Tracy集成使用例子

CMake集成Tracy，使用CPM.cmake工具从github仓库中获取源码，并编译。

```bash
cmake_minimum_required(VERSION 3.10)
project(hellocpp VERSION 0.1.0 LANGUAGES C CXX)

option(TRACY_ENABLE "Enable Tracy" ON)

add_definitions(-std=c++17)
if(TRACY_ENABLE)
    add_definitions(-DTRACY_ENABLE)
    add_definitions(-DTRACY_FIBERS)
    # add_definitions(-DTRACY_NO_EXIT=ON) // 尝试没有效果
    add_compile_options(-g -O3 -fno-omit-frame-pointer)
endif()

if(NOT TRACY_ENABLE)
    set(TRACY_OPTIONS "TRACY_STATIC ON")
endif()

include(cmake/CPM.cmake)

CPMAddPackage(
    NAME tracy
    GITHUB_REPOSITORY wolfpld/tracy
    GIT_TAG master
    OPTIONS ${TRACY_OPTIONS}
    EXCLUDE_FROM_ALL TRUE
)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
message(status ${tracy_SOURCE_DIR})
add_executable(hellocpp main.cpp
)
target_link_libraries(hellocpp PRIVATE Tracy::TracyClient)
target_include_directories(hellocpp PRIVATE ${tracy_SOURCE_DIR}/public)
set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR}/install_dir)

install(TARGETS ${PROJECT_NAME} 
    RUNTIME DESTINATION ins)
```

```c++
// main.cpp
#include <thread>
#include <unistd.h>

#include "tracy/Tracy.hpp"
#include "tracy/TracyC.h"

#include <iostream>
const char* fiber = "job1";
TracyCZoneCtx zone;


void Line(){
    ZoneScopedN("Line");
    for(int i = 0; i < 10; ++i){
        sleep(1);
        std::cout << "Line:"  << i << std::endl;
        TracyPlot("line", (int64_t)i);
    }
}

void Add(){
    ZoneScopedN("Add");
    sleep(1);
    std::cout << "Add:" << std::endl;
    Line();
}

void Sub(){
    ZoneScopedN("Sub");
    sleep(2);
    void *p = malloc(1024);
    TracyAllocN(p, 1024, "malloc_1024");
    sleep(1);
    free(p);
    TracyFreeN(p, "malloc_1024");
    std::cout << "Sub:" << std::endl;
}


int main()
{
    std::thread t1( [] {
        TracyFiberEnter( fiber );
        TracyCZone( ctx, 1 );
        zone = ctx;
        sleep( 1 );
        Add();
        TracyFiberLeave;
    });
    t1.join();

    std::thread t2( [] {
        TracyFiberEnter( fiber );
        sleep( 1 );
        Sub();
        TracyCZoneEnd( zone );
        TracyFiberLeave;
    });
    t2.join();
}
```

## 相关资料
+ Tracy 的CMake使用，可以参考作者vv工具的使用
[wolfpld/vv](https://github1s.com/wolfpld/vv/blob/master/CMakeLists.txt)

+ Tracy C++2023会议PPT

讲解了如何CMake集成Tracy Client，如何编译Mac、linux、windows三端的Tracy Profiler可视化界面，以及如何使用Tracy Profiler进行性能分析。
https://github.com/CppCon/CppCon2023/blob/main/Presentations/Tracy_Profiler_2024.pdf

+ hdk项目中这个头文件对Tracy的宏定义进行了二次定义：
https://www.sidefx.com/docs/hdk/_u_t___tracing_8h_source.html

原始TracyAlloc等宏定义使用的x,y,z标记参数，看着很不清晰，这个头文件将参数改成了更清晰的ptr,size,name。
```c++
// Memory tracing.
  171 #define utTraceAlloc(ptr,size)              TracyAlloc(ptr,size)
  172 #define utTraceFree(ptr)                    TracyFree(ptr)
  173 #define utTraceAllocN(ptr,size,name)        TracyAllocN(ptr,size,name)
  174 #define utTraceFreeN(ptr,name)              TracyFreeN(ptr,name)
  175 
  176 #define utTraceAllocS(ptr,size,depth)       TracyAllocS(ptr, size, depth)
  177 #define utTraceFreeS(ptr,depth)             TracyFreeS(ptr, depth)
  178 #define utTraceAllocNS(ptr,size,depth,name) TracyAllocNS(ptr, size, depth, name)
  179 #define utTraceFreeNS(ptr,depth,name)       TracyFreeNS(ptr, depth, name)
  180 
```