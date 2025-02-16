---
comments: true
---
## 1 GTest GMock

GMock可以满足如下测试需求：

- mock一个类的虚函数和非虚函数
    
- mock std::function
    

但是如果要测试非类成员函数和虚函数，即c-style函数。

Gmock是处理c++类的，如果要mock一个c-style函数，需要定义Interface的抽象类，然后通过mock类的方式去mock c-stryle函数。

如果要mock一个c语言函数，需要使用基于链接方式的，在测试的时候链接不同的cpp实现库。如果你的程序时c语言实现的，更好的是采用c mock的框架。比如[https://github.com/meekrosoft/fff](https://github.com/meekrosoft/fff)。

参考：

- [https://imageslr.com/2023/gtest.html](https://imageslr.com/2023/gtest.html)
    
- [https://google.github.io/googletest/gmock_cook_book.html](https://google.github.io/googletest/gmock_cook_book.html)
    
- [https://stackoverflow.com/questions/31989040/can-gmock-be-used-for-stubbing-c-functions](https://stackoverflow.com/questions/31989040/can-gmock-be-used-for-stubbing-c-functions)

## 2 c-style function mock

这里介绍两种mock方式，可以mock c-style function.

### 2.1 fff mock框架

使用fff mock框架，只需要引入fff.h头文件，

```cpp
#include "fff.h"

#include <gtest/gtest.h>

DEFINE_FFF_GLOBALS;

// 函数原型

// xxxError xxxInit(const char *arg1);

// 通过mock宏FAKE_VALUE_FUNC，会生成一个xxxInit_fake对象，记录了参数值和返回值，可以更改函数返回值等

FAKE_VALUE_FUNC(xxxError, xxxInit, const char *);

class XClassTests : public testing::Test

{

public:

    // 参考Gtest框架，会在XClassTests test一开始调用SetUp函数

 void SetUp()

 {

  // Register resets

     RESET_FAKE(xxxInit);

  FFF_RESET_HISTORY();

  // non default init

  xxxInit_fake.return_val = 520;

 }

};

TEST(XClassTests, test_xxx_init_mock){

    int err = 0;

    aclInit_fake.return_val = 520;

    // call_xxxInit_func中会去调用xxxInit函数，这样会调用到mock的xxxInit函数

    err = call_xxxInit_func(0, "hello");

    ASSERT_EQ(xxxInit_fake.call_count, 1);

    RESET_FAKE(xxxInit);

}
```

更多例子可以参考：[https://github.com/meekrosoft/fff/tree/master/examples](https://github.com/meekrosoft/fff/tree/master/examples)

### 2.2 借助自定义的同名同参函数

```cpp
// xxx_test.cpp

extern "C"{

    int xxxInit(const char* configPath){

        return 520;

    }

}
```

类似可以实现用自定义的malloc函数和free函数，wrap 系统的malloc和free函数，实现统计内存情况等功能。例如：

[https://github.com/Toddz1/MemLeak](https://github.com/Toddz1/MemLeak)。
## 3 lmock

通过修改插入汇编代码，将mock函数地址设置到rax寄存器，然后jump到rax执行mock函数，可以mock 普通的函数和static函数，目前只实现了x86-64平台。

比如代码中的如下二进制，对应了jump rax。

```bash
"\xff\xe0"  =>  jmp rax
```

注: 可以通过gdb查看汇编和二进制以及源码

```bash
disassemble /r add(int)
```

```
/m 源码和汇编一起排列

/r 还可以看到16进制代码
```

### 3.1 C++ 工程实践(6)：单元测试如何 mock 系统调用

[https://www.cnblogs.com/Solstice/archive/2011/05/16/2047255.html](https://www.cnblogs.com/Solstice/archive/2011/05/16/2047255.html)

参考资料：

- [https://github.com/wangyongfeng5/lmock/tree/main](https://github.com/wangyongfeng5/lmock/tree/main)
    
- [https://mp.weixin.qq.com/s?__biz=MjM5ODYwMjI2MA==&mid=2649760560&idx=2&sn=5b403d105557e2adc0535c4e619eefe2&chksm=beccb04b89bb395d8ee2565f8d838c54e2a203d5828faf74a62e7a3a9f7fb48da94ca581dbb6#rd%20%E5%A6%82%E6%9C%89%E4%BE%B5%E6%9D%83%E8%AF%B7%E8%81%94%E7%B3%BB:admin#unsafe.sh](https://mp.weixin.qq.com/s?__biz=MjM5ODYwMjI2MA==&mid=2649760560&idx=2&sn=5b403d105557e2adc0535c4e619eefe2&chksm=beccb04b89bb395d8ee2565f8d838c54e2a203d5828faf74a62e7a3a9f7fb48da94ca581dbb6#rd%20%E5%A6%82%E6%9C%89%E4%BE%B5%E6%9D%83%E8%AF%B7%E8%81%94%E7%B3%BB:admin#unsafe.sh)
    
- [https://shell-storm.org/online/Online-Assembler-and-Disassembler/?inst=jmp+rax&arch=x86-64&as_format=inline#assembly](https://shell-storm.org/online/Online-Assembler-and-Disassembler/?inst=jmp+rax&arch=x86-64&as_format=inline#assembly)
    
- [https://www.cnblogs.com/Solstice/archive/2011/05/16/2047255.html](https://www.cnblogs.com/Solstice/archive/2011/05/16/2047255.html)