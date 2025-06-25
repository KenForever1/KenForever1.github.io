---
title: 一文搞懂c++中rvo返回值优化如何使用
date: 2025-01-26
authors: [KenForever1]
categories: 
  - C++
labels: []
comments: true
---

## 一个例子搞懂返回值优化

现代C++编译器（C++11及以上）会进行返回值优化（Return Value Optimization, RVO）和命名返回值优化（Named Return Value Optimization, NRVO）。
<!-- more -->
其作用是避免创建临时对象，避免拷贝/赋值构造函数的调用，从而提升性能。

```c++
#include <iostream>

class BigObj {
 public:
  BigObj() {
    printf("construction …\n");
  }
  BigObj(const BigObj& s) {
    printf("Expensive copy …\n");
  }
  BigObj& operator=(const BigObj& s) {
    printf("Expensive assignment …\n");
    return *this;
  }
  ~BigObj() {
    printf("destruction …\n");
  }

  static BigObj BigObjFactory() {
    BigObj local;
    return local;
  }

  // 不同分支返回的是同一个对象，编译器可以优化。
  static BigObj BranchFactorySameObj(bool flag) {
    BigObj a;
    if (flag) return a;
    else return a;
  }
};
```

BigObj::BigObjFactory()函数返回一个局部对象local，而main()函数中直接使用这个返回值来构造obj。编译器会优化这个过程，直接将local构造在obj的内存位置上，**避免了复制构造函数的调用**。

```c++
int main(){
    // 返回值优化
    BigObj obj = BigObj::BigObjFactory();
    // construction …
    // destruction …

    // 分支返回值优化
    BigObj obj =  BigObj::BranchFactorySameObj(true);
    // construction …
    // destruction …
    return 0;
}
```

正如上一篇[一文搞懂c++move函数的使用]中提到，由于返回值优化的存在，返回值不需要使用std::move。

编译器已优化返回值的场景（RVO）不用std::move。

```c++
// 不好的用法
static BigObj BigObjFactory() {
  BigObj local;
  return std::move(local);
}
```

## 无法进行rvo优化的情况

1. 赋值已经存在的对象
当返回的对象用于赋值给已存在的对象时（而非直接初始化新对象），编译器会调用赋值运算符而非拷贝构造函数，此时优化不会触发。
```c++
// 没有返回值优化
BigObj obj;
obj = BigObj::BigObjFactory();

// 无法优化的情况，打印：
// construction …
// construction …
// Expensive assignment …
// destruction …
// destruction …
```
2. 多分支返回不同对象

若函数存在多个返回路径（如if-else分支），且返回不同的命名对象，编译器可能无法确定优化目标而放弃优化。

```c++
    static BigObj BranchFactory(bool flag) {
        BigObj a, b;
        if (flag) return a;  // 无法优化
        else return b;       // 无法优化
    }

// 无法优化的情况，打印：
// BigObj obj =  BigObj::BranchFactory(true);
// construction …
// construction …
// Expensive copy …
// destruction …
// destruction …
// destruction …
```
不同分支返回的是同一个对象，编译器可以优化。