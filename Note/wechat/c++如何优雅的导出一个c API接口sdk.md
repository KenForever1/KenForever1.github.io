---
title: c++如何优雅的导出一个c API接口sdk
date: 2025-01-04
authors: [KenForever1]
categories: 
  - cpp
labels: [cpp]
---

<!-- [TOC] -->

一些c++开发的代码，比如sdk，往往需要导出c sdk提供给c用户、或者其它跨语言FFI绑定调用（比如python、lua等）。

<!-- more -->

## 如果全局只有一个实例

在 C++ 中，如果需要通过 C 接口封装一个 SDK，并且在全局范围内只需要一个类对象。如何导出c api sdk，让c调用cpp呢？

### 单例模式
可以考虑使用单例模式（Singleton Pattern）。这种模式可以确保一个类只有一个实例，并提供一个全局访问点。


以下是一个简单的示例，展示如何使用单例模式封装 C++ 类对象，并通过 C 接口进行调用：

```cpp
// MySingleton.h
#ifndef MY_SINGLETON_H
#define MY_SINGLETON_H

class MySingleton {
public:
    // 禁用拷贝构造和赋值操作
    MySingleton(const MySingleton&) = delete;
    MySingleton& operator=(const MySingleton&) = delete;

    // 提供获取实例的静态方法
    static MySingleton& getInstance() {
        static MySingleton instance; // 局部静态变量，保证在第一次调用时创建，并且只创建一次
        return instance;
    }

    // 示例方法
    void doSomething() {
        // 实现功能
    }

private:
    // 私有构造函数
    MySingleton() = default;
    ~MySingleton() = default;
};

#endif // MY_SINGLETON_H
```

```cpp
// MySingletonCInterface.h
#ifndef MY_SINGLETON_C_INTERFACE_H
#define MY_SINGLETON_C_INTERFACE_H

#ifdef __cplusplus
extern "C" {
#endif

// C 接口函数声明
void MySingleton_doSomething();

#ifdef __cplusplus
}
#endif

#endif // MY_SINGLETON_C_INTERFACE_H
```

```cpp
// MySingletonCInterface.cpp
#include "MySingleton.h"
#include "MySingletonCInterface.h"

// C 接口函数实现
void MySingleton_doSomething() {
    MySingleton::getInstance().doSomething();
}
```

在上面的代码中：

1. `MySingleton` 类是一个单例类，其构造函数私有化，并提供了一个 `getInstance()` 静态方法用于访问单例实例。

2. `MySingletonCInterface.h` 和 `MySingletonCInterface.cpp` 提供了 C 接口函数 `MySingleton_doSomething()`。这个函数内部调用了 `MySingleton` 的单例实例的方法。

这样，你可以通过 C 接口调用 C++ 的单例对象方法，从而实现了用 C 接口封装 C++ SDK 的目的。注意，这种方法适用于需要在全局范围内共享一个对象实例的情况。

除了单例模式外，还有其他方式可以在 C++ 中通过 C 接口封装一个 SDK，并在全局范围内只使用一个类对象。

单例方式适用的场景：
1. **全局唯一资源**：例如日志系统、配置管理器等，这些资源通常只需要一个全局实例。
2. **简化接口**：如果确实只需要一个全局对象，单例模式可以简化接口设计。
3. **性能优化**：单例模式避免了频繁创建和销毁对象的开销。

### **静态实例（Static Instance）**

可以在一个源文件中创建一个静态的类实例，并通过 C 接口提供访问这些实例的函数。这种方法适用于不需要动态初始化的简单对象。使用静态实例还需要注意初始化顺序。

```cpp
// MyStaticInstance.h
#ifndef MY_STATIC_INSTANCE_H
#define MY_STATIC_INSTANCE_H

class MyStaticInstance {
public:
    void doSomething();
};

#endif

// MyStaticInstance.cpp
#include "MyStaticInstance.h"
#include <iostream>

void MyStaticInstance::doSomething() {
    std::cout << "Doing something in static instance." << std::endl;
}

// MyStaticInstanceCInterface.h
#ifndef MY_STATIC_INSTANCE_C_INTERFACE_H
#define MY_STATIC_INSTANCE_C_INTERFACE_H

#ifdef __cplusplus
extern "C" {
#endif

void MyStaticInstance_doSomething();

#ifdef __cplusplus
}
#endif

#endif

// MyStaticInstanceCInterface.cpp
#include "MyStaticInstance.h"
#include "MyStaticInstanceCInterface.h"

static MyStaticInstance instance;  // 静态实例

void MyStaticInstance_doSomething() {
    instance.doSomething();
}
```

### **全局对象（Global Object）**

可以直接声明一个全局对象，并通过 C 接口访问它。这种方法简单直接，但全局对象的生命周期始终与程序生命周期一致。

```cpp
// MyGlobalObject.h
#ifndef MY_GLOBAL_OBJECT_H
#define MY_GLOBAL_OBJECT_H

class MyGlobalObject {
public:
    void doSomething();
};

#endif

// MyGlobalObject.cpp
#include "MyGlobalObject.h"
#include <iostream>

void MyGlobalObject::doSomething() {
    std::cout << "Doing something in global object." << std::endl;
}

MyGlobalObject globalObject;  // 全局对象

// MyGlobalObjectCInterface.h
#ifndef MY_GLOBAL_OBJECT_C_INTERFACE_H
#define MY_GLOBAL_OBJECT_C_INTERFACE_H

#ifdef __cplusplus
extern "C" {
#endif

void MyGlobalObject_doSomething();

#ifdef __cplusplus
}
#endif

#endif

// MyGlobalObjectCInterface.cpp
#include "MyGlobalObject.h"
#include "MyGlobalObjectCInterface.h"

extern MyGlobalObject globalObject;

void MyGlobalObject_doSomething() {
    globalObject.doSomething();
}
```

### **智能指针（Smart Pointer）**：

使用 `std::unique_ptr` 或 `std::shared_ptr` 控制对象的生命周期，并通过 C 接口提供访问。这种方式可以更好地管理资源，但复杂度稍高。

```cpp
// MySmartPointerInstance.h
#ifndef MY_SMART_POINTER_INSTANCE_H
#define MY_SMART_POINTER_INSTANCE_H

#include <memory>

class MySmartPointerInstance {
public:
    void doSomething();
};

using MySmartPointerInstancePtr = std::unique_ptr<MySmartPointerInstance>;

MySmartPointerInstancePtr& getSmartPointerInstance();

#endif

// MySmartPointerInstance.cpp
#include "MySmartPointerInstance.h"
#include <iostream>

void MySmartPointerInstance::doSomething() {
    std::cout << "Doing something in smart pointer instance." << std::endl;
}

MySmartPointerInstancePtr& getSmartPointerInstance() {
    static MySmartPointerInstancePtr instance = std::make_unique<MySmartPointerInstance>();
    return instance;
}

// MySmartPointerInstanceCInterface.h
#ifndef MY_SMART_POINTER_INSTANCE_C_INTERFACE_H
#define MY_SMART_POINTER_INSTANCE_C_INTERFACE_H

#ifdef __cplusplus
extern "C" {
#endif

void MySmartPointerInstance_doSomething();

#ifdef __cplusplus
}
#endif

#endif

// MySmartPointerInstanceCInterface.cpp
#include "MySmartPointerInstance.h"
#include "MySmartPointerInstanceCInterface.h"

void MySmartPointerInstance_doSomething() {
    getSmartPointerInstance()->doSomething();
}
```

## 支持多个实例，不透明指针方式

```c++
// my_class.h
#ifdef __cplusplus
extern "C" {
#endif

typedef void* MyClassHandle;

MyClassHandle create_my_class();
void destroy_my_class(MyClassHandle handle);
void set_value(MyClassHandle handle, int value);
int get_value(MyClassHandle handle);

#ifdef __cplusplus
}
#endif

// my_class.cpp
#include "my_class.h"
#include <iostream>

class MyClass {
public:
    MyClass() : value(0) {}
    void setValue(int v) { value = v; }
    int getValue() const { return value; }
private:
    int value;
};

MyClassHandle create_my_class() {
    return new MyClass();
}

void destroy_my_class(MyClassHandle handle) {
    delete static_cast<MyClass*>(handle);
}

void set_value(MyClassHandle handle, int value) {
    static_cast<MyClass*>(handle)->setValue(value);
}

int get_value(MyClassHandle handle) {
    return static_cast<MyClass*>(handle)->getValue();
}

// main.c
#include "my_class.h"
#include <stdio.h>

int main() {
    MyClassHandle handle = create_my_class();
    set_value(handle, 42);
    printf("Value: %d\n", get_value(handle));
    destroy_my_class(handle);
    return 0;
}
```

```bash
g++ -shared -fPIC -o libmy_class.so my_class.cpp

g++ -o main main.c -L. -lmy_class -Wl,-rpath,.
```

**不透明指针**：允许创建多个独立的实例，每个实例可以有自己的状态和数据。这在需要管理多个对象时非常有用。**单例模式**：全局只有一个实例，无法支持多个独立的对象。如果需要多个对象，单例模式就不适用。


**不透明指针**有**更灵活的生命周期管理**：对象的创建和销毁由调用者显式控制（通过 `create` 和 `destroy` 函数），可以更灵活地管理对象的生命周期。 **单例模式**：对象的生命周期通常是全局的，由单例模式内部管理，无法灵活控制。如果需要对对象的生命周期进行精细控制（如延迟初始化、提前销毁等），不透明指针更合适。

部分示例代码:

```
https://github.com/KenForever1/c-sdk-example
```

## 总结

| 特性               | 不透明指针                          | 单例模式                          |
|--------------------|-------------------------------------|-----------------------------------|
| 实例数量           | 支持多个实例                        | 仅支持单个实例                    |
| 封装性             | 更好，完全隐藏实现细节              | 较好，但可能暴露全局状态          |
| 生命周期管理       | 灵活，由调用者控制                  | 固定，全局生命周期                |
| 线程安全性         | 更容易实现线程安全                  | 需要额外同步机制                  |
| 耦合性             | 低耦合                              | 较高耦合                          |
| 可测试性           | 更好，易于隔离测试                  | 较差，全局状态可能导致测试困难    |
| 应用场景           | 广泛，适用于多种场景                | 较窄，适用于全局唯一对象          |

如果应用需要支持多个实例、灵活的生命周期管理、更好的封装性和可测试性，**不透明指针**是更好的选择。而如果应用确实只需要一个全局对象，且不需要复杂的对象管理，**单例模式**可能更简单直接。