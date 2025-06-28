---
title: C++技法：如何为一个类动态附加任意类型数据
date: 2025-06-22
authors: [KenForever1]
categories: 
  - C++
labels: []
comments: true
---

## C++技法：如何为一个类动态附加任意类型数据

在设计一个类时，类的成员就固定下来了。但是某些场景下，需要扩展类的成员，但是又不想改变类的定义。这种情况怎么办呢？
<!-- more -->
有一些方法：
+ 在设计类时，增加一些保留字段，留待后面使用。
+ 在类中可以存储std::string，存储json数据之类的，如果要增加数据，可以存储到string中。
+ 另外就是本文介绍的，利用类型擦除技术，通过std::shared_ptr\<void\>存储数据，可以存储任意类型的数据。

C++可以通过std::shared_ptr\<void\>实现为一个类动态附加任意类型数据。可以用于需要存储不同类型数据但希望统一管理的场景。
std::shared_ptr\<void\>通过类型擦除机制，可以持有任意类型的指针，同时利用引用计数自动管理内存生命周期。

```c++
std::shared_ptr<void> data = std::make_shared<int>(42); // 存储int类型
data = std::make_shared<std::string>("Hello"); // 动态替换为string类型
```
使用时需通过std::dynamic_pointer_cast或std::static_pointer_cast将void指针转换回具体类型，确保类型安全。

```c++
auto strPtr = std::static_pointer_cast<std::string>(data);
if (strPtr) { /* 安全使用 */ }
```

## 一个例子看超实用的使用方法

详细实现，参考开源地址[KenForever1/attachment_set](https://github.com/KenForever1/attachment_set/tree/main), 只需要100行头文件实现。

```c++
// 定义容器，可以是你需要扩展的类的成员变量
AttachmentSet attachments;

// 存储附件
attachments.set<MyClass>("obj", std::make_shared<MyClass>());
attachments.set<int>("value", std::make_shared<int>(42));

// 获取附件
if (auto ptr = attachments.get<MyClass>("obj")) {
    ptr->doSomething();
}

// 类型检查
if (attachments.is<int>("value")) {
    // 安全操作...
}

// 遍历附件
for (auto it = attachments.begin(); it != attachments.end(); ++it) {
    if (it.is<MyClass>()) {
        auto obj = it.get<MyClass>();
    }
}
```

## 核心方法实现

### AttachmentSet类实现

如果把扩展存储的数据称为附件，如果要存储多个附件，肯定要用容器去装。这里可以用vector存储，还能利用上iterator迭代器。

最简单的就是std::vector\<std::shared_ptr\<void\>\>成员变量，也可以存储为any，通过any_cast进行转换，用于存储附件。

除了存储数据，还要存储查找的索引Key，以及用于对比类型的typeid, 也可以是typeid(T).hash_code()。

```c++
class AttachmentSet {
    struct TypeInfo {
        // 通过typeid(T).hash_code()匹配和校验类型是否正常
        size_t type;
        // 类型转换，在存储附件时绑定该函数
        std::function<std::shared_ptr<void>(const std::any&)> caster;
    };

    struct Attachment {
        // 附件名称
        std::string name;
        // 附件数据
        std::any data;
        // 附件类型信息
        TypeInfo type_info;
    };

    // 存储附件容器
    using Storage = std::vector<Attachment>;
}
```

### set方法实现

```c++
template<typename T>
void set(const std::string& name, std::shared_ptr<T> data) {
    TypeInfo type_info{
        // 类型信息
        typeid(T).hash_code(),
        // 类型转换函数，杜绝错误类型转换，避免std::bad_any_cast异常
        [](const std::any& a) { return std::any_cast<std::shared_ptr<T>>(a); }
    };

    auto it = find(name);
    if (it != storage_.end()) {
        it->data = data;
        it->type_info = type_info;
    } else {
        storage_.push_back({name, data, type_info});
    }
}
```

### get方法实现

```c++
// 通过name 查找
Storage::const_iterator find(const std::string& name) const {
    return std::find_if(storage_.begin(), storage_.end(),
        [&name](const auto& a) { return a.name == name; });
}

template<typename T>
std::shared_ptr<T> get(const std::string& name) const {
    auto it = find(name);
    // 找不到或者类型不匹配，返回nullptr
    if (it == storage_.end() || it->type_info.type != typeid(T).hash_code()) {
        return nullptr;
    }
    // 获取数据
    return std::static_pointer_cast<T>(it->type_info.caster(it->data));
}
```

### 迭代器实现

借助vector，很容器实现迭代器，并且实现了name()、is()、get()方法。
```c++
class Iterator {
    Storage::const_iterator it_;
public:
    explicit Iterator(Storage::const_iterator it) : it_(it) {}
    
    const std::string& name() const { return it_->name; }
    
    template<typename T>
    bool is() const { return it_->type_info.type == typeid(T).hash_code(); }
    
    template<typename T>
    std::shared_ptr<T> get() const {
        if (!is<T>()) return nullptr;
        // 获取数据, 从std::any转换成std::shared_ptr<T>
        return std::static_pointer_cast<T>(it_->type_info.caster(it_->data));
    }
    
    Iterator& operator++() { ++it_; return *this; }
    bool operator!=(const Iterator& other) const { return it_ != other.it_; }
};
```