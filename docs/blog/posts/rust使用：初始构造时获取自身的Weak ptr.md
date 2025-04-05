---
title: rust使用：Arc new_cyclic 构建自身的弱引用指针
date: 2025-04-05
authors: [KenForever1]
categories: 
  - rust
labels: []
comments: true
---
## Arc new_cyclic 构建自身的弱引用指针

创建一个新的`Arc<T>`，同时为你提供对分配的`Weak<T>`，以便你构建一个持有对自身弱指针的`T`。
通常，直接或间接循环引用自身的结构不应持有对自身的强引用，以防止内存泄漏。使用此函数，你可以在`T`初始化期间（在`Arc<T>`创建之前）访问弱指针，以便你可以克隆并将其存储在`T`内部。

<!-- more -->

```rust

use std::sync::{Arc, Weak};

struct Gadget {
    me: Weak<Gadget>,
}

impl Gadget {
    /// Constructs a reference counted Gadget.
    fn new() -> Arc<Self> {
        // `me` is a `Weak<Gadget>` pointing at the new allocation of the
        // `Arc` we're constructing.
        Arc::new_cyclic(|me| {
            // Create the actual struct here.
            Gadget { me: me.clone() }
        })
    }

    /// Returns a reference counted pointer to Self.
    fn me(&self) -> Arc<Self> {
        self.me.upgrade().unwrap()
    }
}
```

看一个例子，再注册到一个Vec中保持了Sampler的Weak引用，而这个引用就通过Arc new_cyclic 构建的弱引用指针。

详细使用参考[KenForever1/bvar-rust](https://github.com/KenForever1/bvar-rust/blob/main/examples/create_weak_when_init.rs)项目中用法。

```rust
use std::sync::{Arc, Weak, Mutex};

// 全局注册中心
lazy_static::lazy_static! {
    static ref REGISTRY: Mutex<Vec<Weak<dyn Sampler>>> = Mutex::new(Vec::new());
}

trait Sampler: Send + Sync {
    fn sample(&self);
}

struct MySampler {
    weak_self: Mutex<Option<Weak<Self>>>, // 存储自身的弱引用
    data: u32,
}

impl MySampler {
    // 关键：使用 new_cyclic 在构造期间获取弱引用
    pub fn new(data: u32) -> Arc<Self> {
        Arc::new_cyclic(|weak| {
            let this = Self {
                weak_self: Mutex::new(Some(weak.clone())), // ✅ 正确获取弱引用
                data,
            };
            this.after_init();
            this
        })
    }

    // 初始化后操作
    fn after_init(&self) {
        let guard = self.weak_self.lock().unwrap();
        let weak = guard.as_ref().unwrap().clone();
        REGISTRY.lock().unwrap().push(weak); // 安全注册
    }
}

impl Sampler for MySampler {
    fn sample(&self) {
        println!("Sampling data: {}", self.data);
    }
}


fn main () {
    {
        let sampler = MySampler::new(42);
        let registry = REGISTRY.lock().unwrap();
    
        assert_eq!(registry.len(), 1);
        assert!((*registry)[0].upgrade().is_some()); // 弱引用有效
    
    }
}
```

## Mutex 成员变量的结构体实现 Clone 

Mutex 本身没有实现 Clone trait，自动派生要求所有字段都实现 Clone。

```rust
#[derive(Clone)] // ❌ 会编译失败
struct BadClone {
    mutex_data: Mutex<String>
}
```

```rust
use std::sync::{Arc, Mutex};

/// Mutex 成员变量的结构体实现 Clone 

#[derive(Debug)]
struct ThreadSafeData {
    counter: Mutex<i32>,      // 被互斥锁保护的计数器
    data: Mutex<Vec<String>>, // 被互斥锁保护的复杂数据
}

impl ThreadSafeData {
    pub fn new(init_val: i32) -> Self {
        Self {
            counter: Mutex::new(init_val),
            data: Mutex::new(vec!["default".to_string()]),
        }
    }
}

// 手动实现 Clone（无法通过 #[derive] 自动实现）
impl Clone for ThreadSafeData {
    fn clone(&self) -> Self {
        // 安全获取锁并克隆数据
        let counter_val = *self.counter.lock().unwrap(); // 解引用获取 i32
        let data_clone = self.data.lock().unwrap().clone(); // 克隆 Vec<String>
        
        ThreadSafeData {
            counter: Mutex::new(counter_val), // 创建新 Mutex
            data: Mutex::new(data_clone),     // 创建新 Mutex
        }
    }
}


fn test_clone() {
    let original = ThreadSafeData::new(5);
    original.data.lock().unwrap().push("test".to_string());

    let cloned = original.clone();
    
    // 验证计数器克隆
    assert_eq!(*cloned.counter.lock().unwrap(), 5);
    
    // 修改原数据验证独立性
    *original.counter.lock().unwrap() += 1;
    original.data.lock().unwrap().push("modified".to_string());
    
    // 验证克隆体数据独立
    assert_eq!(*cloned.counter.lock().unwrap(), 5); // 原值保持
    assert_eq!(cloned.data.lock().unwrap().len(), 2); // 原始克隆时的数据
}
```



### Arc和Mutex实现线程共享

当需要跨线程共享克隆体时，可以结合 Arc 使用

```rust
#[derive(Clone)]
struct ArcData {
    shared: Arc<Mutex<Vec<u8>>>, // Arc 可克隆
}

impl ArcData {
    // 克隆时共享同一个 Mutex
    pub fn new() -> Self {
        Self {
            shared: Arc::new(Mutex::new(vec![])),
        }
    }
}

fn test_share(){
    // 此时克隆体会共享同一个 Mutex
    let a = ArcData::new();
    let b = a.clone();
    a.shared.lock().unwrap().push(1);
    assert_eq!(b.shared.lock().unwrap().len(), 1); // 共享修改
}
```