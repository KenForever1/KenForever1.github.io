---
title: cista零拷贝反序列化库实现之swiss_table哈希表实现
date: 2025-06-28
authors: [KenForever1]
categories: 
  - C++
labels: []
comments: true
---

## 实现原理

可以中cista库的介绍中看到如下内容：

> Comes with a serializable high-performance hash map and hash set implementation based on Google's Swiss Table technique.

swiss table的实现可以参考[简单了解下最近正火的SwissTable](https://www.cnblogs.com/apocelipes/p/17562468.html),讲解的很清晰。

以三种实现hashmap的方式，看优缺点：

+ 链表法：指针稳定性，能采取扩容之外的手段阻止查询性能退化，比如把过长链表转换成搜索树。缺点：缓存不够友好，冲突较多的时候缓存命中率较低从而影响性能。

+ 线性探测法：缓存友好，加上冲突会有连锁影响，没有指针稳定性。

<!-- more -->

swiss table是为了改进哈希表本身的结构力求在缓存友好、性能和内存用量上找到平衡。

> swisstable拥有惊人性能的主要原因：它尽量避免线性探测法导致的大量等值比较和链表法会带来的缓存命中率低下，以及在单位时间内它能同时过滤N个（通常为16，因为16x8=128，是大多数平台上SIMD指令专用的向量寄存器的大小）元素，且可以用位运算代替对数据的遍历。这会为哈希表的吞吐量带来质的飞跃。

+ 改进的线性探测方式，分group处理，可以SIMD指令加速处理。

+ swiss table采用了 control控制信息 和 slot存储数据 分离的方式进行存储。相比链式，数据局部性更好。

+ hash一共64位，57位为h1，用于锁定是slot位置。7位为h2，作为key。

+ resize扩容2倍。

控制信息contrl中，下面的定义，表示了三种状态，都是1开头。如果是0开头，后面就是7位h2数据。

```c++
enum ctrl_t : int8_t {
    EMPTY = -128,  // 10000000
    DELETED = -2,  // 11111110
    END = -1  // 11111111
  };
```

abseil的实现中，文章里也提到了，采用了SSE（SIMD指令）进行加速，一次同时比较一个Group（16个slot）。在cista中，没有实现SSE。

在cista中通过位运算并行处理，虽然未使用实际SIMD指令，但通过64位整数运算模拟了并行处理8个控制字节的效果。所有操作都通过**精心设计的位运算**完成，避免了分支和循环。**内存局部性**：控制字节连续存储，充分利用CPU缓存。**状态快速判断**：可以一次性判断整个group的状态(全空/部分匹配等)。通过这些设计，使得Swiss Table在查找、插入等操作上获得高吞吐量。

## find查找实现
find_impl实现如下，可以看到通过h1确定查找的group，然后通过h2进行匹配，判断key相等，返回iterator。
```c++
template <typename Key>
iterator find_impl(Key&& key) {
    auto const hash = compute_hash(key);
    for (auto seq = probe_seq{h1(hash), capacity_}; true; seq.next()) {
        group g{ctrl_ + seq.offset_};
        for (auto const i : g.match(h2(hash))) {
        if (Eq{}(GetKey()(entries_[seq.offset(i)]), key)) {
            return iterator_at(seq.offset(i));
        }
        }
        if (g.match_empty()) {
        return end();
        }
    }   
}
```
由于control信息和slot信息的index是一一对应的，因此iterator中保持的ctrl和entry的指针，获取value，就是*entry_。
```c++
struct iterator {
    reference operator*() const noexcept { return *entry_; }
    ctrl_t* ctrl_{nullptr};
    T* entry_{nullptr};
}
```
### entries_的初始化
WIDTH为8，ALIGNMENT为alignof(T)。
```c++
  static constexpr size_type const WIDTH = 8U;
  static constexpr std::size_t const ALIGNMENT = alignof(T);

  void initialize_entries() {
    self_allocated_ = true;
    // capacity_ * sizeof(T)  保存slot数据
    // (capacity_ + 1U + WIDTH) * sizeof(ctrl_t) 保存control信息，加上后面的padding对齐
    auto const size = static_cast<size_type>(
        capacity_ * sizeof(T) + (capacity_ + 1U + WIDTH) * sizeof(ctrl_t));
    entries_ = reinterpret_cast<T*>(
        CISTA_ALIGNED_ALLOC(ALIGNMENT, static_cast<std::size_t>(size)));
    if (entries_ == nullptr) {
      throw_exception(std::bad_alloc{});
    }
#if defined(CISTA_ZERO_OUT)
    std::memset(entries_, 0, size);
#endif
    ctrl_ = reinterpret_cast<ctrl_t*>(
        reinterpret_cast<std::uint8_t*>(ptr_cast(entries_)) +
        capacity_ * sizeof(T));
    reset_ctrl();
    reset_growth_left();
  }
```

在hashmap的实现中，哈希函数的算法也很重要，一个更加高效“完美哈希函数”，可以减少冲突，减少等值比较，提高性能。