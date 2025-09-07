---
title: cista零拷贝反序列化库实现之swiss_table--group匹配位运算
date: 2025-06-28
authors: [KenForever1]
categories: 
  - C++
labels: []
comments: true
---


## Swiss Table中Group的实现解析

这个`group`结构体是Swiss Table实现中的核心组件，用于高效处理哈希表控制位的批量操作。

<!-- more -->

定义核心常量，用于位运算：
```cpp
static constexpr auto MSBS = 0x8080808080808080ULL;  // 每个字节的最高位掩码
static constexpr auto LSBS = 0x0101010101010101ULL;  // 每个字节的最低位掩码
static constexpr auto GAPS = 0x00FEFEFEFEFEFEFEULL;  // 用于计算前导空位的掩码
```
构造函数：
```cpp
using group_t = std::uint64_t;
using h2_t = std::uint8_t;

explicit group(ctrl_t const* pos) noexcept {
  std::memcpy(&ctrl_, pos, WIDTH);
#if defined(CISTA_BIG_ENDIAN)
  ctrl_ = endian_swap(ctrl_);
#endif
}
```
* 从指定位置拷贝8个控制字节到`ctrl_`成员
* 大端系统需要做字节序转换

## 关键匹配方法

### `match`方法 - 匹配特定h2值

```cpp
bit_mask match(h2_t const hash) const noexcept {
  auto const x = ctrl_ ^ (LSBS * hash);
  return bit_mask{(x - LSBS) & ~x & MSBS};
}
```
* 通过位运算同时检查8个控制字节是否匹配给定的h2值
* 算法原理：
  1. `LSBS * hash`：将h2值复制到每个字节
  2. `ctrl_ ^ ...`：异或操作找出匹配的字节
  3. `(x - LSBS) & ~x & MSBS`：精确定位匹配的字节

#### 通过一个例子来理解`match`：

相关例子实现源码[KenForever1/cpp_idioms](https://github.com/KenForever1/cpp_idioms/blob/main/cpp/swiss_table/bit_verify_match.cpp)。

让我们通过一个具体例子来说明match方法的位运算过程。假设：

当前group的8个控制字节为：[0x12, 0x34, 0x56, 0x78, 0x12, 0x9A, EMPTY, DELETED]

对应ctrl_值：0xFE80129A78563412 (小端序)

EMPTY = 0x80, DELETED = 0xFE

要匹配的h2值为：0x12

```c++
=== 开始匹配过程 ===
控制字节(ctrl_) (0xff809a1278563412): 0x12 0x34 0x56 0x78 0x12 0x9a 0x80 0xff 
匹配的h2值: 0x12
LSBS * hash (0x1212121212121212): 0x12 0x12 0x12 0x12 0x12 0x12 0x12 0x12 
x = ctrl_ ^ (LSBS*hash) (0xed9288006a442600): 0x00 0x26 0x44 0x6a 0x00 0x88 0x92 0xed 
x - LSBS (0xec9186ff694324ff): 0xff 0x24 0x43 0x69 0xff 0x86 0x91 0xec 
~x (0x126d77ff95bbd9ff): 0xff 0xd9 0xbb 0x95 0xff 0x77 0x6d 0x12 
(x - LSBS) & ~x (0x000106ff010300ff): 0xff 0x00 0x03 0x01 0xff 0x06 0x01 0x00 
最终结果 mask (0x0000008000000080): 0x80 0x00 0x00 0x00 0x80 0x00 0x00 0x00 

=== 匹配结果 ===
匹配掩码 (0x0000008000000080): 0x80 0x00 0x00 0x00 0x80 0x00 0x00 0x00 
匹配的字节位置: 0 4 
```

### `match_empty`方法 - 匹配空槽位

```cpp
bit_mask match_empty() const noexcept {
  return bit_mask{(ctrl_ & (~ctrl_ << 6U)) & MSBS};
}
```

* 通过位运算找出EMPTY(-128)控制字节

* 利用了EMPTY的特殊位模式(10000000)

#### 通过一个例子来理解`match_empty`

假设我们有以下控制字节组（小端序排列）：
[0x12, 0x34, EMPTY, 0x56, DELETED, EMPTY, 0x78, END]

EMPTY = 0x80

DELETED = 0xFE

END = 0xFF

对应的 64 位 ctrl_ 值为：0xFF8078FE56803412

```c++
=== match_empty() 示例 ===
原始控制字节 (0xff7880fe56803412): 0x12 0x34 0x80 0x56 0xfe 0x80 0x78 0xff 
~ctrl_ (0x00877f01a97fcbed): 0xed 0xcb 0x7f 0xa9 0x01 0x7f 0x87 0x00 
~ctrl_ << 6 (0x21dfc06a5ff2fb40): 0x40 0xfb 0xf2 0x5f 0x6a 0xc0 0xdf 0x21 
ctrl_ & (~ctrl_ << 6) (0x2158806a56803000): 0x00 0x30 0x80 0x56 0x6a 0x80 0x58 0x21 
最终结果 mask (0x0000800000800000): 0x00 0x00 0x80 0x00 0x00 0x80 0x00 0x00 

匹配的EMPTY位置: 2 5 
```
### `match_empty_or_deleted`方法

```cpp
bit_mask match_empty_or_deleted() const noexcept {
  return bit_mask{(ctrl_ & (~ctrl_ << 7U)) & MSBS};
}
```

* 类似`match_empty`但匹配EMPTY或DELETED状态

* 利用了这两种状态的高位都是1的特性

#### 通过一个例子来理解`match_empty_or_deleted`
假设我们有以下控制字节组（小端序排列）：
[0x12, 0x34, EMPTY, 0x56, DELETED, EMPTY, 0x78, END]

```c++
shift_pos: 7
=== match_empty() 示例 ===
原始控制字节 (0xff7880fe56803412): 0x12 0x34 0x80 0x56 0xfe 0x80 0x78 0xff 
~ctrl_ (0x00877f01a97fcbed): 0xed 0xcb 0x7f 0xa9 0x01 0x7f 0x87 0x00 
~ctrl_ << $shift_pos$ (0x43bf80d4bfe5f680): 0x80 0xf6 0xe5 0xbf 0xd4 0x80 0xbf 0x43 
ctrl_ & (~ctrl_ << $shift_pos$) (0x433880d416803400): 0x00 0x34 0x80 0x16 0xd4 0x80 0x38 0x43 
最终结果 mask (0x0000808000800000): 0x00 0x00 0x80 0x00 0x80 0x80 0x00 0x00 

匹配的EMPTY位置: 2 4 5
```

### `count_leading_empty_or_deleted`方法

```cpp
std::size_t count_leading_empty_or_deleted() const noexcept {
  return (trailing_zeros(((~ctrl_ & (ctrl_ >> 7U)) | GAPS) + 1U) + 7U) >> 3U;
}
```

* 计算前导的EMPTY或DELETED槽位数量

* 用于探测序列中快速跳过连续无效槽位

使用方法：
```cpp
void skip_empty_or_deleted() noexcept {
    while (is_empty_or_deleted(*ctrl_)) {
        auto const shift = group{ctrl_}.count_leading_empty_or_deleted();
        ctrl_ += shift;
        entry_ += shift;
    }
}
```

```c++
=== 示例1 ===
原始控制字节 (0xff788056fe803412): 0x12 0x34 0x80 0xfe 0x56 0x80 0x78 0xff 
ctrl_ >> 7 (0x01fef100adfd0068): 0x68 0x00 0xfd 0xad 0x00 0xf1 0xfe 0x01 
~ctrl_ (0x00877fa9017fcbed): 0xed 0xcb 0x7f 0x01 0xa9 0x7f 0x87 0x00 
~ctrl_ & (ctrl_ >> 7) (0x00867100017d0068): 0x68 0x00 0x7d 0x01 0x00 0x71 0x86 0x00 
| GAPS (0x00fefffefffffefe): 0xfe 0xfe 0xff 0xff 0xfe 0xff 0xfe 0x00 
+ 1 (0x00fefffefffffeff): 0xff 0xfe 0xff 0xff 0xfe 0xff 0xfe 0x00 
trailing_zeros: 0
前导EMPTY/DELETED数量: 0

=== 示例2 ===
原始控制字节 (0xffbc9a785680fe80): 0x80 0xfe 0x80 0x56 0x78 0x9a 0xbc 0xff 
ctrl_ >> 7 (0x01ff7934f0ad01fd): 0xfd 0x01 0xad 0xf0 0x34 0x79 0xff 0x01 
~ctrl_ (0x00436587a97f017f): 0x7f 0x01 0x7f 0xa9 0x87 0x65 0x43 0x00 
~ctrl_ & (ctrl_ >> 7) (0x00436104a02d017d): 0x7d 0x01 0x2d 0xa0 0x04 0x61 0x43 0x00 
| GAPS (0x00fffffefeffffff): 0xff 0xff 0xff 0xfe 0xfe 0xff 0xff 0x00 
+ 1 (0x00fffffeff000000): 0x00 0x00 0x00 0xff 0xfe 0xff 0xff 0x00 
trailing_zeros: 24
前导EMPTY/DELETED数量: 3
```
可以看到，为了采用位运算，加快计算速度，在控制字节中，EMPTY = 0x80，DELETED = 0xFE，END = 0xFF的数值都是经过特殊设计的。


### bit_mask结构解析

`bit_mask`是Swiss Table中用于高效处理位掩码的辅助结构，主要功能是迭代和操作64位掩码中的匹配位。

```cpp
group_t mask_;  // 64位掩码，每个字节代表一个控制字节的状态
static constexpr auto const SHIFT = 3U;  // 用于字节索引位移(8=2^3)
```
迭代器和掩码操作，实现了类似迭代器的接口，可以在范围for循环中使用：
```cpp
bit_mask& operator++() noexcept {
  mask_ &= (mask_ - 1U);  // 清除最低位的1
  return *this;
}

size_type operator*() const noexcept { 
  return trailing_zeros();  // 返回当前最低位1的字节位置
}

explicit operator bool() const noexcept { 
  return mask_ != 0U;  // 判断是否有匹配
}

bit_mask begin() const noexcept { return *this; }
bit_mask end() const noexcept { return bit_mask{0}; }
```

### 位操作

* `trailing_zeros`: 计算最低位1的位置(以字节为单位)

* `leading_zeros`: 计算最高位1的位置(以字节为单位)

* 都使用了位运算优化，通过位移转换位索引到字节索引

```cpp
size_type trailing_zeros() const noexcept {
  // ::cista::trailing_zeros(mask_) 返回的是最低位1的位置(以bit为单位)
  // 这里 >> SHIFT 是将bit索引转换为字节索引
  return ::cista::trailing_zeros(mask_) >> SHIFT;
}

size_type leading_zeros() const noexcept {
  constexpr int total_significant_bits = 8 << SHIFT;  // 64
  constexpr int extra_bits = sizeof(group_t) * 8 - total_significant_bits;
  return ::cista::leading_zeros(mask_ << extra_bits) >> SHIFT;
}

//         00000000 000000000 00000000 00000000
// bit索引： 1-8 9-16 17-24 25-32
// 字节索引： 1 2 3 4
```

```c++
template <typename T>
constexpr unsigned trailing_zeros(T t) noexcept {
    if constexpr (sizeof(T) == 8U) {
        return static_cast<unsigned>(__builtin_ctzll(t));
    } else if constexpr (sizeof(T) == 4U) {  // 32bit
        return static_cast<unsigned>(__builtin_ctz(t));
    }
}
```

这个函数作用是返回输入数二进制表示从最低位开始(右起)的连续的0的个数；如果传入0则行为未定义。通过__builtin_ctzll实现。

```c++
template <typename T>
constexpr unsigned leading_zeros(T t) noexcept {
    if constexpr (sizeof(T) == 8U) {
        return static_cast<unsigned>(__builtin_clzll(t));
    } else if constexpr (sizeof(T) == 4U) {  // 32bit
        return static_cast<unsigned>(__builtin_clz(t));
    }
}
```

这个函数作用是返回输入数二进制表示从最高位开始(左起)的连续的0的个数；如果传入0则行为未定义。通过__builtin_clzll实现。

## 总结

通过位运算并行处理，虽然未使用实际SIMD指令，但通过64位整数运算模拟了并行处理8个控制字节的效果。所有操作都通过**精心设计的位运算**完成，避免了分支和循环。**内存局部性**：控制字节连续存储，充分利用CPU缓存。**状态快速判断**：可以一次性判断整个group的状态(全空/部分匹配等)。通过这些设计，使得Swiss Table在查找、插入等操作上获得高吞吐量。