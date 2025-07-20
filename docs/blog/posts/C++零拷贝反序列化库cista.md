---
title: C++零拷贝反序列化库cista
date: 2025-06-28
authors: [KenForever1]
categories: 
  - C++
labels: []
comments: true
---
## C++零拷贝反序列化库cista

在阅读Candle（rust的机器学习库）时，看到了rust中利用mmap和Cow机制实现零拷贝反序列化加载模型的tensor。

其中为了解决mmap的数据生命周期为'a, 如果用Cow<'a>引用mmap返回的指针, 使用的生命周期如果超过'a则不能使用，不利于代码开发和引用。因此引入了yoke crate擦除生命周期。

借此，搜索了一下c++的零拷贝反序列化库，找到了[cista](https://github.com/felixguendling/cista)。

<!-- more -->

```c++
namespace data = cista::offset;
constexpr auto const MODE =  // opt. versioning + check sum
    cista::mode::WITH_VERSION | cista::mode::WITH_INTEGRITY;

struct pos { int x, y; };
using pos_map =  // Automatic deduction of hash & equality
    data::hash_map<data::vector<pos>,
                   data::hash_set<data::string>>;

{  // Serialize.
  auto positions =
      pos_map{{{{1, 2}, {3, 4}}, {"hello", "cista"}},
              {{{5, 6}, {7, 8}}, {"hello", "world"}}};
  cista::buf mmap{cista::mmap{"data"}};
  cista::serialize<MODE>(mmap, positions);
}

// Deserialize.
auto b = cista::mmap("data", cista::mmap::protection::READ);
auto positions = cista::deserialize<pos_map, MODE>(b);
```

它有两种模式：
```c++
namespace data = cista::raw;
和
namespace data = cista::offset;
```
在data命名空间下，实现了hashmap、hashset、string、vector等数据结构，支持判断相等、hash等。

+ 为什么offset方式相比raw方式速度更快？

+ hashmap如何实现的？

## offset和raw方式

Offset Based Data Structures

* \+ can be read without any deserialization step (i.e. reinterpret_cast<T> is sufficient).

* \+ suitable for shared memory applications

* \- slower at runtime (pointers need to be resolved using one more add)

Raw Data Structures

* \- deserialize step takes time (but still very fast also for GBs of data)

* \- the buffer containing the serialized data needs to be modified

* \+ fast runtime access (raw access)

我们简单过一下代码实现，有兴趣的朋友可以阅读源码，代码很清晰。

## 零拷贝反序列化

零拷贝主要使用了mmap方式，将文件映射到进程地址空间，直接通过指针访问文件内容。

```c++
template <mode Mode = kDefaultMode, typename T>
void write(std::filesystem::path const& p, T const& w) {
  auto mmap =
      cista::mmap{p.generic_string().c_str(), cista::mmap::protection::WRITE};
  auto writer = cista::buf<cista::mmap>(std::move(mmap));
  cista::serialize<Mode>(writer, w);
}

template <mode Mode = kDefaultMode, typename T>
void write(std::filesystem::path const& p, wrapped<T> const& w) {
  write<Mode>(p, *w);
}

template <typename T, mode Mode = kDefaultMode>
cista::wrapped<T> read(std::filesystem::path const& p) {
  auto b = cista::file{p.generic_string().c_str(), "r"}.content();
  auto const ptr = cista::deserialize<T, Mode>(b);
  auto mem = cista::memory_holder{std::move(b)};
  return cista::wrapped{std::move(mem), ptr};
}

template <typename T, mode Mode = kDefaultMode>
cista::wrapped<T> read_mmap(std::filesystem::path const& p) {
  auto mmap =
      cista::mmap{p.generic_string().c_str(), cista::mmap::protection::READ};
  auto const ptr = cista::deserialize<T, Mode>(mmap);
  auto mem = cista::memory_holder{buf{std::move(mmap)}};
  return cista::wrapped{std::move(mem), ptr};
}
```


## hashmap实现

### swiss table实现

参考了[abseil的swiss tables](https://abseil.io/blog/20180927-swisstables)实现，在原先实现的上删除了多余的功能。

swiss table从用法上分为两类：

+ absl::flat_hash_map and absl::flat_hash_set

flat方式将value_type存储在容器的主数组中，以避免内存间接寻址。由于它们在重新哈希时会移动数据，因此元素无法保持指针稳定性。

![](https://abseil.io/img/flat_hash_map.svg)

+ absl::node_hash_map and absl::node_hash_set

追求指针稳定性，或者你的值很大，就采用node方式。

![](https://abseil.io/img/node_hash_map.svg)

[blog/20180927-swisstables](https://abseil.io/blog/20180927-swisstables)中介绍，更推荐使用 absl::flat_hash_map\<K, std::unique_ptr\<V\>\>方式替代 absl::node_hash_map\<K, V\>。

### hash_storage实现

hashmap和hashset的核心实现是**hash_storage** struct, hash_storage采用**swiss_table**实现。在后面的文章的介绍。

```c++
namespace raw {
template <typename Key, typename Value, typename Hash = hashing<Key>,
          typename Eq = equal_to<Key>>
using hash_map =
    hash_storage<pair<Key, Value>, ptr, get_first, get_second, Hash, Eq>;
}  // namespace raw

namespace offset {
template <typename Key, typename Value, typename Hash = hashing<Key>,
          typename Eq = equal_to<Key>>
using hash_map =
    hash_storage<pair<Key, Value>, ptr, get_first, get_second, Hash, Eq>;
}  // namespace offset

namespace offset {
template <typename T, typename Hash = hashing<T>, typename Eq = equal_to<T>>
using hash_set = hash_storage<T, ptr, identity, identity, Hash, Eq>;
}  // namespace offset

```

## string的实现

根据string的大小，分为两种实现，小的用stack，大的用heap, short_length_limit是16字节。
```c++
struct generate_string{
  struct heap {
    bool is_short_{false};
    bool self_allocated_{false};
    std::uint16_t __fill__{0};
    std::uint32_t size_{0};
    Ptr ptr_{nullptr};
  };

  struct stack {
    union {
      bool is_short_{true};
      CharT __fill__;
    };
    CharT s_[short_length_limit]{0};
  };

  union {
    heap h_{};
    stack s_;
  };
}

```

分为两种情况，own和non_own。也就是管理内存还是不管理内存。

```c++
  static constexpr struct owning_t {
  } owning{};
  static constexpr struct non_owning_t {
  } non_owning{};
```

如果要从non_owning转换成owning，则需要调用std::memcpy。

```c++
static constexpr msize_t short_length_limit = 15U / sizeof(CharT);

void set_owning(CharT const* str, msize_t const len) {
    reset();
    if (str == nullptr || len == 0U) {
    return;
    }
    s_.is_short_ = (len <= short_length_limit);
    if (s_.is_short_) {
    std::memcpy(s_.s_, str, len * sizeof(CharT));
    for (auto i = len; i < short_length_limit; ++i) {
        s_.s_[i] = 0;
    }
    } else {
    h_.ptr_ = static_cast<CharT*>(std::malloc(len * sizeof(CharT)));
    if (h_.ptr_ == nullptr) {
        throw_exception(std::bad_alloc{});
    }
    h_.size_ = len;
    h_.self_allocated_ = true;
    std::memcpy(data(), str, len * sizeof(CharT));
    }
}
```

## 其它序列化库

+ Protocol Buffers

+ Cap’n Proto

+ [Flatbuffers](https://flatbuffers.dev/languages/rust/)

+ cereal

+ Boost Serialization

+ MessagePack