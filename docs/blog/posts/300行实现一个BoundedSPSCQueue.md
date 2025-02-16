---
title: 300行实现一个BoundedSPSCQueue
date: 2024-09-08
authors: [KenForever1]
categories: 
  - cpp
  - 队列
labels: []
pin: true
comments: true
---
我每天早上都会打开Github的Trendings板块，了解最新的动态，我相信这是个还不错的习惯。遇到感兴趣的项目我会打开看看源码，学习一下大佬们的写法和实现。

??? note "快速查看github项目源码"
    修改github.com/xxx/yyy链接为github1s.com/xxx/yyy，就可以在浏览器打开vscode类似界面阅读代码了。

今天分享的是[quill](https://github.com/odygrd/quill)，一个异步低延迟的高效日志库实现。
<!-- more -->
+ 分为Frontend hot线程（支持多线程）和Backend线程。hot线程将日志信息提交给Bounded/Unbounded SPSC Queue，由统一的Backend线程去处理。Backend pop SPSC队列,格式化日志消息保存到一个buffer中；为了保持日志顺序，等SPSC为空或者buffer最大size才输出一次日志。实现了类似spdlog库一样的可扩展的Sinks，比如输出到文件、stdout等。
+ 实现了Linux huge page大页支持。
+ 采用了编译时优化，使用了很多gnu attribute标记优化代码，比如__attribute__((hot))、__attribute__((cold))、likely和unlikely等。
+ 针对x86架构在SPSC中进行了刷新cache、预取等优化，比如_flush_cachelines、_mm_prefetch。

## 架构

![架构图](https://github.com/odygrd/quill/blob/master/docs%2Fdesign.jpg)

## BoundedSPSC队列实现

### 类以及成员变量声明

SPSC表示单生产者单消费者，实现上采用了mmap申请一个2 * capacity的内存区域，先简单看看Class声明：
```cpp
template <typename T>
class BoundedSPSCQueueImpl
{
public:
  using integer_type = T;

private:
  static constexpr integer_type CACHELINE_MASK{CACHE_LINE_SIZE - 1};

  // queue的内存容量，一般为2的n次方
  integer_type const _capacity;
  // 大小为_capacity - 1，用于函数实现中的一些计算
  integer_type const _mask;
  integer_type const _bytes_per_batch;
  // mmap映射的内存区域指针，writer_pos和reader_pos就是基于这个位置操作
  std::byte* const _storage{nullptr};
  // linux平台huge page支持，在mmap函数的时候指定flags MAP_HUGETLB开启
  bool const _huge_pages_enabled;
  // 记录write指针的位置（对齐cacheline避免cpu伪共享）
  alignas(CACHE_LINE_ALIGNED) std::atomic<integer_type> _atomic_writer_pos{0};
  alignas(CACHE_LINE_ALIGNED) integer_type _writer_pos{0};
  integer_type _reader_pos_cache{0}; // cache优化，writer判断空间是否充足时，减少对_atomic_reader_pos的load
  integer_type _last_flushed_writer_pos{0};
  // 记录read指针的位置
  alignas(CACHE_LINE_ALIGNED) std::atomic<integer_type> _atomic_reader_pos{0};
  alignas(CACHE_LINE_ALIGNED) integer_type _reader_pos{0};
  mutable integer_type _writer_pos_cache{0};  // 这个cache也是一个优化点，通过read/write加commit两步操作，在read判断empty的时候，减少对_atomic_writer_pos的load，先使用cache判断相等，然后再确认是否和_atomic_writer_pos相等
  integer_type _last_flushed_reader_pos{0};
}
```
**alignas(CACHE_LINE_ALIGNED)** 标记的作用是将读写位置标记进行cacheline对齐（一般是64字节），避免了**伪共享**问题。

### 内存区域申请

申请一个内存，

```cpp
QUILL_NODISCARD static void* _alloc_aligned(size_t size, size_t alignment, QUILL_MAYBE_UNUSED bool huges_pages_enabled)
  {
    // Calculate the total size including the metadata and alignment
    // metadata保存了total_size和offset
    // |  total_size | offs  | xxxx         | storage area |
    // |          元数据      |未对齐区域    | 对齐的存储区域 |
    // |                  分配的全部区域                     |

    constexpr size_t metadata_size{2u * sizeof(size_t)};
    size_t const total_size{size + metadata_size + alignment};

    // Allocate the memory，匿名 私有内存区域，私有用到了写时拷贝机制，修改的内容其它进程看不到。
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;

  // 宏定义开启huge page
  #if defined(__linux__)
    if (huges_pages_enabled)
    {
      flags |= MAP_HUGETLB;
    }
  #endif

    void* mem = ::mmap(nullptr, total_size, PROT_READ | PROT_WRITE, flags, -1, 0);

    if (mem == MAP_FAILED)
    {
      QUILL_THROW(QuillError{std::string{"mmap failed. errno: "} + std::to_string(errno) +
                             " error: " + strerror(errno)});
    }

    // Calculate the aligned address after the metadata
    std::byte* aligned_address = _align_pointer(static_cast<std::byte*>(mem) + metadata_size, alignment);

    // Calculate the offset from the original memory location
    auto const offset = static_cast<size_t>(aligned_address - static_cast<std::byte*>(mem));

    // Store the size and offset information in the metadata
    std::memcpy(aligned_address - sizeof(size_t), &total_size, sizeof(total_size));
    std::memcpy(aligned_address - (2u * sizeof(size_t)), &offset, sizeof(offset));

    return aligned_address;
  }
```

分配的内存区域如下所示，总分配的区域大小为2 * cap；元数据区域大小 2 * sizeof(size_t), 分别保存了total_size和offset偏移量；存储区域（storage指针指向的区域）按照alignment参数对齐，所以有一段未对齐区域空闲，offset = storage - start_pos。
```bash
|  total_size | offset  | xxxx         | storage area |
|          元数据        |未对齐区域    | 对齐的存储区域 |
|                     分配的全部区域                    |
```

??? note "宏定义的用法"
    QUILL_NODISCARD实际上是gnu attribute标注[[nodiscard]]，要求要函数调用者处理返回值。

_align_pointer也是常用的实现方法，比如一个数，要实现16对齐，按照位运算就是：(num + 16 - 1) & ~15 。
```bash
  QUILL_NODISCARD static std::byte* _align_pointer(void* pointer, size_t alignment) noexcept
  {
    assert(is_power_of_two(alignment) && "alignment must be a power of two");
    return reinterpret_cast<std::byte*>((reinterpret_cast<uintptr_t>(pointer) + (alignment - 1ul)) &
                                        ~(alignment - 1ul));
  }
```

??? note "判断一个数是不是2的n次方"
    num & (num - 1) == 0

看看构造函数中怎么分配内存的，_storage变量赋值调用了_alloc_aligned函数。分配的size是2 * cap, 对齐 2 * cacheline_size, 128字节。
```cpp
  QUILL_ATTRIBUTE_HOT explicit BoundedSPSCQueueImpl(integer_type capacity, bool huges_pages_enabled = false,
                                                    integer_type reader_store_percent = 5)
    : _capacity(next_power_of_two(capacity)),
      _mask(_capacity - 1),
      _bytes_per_batch(static_cast<integer_type>(_capacity * static_cast<double>(reader_store_percent) / 100.0)),
      _storage(static_cast<std::byte*>(_alloc_aligned(2ull * static_cast<uint64_t>(_capacity),
                                                      CACHE_LINE_ALIGNED, huges_pages_enabled))),
      _huge_pages_enabled(huges_pages_enabled)
  {
    std::memset(_storage, 0, 2ull * static_cast<uint64_t>(_capacity));

    _atomic_writer_pos.store(0);
    _atomic_reader_pos.store(0);

  }

  ~BoundedSPSCQueueImpl() { _free_aligned(_storage); }
```

析构函数释放内存区域，调用unmap函数, 根据元数据获取到偏移量，然后定位到mmap返回的指针。
```cpp
  void static _free_aligned(void* ptr) noexcept
  {
    // Retrieve the size and offset information from the metadata
    size_t offset;
    std::memcpy(&offset, static_cast<std::byte*>(ptr) - (2u * sizeof(size_t)), sizeof(offset));

    size_t total_size;
    std::memcpy(&total_size, static_cast<std::byte*>(ptr) - sizeof(size_t), sizeof(total_size));

    // Calculate the original memory block address
    void* mem = static_cast<std::byte*>(ptr) - offset;

    ::munmap(mem, total_size);
  }

```

### 队列的常用函数实现

#### empty

如果读写位置相等，就表示队列为空，这里使用了两次判断，先比较_writer_pos_cache == _reader_pos，如果相等，load atomic变量二次确认。
该empty函数只reader调用有效, 写者不用关心队列是否为空。
```cpp
  QUILL_NODISCARD QUILL_ATTRIBUTE_HOT bool empty() const noexcept
  {
    // reader会一直等writer commit提交了以后，也就是改变了_atomic_writer_pos，empty返回false，才能读取内容
    if (_writer_pos_cache == _reader_pos)
    {
      // if we think the queue is empty we also load the atomic variable to check further
      _writer_pos_cache = _atomic_writer_pos.load(std::memory_order_acquire);

      if (_writer_pos_cache == _reader_pos)
      {
        return true;
      }
    }

    return false;
  }
```

读者在读之前会调用prepare_read函数，实际上是调用empty函数判断是否可读。
```cpp
  QUILL_NODISCARD QUILL_ATTRIBUTE_HOT std::byte* prepare_read() noexcept
  {
    if (empty())
    {
      return nullptr;
    }

    return _storage + (_reader_pos & _mask);
  }
```

#### read操作
read操作分为两个函数, 每次读取调用finish_read修改读指针位置，commit_read修改_atomic_reader_pos，让writer线程可以看到修改。
```cpp
  // 修改_reader_pos指向内存区域的位置
  QUILL_ATTRIBUTE_HOT void finish_read(integer_type n) noexcept { _reader_pos += n; }

  QUILL_ATTRIBUTE_HOT void commit_read() noexcept
  {
    // 判断是否读取内容超过_bytes_per_batch，超过就修改_atomic_reader_pos
    if (static_cast<integer_type>(_reader_pos - _atomic_reader_pos.load(std::memory_order_relaxed)) >= _bytes_per_batch)
    {
      _atomic_reader_pos.store(_reader_pos, std::memory_order_release);

#if defined(QUILL_X86ARCH) // 针对x86架构flush_cacheline的优化
      _flush_cachelines(_last_flushed_reader_pos, _reader_pos);
#endif
    }
  }
```

如何使用的呢？这里通过多次读取修改read_pos，一次提交修改shared atomic_read_pos，增强缓存一致性。
```cpp
do {
  // 更新read_pos
  frontend_queue.finish_read(bytes_read);
}while(xxx);
// If we read something from the queue, we commit all the reads together at the end.
// This strategy enhances cache coherence performance by updating the shared atomic flag
// only once.
// 多次读取修改read_pos，一次提交修改shared atomic_read_pos，增强缓存一致性
frontend_queue.commit_read();
```


#### write操作

```cpp
QUILL_NODISCARD QUILL_ATTRIBUTE_HOT std::byte* prepare_write(integer_type n) noexcept
  {
    // 判断剩余内存容量是否小于n
    if ((_capacity - static_cast<integer_type>(_writer_pos - _reader_pos_cache)) < n)
    {
      // not enough space, we need to load reader and re-check
      // 获取_atomic_reader_pos，二次确认reader是否读了并且commit了
      _reader_pos_cache = _atomic_reader_pos.load(std::memory_order_acquire);

      if ((_capacity - static_cast<integer_type>(_writer_pos - _reader_pos_cache)) < n)
      {
        return nullptr;
      }
    }

    // 如果内存容量充裕，返回writer_pos对应的指针

    return _storage + (_writer_pos & _mask);
  }
```

read操作分为两个函数, 每次读取调用finish_write修改写指针位置。commit_write修改_atomic_write_pos，让reader线程可以看到修改, reader调用empty函数就会返回false，不为空。
```cpp

  QUILL_ATTRIBUTE_HOT void finish_write(integer_type n) noexcept { _writer_pos += n; }

  QUILL_ATTRIBUTE_HOT void commit_write() noexcept
  {
    // set the atomic flag so the reader can see write
    _atomic_writer_pos.store(_writer_pos, std::memory_order_release);

#if defined(QUILL_X86ARCH)
    // flush writen cache lines
    _flush_cachelines(_last_flushed_writer_pos, _writer_pos);

    // prefetch a future cache line
    _mm_prefetch(reinterpret_cast<char const*>(_storage + (_writer_pos & _mask) + (CACHE_LINE_SIZE * 10)),
                 _MM_HINT_T0);
#endif
  }
```


### flush 缓存优化

flush缓存优化只针对了x86 arch。

```bash
#if defined(QUILL_X86ARCH)
    // flush writen cache lines
    _flush_cachelines(_last_flushed_writer_pos, _writer_pos);

    // prefetch a future cache line
    // 预取(CACHE_LINE_SIZE * 10)的内存加入缓存，用于加速writer写操作
    _mm_prefetch(reinterpret_cast<char const*>(_storage + (_writer_pos & _mask) + (CACHE_LINE_SIZE * 10)),
                 _MM_HINT_T0);
#endif
```

_flush_cachelines函数实现如下，比如reader对某个区域进行了读取，可以理解为一个ringbuffer，这个区域临近的下次下下次就不会访问了；又或者writer对某个区域进行了写commit，那这个区域也可以flush缓存行刷回。（把指定缓存行（Cache Line）从所有级缓存中淘汰，若该缓存行中的数据被修改过，则将该数据写入主存）
```cpp
#if defined(QUILL_X86ARCH)
  QUILL_ATTRIBUTE_HOT void _flush_cachelines(integer_type& last, integer_type offset)
  {
    // 对last指针进行CACHELINE_size对齐
    integer_type last_diff = last - (last & CACHELINE_MASK);
    // 对offset指针进行CACHELINE_size对齐
    integer_type const cur_diff = offset - (offset & CACHELINE_MASK);

    // 调用_mm_clflushopt对[last, offset]缓存行刷回
    while (cur_diff > last_diff)
    {
      _mm_clflushopt(_storage + (last_diff & _mask));
      last_diff += CACHE_LINE_SIZE;
      last = last_diff;
    }
  }
#endif
```

阅读源码请前往，[BoundedSPSC队列实现](https://github1s.com/odygrd/quill/blob/master/include/quill/core/BoundedSPSCQueue.h)。