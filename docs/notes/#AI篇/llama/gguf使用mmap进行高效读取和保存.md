---
comments: true
---
## llama中的GGUF格式如何加载模型文件？
gguf是gg大佬发明的保存llm模型的格式。
保存了header、k-v、tensor，支持多种模型，保存GPT、Phi3、transformer等等，支持扩展。
在gguf经过多个版本进化而来，ggml、GGJT。
从GGJT开始支持mmap。
我们接下来要聊一聊为什么要支持mmap方式，有什么作用？
gguf在保存tensor数据时进行了align对齐操作，使用mmap就可以高效快速的加载数据。

下面介绍的源码都出自llama-cpp项目中llama.h和llama.cpp。
## 一起窥探源码
关键是align对齐，然后方便使用mmap高效load读取数据。
在llama_model_params中定义了use_mmap，可以通过命令行等控制，是否要使用mmap。
```c++
// llama.h
struct llama_model_params {
    bool use_mmap          = true;  // use mmap for faster loads
    bool use_mlock         = false; // use mlock to keep model in memory
}
```
在llama的实现中，mmap在win平台和linux平台实现api是不同的，通过宏定义使用不同的代码段，下面介绍linux平台实现，win可以阅读源码。

### llama获取tensor数据
在了解mmap之前，先从llama的使用上逐步进行窥探，通过load_data_for函数对ggml_tensor权重数据进行的读取。可以看到分为两种，从mmap读取，从file读取。
```c++
// llama.cpp
void load_data_for(struct ggml_tensor * cur) const {
    const auto & w = require_weight(ggml_get_name(cur));

    // 如果使用mmap，直接从mmap获取数据
    if (use_mmap) {
        const auto & mapping = mappings.at(w.idx);
        if (cur->data == nullptr) {
            // 将指针赋值给cur->data
            cur->data = (uint8_t *)mapping->addr + w.offs;
        } else {
            // 拷贝数据到cur->data
            memcpy(cur->data, (uint8_t *)mapping->addr + w.offs, ggml_nbytes(cur));
        }
    } else {
        // 通过文件描述符读取数据
        GGML_ASSERT(cur->data != nullptr);
        GGML_ASSERT(w.idx < files.size());
        const auto & file = files.at(w.idx);
        file->seek(w.offs, SEEK_SET);
        file->read_raw(cur->data, ggml_nbytes(cur));
    }

    if (check_tensors && !ggml_validate_row_data(cur->type, cur->data, ggml_nbytes(cur))) {
        throw std::runtime_error(format("tensor '%s' has invalid data", ggml_get_name(cur)));
    }
}
```

两种数据结构**llama_mmap**和**llama_file**，分别定义了mmap方式和file读取方式。

### llama_mmap类的实现
先看看llama_mmap, 在llama_mmap的构造函数中，
+ 1. 先获取file的fd文件描述符
+ 2. 调用mmap函数，获取到映射的addr，一个llama_mmap就保存了映射文件后的地址和文件的size
+ 3. 将文件size保存到mapped_fragments中
可以看到除了上面的逻辑，还有一些调用posix_fadvise函数的优化，比如建议内核顺序读取文件、预取等。
```c++
struct llama_mmap {
    void * addr;
    size_t size;

    // list of mapped fragments (first_offset, last_offset)
    std::vector<std::pair<size_t, size_t>> mapped_fragments;

    llama_mmap(struct llama_file * file, size_t prefetch = (size_t) -1 /* -1 = max value */, bool numa = false) {
        size = file->size;
        int fd = fileno(file->fp);
        int flags = MAP_SHARED;
        // prefetch/readahead impairs performance on NUMA systems
        if (numa)  { prefetch = 0; }
    #ifdef __linux__
        // advise the kernel to read the file sequentially (increases readahead)
        if (posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL)) {
            LLAMA_LOG_WARN("warning: posix_fadvise(.., POSIX_FADV_SEQUENTIAL) failed: %s\n",
                    strerror(errno));
        }
        if (prefetch) { flags |= MAP_POPULATE; }
    #endif
        addr = mmap(NULL, file->size, PROT_READ, flags, fd, 0);
        if (addr == MAP_FAILED) { // NOLINT
            throw std::runtime_error(format("mmap failed: %s", strerror(errno)));
        }

        if (prefetch > 0) {
            // advise the kernel to preload the mapped memory
            if (posix_madvise(addr, std::min(file->size, prefetch), POSIX_MADV_WILLNEED)) {
                LLAMA_LOG_WARN("warning: posix_madvise(.., POSIX_MADV_WILLNEED) failed: %s\n",
                        strerror(errno));
            }
        }
        if (numa) {
            // advise the kernel not to use readahead
            // (because the next page might not belong on the same node)
            if (posix_madvise(addr, file->size, POSIX_MADV_RANDOM)) {
                LLAMA_LOG_WARN("warning: posix_madvise(.., POSIX_MADV_RANDOM) failed: %s\n",
                        strerror(errno));
            }
        }

        // initialize list of mapped_fragments
        mapped_fragments.emplace_back(0, file->size);
    }
}
using llama_mmaps = std::vector<std::unique_ptr<llama_mmap>>;
```

### llama_file类的实现

llama_file类使用fopen打开了文件，封装了fread、fwrite操作，保存了文件流指针和size。

```c++
struct llama_file {
    // use FILE * so we don't have to re-open the file to mmap
    FILE * fp; // 文件流指针
    size_t size; // 文件size

    llama_file(const char * fname, const char * mode) {
        // ggml_fopen封装了win32和linux api的区别，在linux实现这里直接是fopen(fname, mode)
        fp = ggml_fopen(fname, mode);
        if (fp == NULL) {
            throw std::runtime_error(format("failed to open %s: %s", fname, strerror(errno)));
        }
        seek(0, SEEK_END);
        size = tell();
        seek(0, SEEK_SET);
    }
}
```
> fopen和open最主要的区别是fopen在用户态下就有了缓存，在进行read和write的时候减少了用户态和内核态的切换，而open则每次都需要进行内核态和用户态的切换；表现为，如果顺序访问文件，fopen系列的函数要比直接调用open系列快；如果随机访问文件open要比fopen快。fopen返回文件流而不是linux下文件句柄。

所以，llama如果不使用mmap的方式（use_mmap = false时），**采用file读取数据在linux下是使用fread、fwrite api实现的**。

### llama初始化mmap

有了上面的数据结构介绍，我们再来看看mmaping初始化操作就很清晰了。将模型文件进行mmap，保存mmap以后的地址。在上面提到的load_data_for()函数中使用，llama_mmap保存的地址读取数据。

init_mappings函数和load_data_for函数一样，都在llama_model_loader类中定义：
```c++
struct llama_model_loader {
    bool use_mmap = false;

    llama_files files;
    llama_mmaps mappings;
}
```

```c++
// llama.cpp
void init_mappings(bool prefetch = true, llama_mlocks * mlock_mmaps = nullptr) {
    if (use_mmap) {
        mappings.reserve(files.size());
        mmaps_used.reserve(files.size());
        for (const auto & file : files) {
            std::unique_ptr<llama_mmap> mapping(new llama_mmap(file.get(), prefetch ? -1 : 0, ggml_is_numa()));
            mmaps_used.emplace_back(mapping->size, 0);
            if (mlock_mmaps) {
                std::unique_ptr<llama_mlock> mlock_mmap(new llama_mlock());
                mlock_mmap->init(mapping->addr);
                mlock_mmaps->emplace_back(std::move(mlock_mmap));
            }
            mappings.emplace_back(std::move(mapping));
        }
    }

    // compute the total size of all tensors for progress reporting
    for (auto & w : weights) {
        size_data += ggml_nbytes(w.tensor);
    }
}
```

## 再来聊一聊mlock
书接上文，在llama_model_params中定义了use_mmap，可以通过命令行等控制，是否要使用mmap。
```c++
// llama.h
struct llama_model_params {
    bool use_mmap          = true;  // use mmap for faster loads
    bool use_mlock         = false; // use mlock to keep model in memory
}
```
在前面的llama_model_params参数中除了提到了use_mmap以外，还有一个参数use_mlock。它的意思是将模型的内存锁住，避免回收。也就是将模型文件中保存的tensors的weight留在内存中。

### llama_mlock类如何定义
在llama中定义为llama_mlock结构体，
```c++
// Represents some region of memory being locked using mlock or VirtualLock;
// will automatically unlock on destruction.
struct llama_mlock {
    void * addr = NULL;
    size_t size = 0;

    ~llama_mlock() {
        if (size) {
            // 调用了munlock函数
            raw_unlock(addr, size);
        }
    }

    void init(void * ptr) {
        // NOLINT注释，如果编码者确认没问题，是让一些静态代码分析工具不报警
        GGML_ASSERT(addr == NULL && size == 0); // NOLINT
        addr = ptr;
    }

    // 在llama_load_all 函数中调用
    void grow_to(size_t target_size) {
        GGML_ASSERT(addr);
        if (failed_already) {
            return;
        }
        // 获取pagesize
        size_t granularity = lock_granularity();
        // 将target_size按照page_size对齐，这是一种常用的写法，比如将数字7按照8对齐，对齐结果就是8
        target_size = (target_size + granularity - 1) & ~(granularity - 1);
        if (target_size > size) {
            // 调用mlock
            if (raw_lock((uint8_t *) addr + size, target_size - size)) {
                size = target_size;
            } else {
                failed_already = true;
            }
        }
    }

    bool raw_lock(const void * addr, size_t size) const {
        if (!mlock(addr, size)) {
            return true;
        }
        ...
        // 如果内存不足，通过ulimit -l进行查看，通过ulimit设置更大的数值
        return false;
    }
}
```

> 一般用户空间关联的物理页面是按需通过缺页异常的方式分配和调页，当系统物理内存不足时页面回收算法会回收一些最近很少使用的页面，但是有时候我们需要锁住一些物理页面防止其被回收（如时间有严格要求的应用），Linux中提供了mlock相关的系统调用供用户空间使用来锁住部分或全部的地址空间关联的物理页面。

### grow_to函数的调用之load_all_data
llama提供了grow_to函数，对target_size进行pagesize对齐，然后mlock住这块内存。那么它是在哪儿使用的呢？
这就和上面的use_map联系起来了，在load_all_data函数实现中有如下代码，在使用use_mmap的时候，调用了grow_to函数。

```c++
struct llama_model_loader {
    bool use_mmap = false;

    llama_files files;
    llama_mmaps mappings;

    bool load_all_data(
            struct ggml_context * ctx,
            llama_buf_map & bufs_mmap,
            llama_mlocks * lmlocks,
            llama_progress_callback progress_callback,
            void * progress_callback_user_data);

    void init_mappings(bool prefetch = true, llama_mlocks * mlock_mmaps = nullptr);
    
    void load_data_for(struct ggml_tensor * cur);
    
}
```

```c++
bool load_all_data(
            struct ggml_context * ctx,
            llama_buf_map & bufs_mmap,
            llama_mlocks * lmlocks,
            llama_progress_callback progress_callback,
            void * progress_callback_user_data) {
    // 循环处理每一个tensor
    for (struct ggml_tensor * cur = ggml_get_first_tensor(ctx); cur != NULL; cur = ggml_get_next_tensor(ctx, cur)) {
        const auto * weight = get_weight(ggml_get_name(cur));
            ......
        if (use_mmap) {
            // 获取每个tensor 权重weight的mmaping
            const auto & mapping = mappings.at(weight->idx);
            ggml_backend_buffer_t buf_mmap = nullptr;
            if (bufs_mmap.count(weight->idx)) {
                buf_mmap = bufs_mmap.at(weight->idx);
            }
            // 根据offset定位到文件中权重保存的位置，获取mmap data指针
            uint8_t * data = (uint8_t *) mapping->addr + weight->offs;

            if (check_tensors) {
                validation_result.emplace_back(std::async(std::launch::async, [cur, data, n_size] {
                    return std::make_pair(cur, ggml_validate_row_data(cur->type, data, n_size));
                }));
            }

            GGML_ASSERT(buf_mmap || cur->data); // either we have a buffer to allocate the tensor in, or it is already allocated
            if (buf_mmap && cur->data == nullptr) {
                ggml_backend_tensor_alloc(buf_mmap, cur, data);
                // 锁住保存weight的内存
                if (lmlocks) {
                    const auto & lmlock = lmlocks->at(weight->idx);
                    lmlock->grow_to(weight->offs + n_size);
                }

                auto & mmap_used = mmaps_used[weight->idx];
                mmap_used.first  = std::min(mmap_used.first,  weight->offs);
                mmap_used.second = std::max(mmap_used.second, weight->offs + n_size);
            } else {
                ggml_backend_tensor_set(cur, data, 0, n_size);
            }
        }
    ....
} 
```


## 参考
https://github.com/ggerganov/llama.cpp
https://github.com/ggerganov/ggml/discussions/492
https://github.com/zylon-ai/private-gpt/issues/15


## 下一讲
如何align对齐