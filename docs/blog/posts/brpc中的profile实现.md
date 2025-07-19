---
title: brpc中的profile实现
date: 2025-06-02
authors: [KenForever1]
categories: 
  - cpp
labels: [cpp]
pin: true
comments: true
---

<!-- [TOC] -->

## brpc的cpu profile

使用了gperftools库。

<!-- more -->

[butil/gperftools_profiler.h](https://sourcegraph.com/github.com/apache/brpc@66e9635e915d120d1b73b4bb4523a4f5c9cdc084/-/blob/src/butil/gperftools_profiler.h?L60:19-60:32)

```c++
/* All this code should be usable from within C apps. */
#ifdef __cplusplus
extern "C" {
#endif

/* Start profiling and write profile info into fname, discarding any
 * existing profiling data in that file.
 *
 * This is equivalent to calling ProfilerStartWithOptions(fname, NULL).
 */
BRPC_DLL_DECL int ProfilerStart(const char* fname);

    ......
}

```
实现在[gperftools](https://github.com/gperftools/gperftools/blob/master/src/profiler.cc)中：

```c++
extern "C" PERFTOOLS_DLL_DECL int ProfilerStart(const char* fname) {
  return CpuProfiler::instance_.Start(fname, NULL);
}

```

## brpc的heap profile

https://brpc.apache.org/docs/builtin-services/heap_profiler/

在service中，[heap profile](https://sourcegraph.com/github.com/apache/brpc@66e9635e915d120d1b73b4bb4523a4f5c9cdc084/-/blob/src/brpc/builtin/pprof_service.cpp?L219:9-219:25)的实现方式有两种：
```c++
void PProfService::heap(
    ::google::protobuf::RpcController* controller_base,
    const ::brpc::ProfileRequest* /*request*/,
    ::brpc::ProfileResponse* /*response*/,
    ::google::protobuf::Closure* done) {
    ClosureGuard done_guard(done);
    Controller* cntl = static_cast<Controller*>(controller_base);

    if (HasJemalloc()) {
        JeControlProfile(cntl);
        return;
    }

    MallocExtension* malloc_ext = MallocExtension::instance();
    ...
    std::string obj;
    malloc_ext->GetHeapSample(&obj);
    cntl->http_response().set_content_type("text/plain");
    cntl->response_attachment().append(obj);    
}

```

方式一：JeControlProfile(cntl);
方式二：malloc_ext->GetHeapSample(&obj);

### jemalloc profile

jemalloc profile原理是使用了jeMalloc的mallctl接口和malloc_stats_print接口获取内存分配信息。

```c++
extern "C" {
// weak symbol: resolved at runtime by the linker if we are using jemalloc, nullptr otherwise
int BAIDU_WEAK mallctl(const char*, void*, size_t*, void*, size_t);
void BAIDU_WEAK malloc_stats_print(void (*write_cb)(void *, const char *), void *cbopaque, const char *opts);
}
```

[mallocctl](http://jemalloc.net/jemalloc.3.html)的使用可以参考例子：
```c++
// https://github.com/jemalloc/jemalloc/wiki/Use-Case:-Introspection-Via-mallctl*()
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <jemalloc/jemalloc.h>

void
do_something(size_t i)
{

        // Leak some memory.
        malloc(i * 100);
}

int
main(int argc, char **argv)
{
        size_t i, sz;

        for (i = 0; i < 100; i++) {
                do_something(i);

                // Update the statistics cached by mallctl.
                uint64_t epoch = 1;
                sz = sizeof(epoch);
                mallctl("epoch", &epoch, &sz, &epoch, sz);

                // Get basic allocation statistics.  Take care to check for
                // errors, since --enable-stats must have been specified at
                // build time for these statistics to be available.
                size_t sz, allocated, active, metadata, resident, mapped;
                sz = sizeof(size_t);
                if (mallctl("stats.allocated", &allocated, &sz, NULL, 0) == 0
                    && mallctl("stats.active", &active, &sz, NULL, 0) == 0
                    && mallctl("stats.metadata", &metadata, &sz, NULL, 0) == 0
                    && mallctl("stats.resident", &resident, &sz, NULL, 0) == 0
                    && mallctl("stats.mapped", &mapped, &sz, NULL, 0) == 0) {
                        fprintf(stderr,
                            "Current allocated/active/metadata/resident/mapped: %zu/%zu/%zu/%zu/%zu\n",
                            allocated, active, metadata, resident, mapped);
                }
        }

        return (0);
}
```
> The mallctl() function provides a general interface for introspecting the memory allocator, as well as setting modifiable parameters and triggering actions. The period-separated name argument specifies a location in a tree-structured namespace; see the MALLCTL NAMESPACE section for documentation on the tree contents. To read a value, pass a pointer via oldp to adequate space to contain the value, and a pointer to its length via oldlenp; otherwise pass NULL and NULL. Similarly, to write a value, pass a pointer to the value via newp, and its length via newlen; otherwise pass NULL and 0.

[源码实现brpc/details/jemalloc_profiler.h](https://sourcegraph.com/github.com/apache/brpc@66e9635e915d120d1b73b4bb4523a4f5c9cdc084/-/blob/src/brpc/details/jemalloc_profiler.h)
```c++

```
### tc_malloc profile

通过MallocExtension抽象类接口，定义了获取内存信息的虚函数。在brpc的实现中采用了dlsym打开动态库中的实现。

```c++
static void InitGetInstanceFn() {
    g_get_instance_fn = (GetInstanceFn)dlsym(
        RTLD_NEXT, "_ZN15MallocExtension8instanceEv");
}

MallocExtension* BAIDU_WEAK MallocExtension::instance() {
    // On fedora 26, this weak function is NOT overriden by the one in tcmalloc
    // which is dynamically linked.The same issue can't be re-produced in
    // Ubuntu and the exact cause is unknown yet. Using dlsym to get the
    // function works around the issue right now. Note that we can't use dlsym
    // to fully replace the weak-function mechanism since our code are generally
    // not compiled with -rdynamic which writes symbols to the table that
    // dlsym reads.
    pthread_once(&g_get_instance_fn_once, InitGetInstanceFn);
    if (g_get_instance_fn) {
        return g_get_instance_fn();
    }
    return NULL;
}
```

通过instance的注释可以看出为什么要通过dlsym方式。
> 在Fedora 26系统中，该弱函数不会被动态链接的tcmalloc中的同名函数覆盖，而在Ubuntu系统中不会出现这个问题，目前确切原因尚不明确。当下通过使用dlsym函数来获取所需函数，以此绕过这个问题。但要注意，不能完全用dlsym替代弱函数机制，因为代码通常没有使用 -rdynamic进行编译， -rdynamic会将符号写入dlsym读取的表中。例如，在实际开发中，当在Fedora 26系统下运行相关代码时，发现弱函数没有如预期被tcmalloc中的函数覆盖，进而导致功能异常，使用dlsym获取函数后，功能得以正常实现，但因为代码编译时未使用 -rdynamic，所以不能完全依赖dlsym来替代弱函数机制。 

[源码实现brpc/details/tcmalloc_extension.h](https://sourcegraph.com/github.com/apache/brpc@66e9635e915d120d1b73b4bb4523a4f5c9cdc084/-/blob/src/brpc/details/tcmalloc_extension.h)

来源于：[gperftools/malloc_extension.h](https://github.com/couchbase/gperftools/blob/master/src/gperftools/malloc_extension.h)