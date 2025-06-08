
---
title: 函数Hook（LD_PRELOAD）、审计流劫持（LD_AUDIT）及函数插桩
date: 2025-06-08
authors: [KenForever1]
categories: 
  - C++
labels: []
comments: true
---

Linux从符号劫持到运行时追踪：函数Hook（LD_PRELOAD）、审计流劫持（LD_AUDIT）及函数插桩。
<!-- more -->
## LDPRLOAD方式进行Hook

### LDPRLOAD方式进行Hook
LD_PRELOAD允许在程序运行时优先加载指定的动态链接库(.so文件)，覆盖默认的库函数实现。动态链接器会优先检查LD_PRELOAD指定的库，若其中存在与程序调用的同名函数，则使用该库中的实现而非系统默认版本。函数劫持与Hook，比如替换malloc的实现，替换so库中的函数，都可以通过LD_PRELOAD实现。

举个相关的使用例子：
+ **jemalloc的使用**

在[jemalloc的使用中](https://github.com/jemalloc/jemalloc/wiki/Getting-Started), 有几种方法可以将jemalloc集成到应用程序中。

最简单的就是，使用LD_PRELOAD环境变量在运行时将jemalloc注入应用程序。请注意，只有当您的应用程序没有静态链接malloc实现时，此方法才有效。

```shell
LD_PRELOAD=`jemalloc-config --libdir`/libjemalloc.so.`jemalloc-config --revision` app
```
除了这种方式，就是编译时动态链接和静态链接jemalloc到你的应用程序。
在开发中，可以利用这种方式查找内存崩溃的Bug、内存分配情况分析、Heap Profiler、解决内存泄露等。


### RTLD_NEXT方式进行Hook

除了LD_PRELOAD方式，还可以通过‌**dlsym RTLD_NEXT‌**进行Hook。

如果dlsym或dlvsym函数的第一个参数的值被设置为RTLD_NEXT，那么该函数返回下一个共享对象中名为NAME的符号的运行时地址，通常用于在Hook代码中调用原始函数。通过这种方式，我们可以实现一个自定义malloc函数，通过RTLD_NEXT拿到原始malloc的指针，从而可以在自定义malloc中调用原始malloc函数。达到Hook的目的。

运行时符号解析函数，用于在已加载的库链中查找下一个符合条件的符号实现。

举个相关的使用例子：
+ **协程网络库中hook**

[基于协程和事件循环的c++网络库](https://github.com/gatsbyd/melon/blob/master/src/Hook.cpp)中，通过dlsym RTLD_NEXT实现hook。包括了hook read、recv、send、sleep等函数。

```c++
#define DLSYM(name) \
		name ## _f = (name ## _t)::dlsym(RTLD_NEXT, #name);
```

```c++
unsigned int sleep(unsigned int seconds) {
	melon::Processer* processer = melon::Processer::GetProcesserOfThisThread();
	if (!melon::isHookEnabled()) {
        // 不hook时直接调用系统函数，sleep_f = dlsym(RTLD_NEXT, "sleep");
		return sleep_f(seconds);
	}

    // hook时，将当前协程挂起，等待seconds秒后继续执行
	melon::Scheduler* scheduler = processer->getScheduler();
	assert(scheduler != nullptr);
	scheduler->runAt(melon::Timestamp::now() + seconds * melon::Timestamp::kMicrosecondsPerSecond, melon::Coroutine::GetCurrentCoroutine());
	melon::Coroutine::SwapOut();
	return 0;
}
```

## LD_AUDIT链接器监听机制

通过前面的介绍，你对LD_LIBRARY、LD_PRELOAD肯定很熟悉了。LD_AUDIT是Linux系统中glibc动态链接器(ld.so)的另一个环境变量，用于指定审计库(audit library)的路径，主要用于监控和拦截动态链接库的加载过程

通过LD_AUDIT链接器监听机制，我们可以操纵glibc的动态链接过程，比如可以拦截动态链接库的加载过程，或者拦截动态链接库的符号解析过程。在xz-sshd漏洞中，攻击者通过LD_AUDIT劫持RSA解密函数调用链，实现权限提升。

看一个简单的例子：

```cpp
// audit_example.c
#define _GNU_SOURCE
#include <link.h>
#include <stdio.h>

// 必须实现的版本检查函数
unsigned int la_version(unsigned int version) {
    printf("审计库版本: %u (支持最高版本%u)\n", version, LAV_CURRENT);
    return LAV_CURRENT; // 返回支持的版本号
}

// 库加载时触发的回调
unsigned int la_objopen(struct link_map *map, Lmid_t lmid, uintptr_t *cookie) {
    printf("检测到库加载: %s (ID: %p)\n", map->l_name, (void*)*cookie);
    return LA_FLG_BINDTO | LA_FLG_BINDFROM; // 允许符号绑定追踪
}

// 符号绑定前触发的回调
uintptr_t la_symbind64(Elf64_Sym *sym, unsigned int ndx,
                      uintptr_t *refcook, uintptr_t *defcook,
                      unsigned int *flags, const char *symname) {
    printf("符号绑定: %s (地址: %#lx)\n", symname, sym->st_value);
    return sym->st_value; // 返回原始地址（可修改）
}
```
```cpp
// test_program.c
#include <stdio.h>
int main() {
    printf("hello\n");
    return 0;
}
// 编译审计库：gcc -shared -fPIC audit_example.c -o libaudit.so -ldl
// 编译测试程序：gcc test_program.c -o test -ldl
// 运行测试：LD_AUDIT=./libaudit.so ./test
```
打印输出：
```shell
审计库版本: 2 (支持最高版本2)
检测到库加载:  (ID: 0xffff87a2c370)
检测到库加载: /lib/ld-linux-aarch64.so.1 (ID: 0xffff87a2bb88)
检测到库加载: linux-vdso.so.1 (ID: 0xffff87a2c950)
检测到库加载: /lib/aarch64-linux-gnu/libc.so.6 (ID: 0xffff87a1d880)
符号绑定: __libc_start_main (地址: 0xffff87597434)
符号绑定: __cxa_finalize (地址: 0xffff875ad220)
符号绑定: abort (地址: 0xffff8759704c)
符号绑定: puts (地址: 0xffff875dae70)
符号绑定: calloc (地址: 0xffff875fe460)
符号绑定: free (地址: 0xffff875fdbc4)
符号绑定: malloc (地址: 0xffff875fd630)
符号绑定: realloc (地址: 0xffff875fde20)
符号绑定: _dl_catch_exception (地址: 0xffff8769d290)
符号绑定: _dl_signal_exception (地址: 0xffff8769d1e4)
符号绑定: __tls_get_addr (地址: 0xffff87a00cd0)
符号绑定: _dl_signal_error (地址: 0xffff8769d234)
符号绑定: _dl_catch_error (地址: 0xffff8769d390)
符号绑定: __tunable_get_val (地址: 0xffff87a02d40)
符号绑定: __getauxval (地址: 0xffff87655560)
符号绑定: _dl_audit_preinit (地址: 0xffff87a03774)
符号绑定: malloc (地址: 0xffff875fd630)
测试程序运行中...
```
不同于LD_PRELOAD（强制预加载库），LD_AUDIT**更侧重链接过程的‌事件监控‌**而非直接替换函数。当然，也是可以劫持替换函数实现的。

应用例子，在[xz-sshd漏洞可能的原理解读——链接器监听机制](https://zhuanlan.zhihu.com/p/689983608)中，通过la_symbind64函数劫持RSA_public_decrypt， 替换为自己实现的hijack_RSA_public_decrypt函数。
```cpp
// Called when a symbol is bound
uintptr_t la_symbind64(Elf64_Sym *sym, unsigned int ndx, uintptr_t *refcook,
                     uintptr_t *defcook, unsigned int *flags, const char *symname) {
    printf("Symbol bound: %s\n", symname);
    // Perform any custom actions here
    if (strcmp(symname, "RSA_public_decrypt") == 0) {
        return (uintptr_t)hijack_RSA_public_decrypt;
    }
    return sym->st_value; // Return the symbol's actual address
}
```


## GCC函数插桩功能（-finstrument-functions）

GCC的-finstrument-functions是一个强大的编译选项，用于在函数入口和出口自动插入钩子函数，主要用于性能分析和调用追踪。

编译时添加该选项后，GCC会在每个函数的开始插入__cyg_profile_func_enter，在函数返回前插入__cyg_profile_func_exit。这两个钩子函数的参数包含当前函数地址和调用者地址。通过这个功能，我们可以统计函数耗时，定位执行异常的函数，分析性能瓶颈等，当然也可以用ebpf直接抓取。记录函数调用路径，辅助调试复杂调用关系等。

```cpp
// instrument.c
#define _GNU_SOURCE
#include <stdio.h>
#include <dlfcn.h>
#include <time.h>

// 必须禁止钩子函数自身被插桩
void __attribute__((no_instrument_function))
__cyg_profile_func_enter(void *func, void *caller) {
    Dl_info info;
    dladdr(func, &info);
    printf("▶ ENTER: %s [%p]\n", info.dli_sname ? info.dli_sname : "unknown", func);
}

void __attribute__((no_instrument_function)) 
__cyg_profile_func_exit(void *func, void *caller) {
    Dl_info info;
    dladdr(func, &info);
    printf("◀ EXIT: %s [%p]\n", info.dli_sname ? info.dli_sname : "unknown", func);
}
```

```cpp
// main.c
#include <stdio.h>

void test_func() {
    sleep(1); // 模拟耗时操作
}

int main() {
    printf("Start tracing...\n");
    test_func();
    printf("End tracing\n");
    return 0;
}
// gcc -finstrument-functions main.c instrument.c -ldl -rdynamic -o demo
// ./demo
```

输出：
```shell
▶ ENTER: main [0xaaaadcf80bf4]
Start tracing...
▶ ENTER: test_func [0xaaaadcf80b94]
◀ EXIT: test_func [0xaaaadcf80b94]
End tracing
◀ EXIT: main [0xaaaadcf80bf4]
```

使用__attribute__((no_instrument_function))避免钩子函数自身被插桩。需自定义钩子函数，通常结合dladdr解析函数名和文件名。

通过addr2line工具将地址转换为源代码行号，结合perf等工具分析性能。

这个功能具体咋用的呢？举两个例子：

+ 开源工具[uftrace](https://github.com/namhyung/uftrace)就使用到了这个功能获取数据, uftrace是一款用于C、C++、Rust和Python程序的函数调用图追踪工具。

> User space C/C++/Rust functions, by either dynamically patching functions using -P., or else selective NOP patching using code compiled with -pg, -finstrument-functions or -fpatchable-function-entry=N.

+ [使用GCC函数插桩功能找到耗时异常的函数](https://zhuanlan.zhihu.com/p/706025483)

这篇文章通过函数插桩，在函数入口和出口自动插入钩子函数，用于统计函数耗时。在cyg_profile_func_enter中记录函数的开始ticks，在cyg_profile_func_exit中记录函数的结束ticks，两者相减便得到了函数运行消耗的ticks，再除以CPU频率（g_cs_hz）得到耗时。使用x86指令集架构中的RDTSC（Read Time Stamp Counter）指令读取处理器的时钟周期计数器。

如果耗时大于5ms，便将函数指针和耗时push到一个栈中，记录下来，后续打印出来。