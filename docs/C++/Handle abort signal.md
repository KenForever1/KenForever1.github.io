```c++
#include <csetjmp>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <dlfcn.h>

jmp_buf env;

void on_sigabrt(int signum) {
    signal(signum, SIG_DFL);
    longjmp(env, 1);
}

void try_and_catch_abort(void (*func)(void)) {
    if (setjmp (env) == 0) {
        signal(SIGABRT, &on_sigabrt);
        (*func)();
        signal(SIGABRT, SIG_DFL);
    } else {
        std::cout << "aborted\n";
    }
}

void do_stuff_aborted() {
    std::cout << "step 1\n";
    abort();
    std::cout << "step 2\n";
}

void do_stuff() {
    std::cout << "step 1\n";
    std::cout << "step 2\n";
}

void load_so() {
    typedef int (*some_func)(int);

    void *myso = dlopen("libdemo_so.so",
                        RTLD_NOW);
    auto func = (int (*)(int)) dlsym(myso, "hello");
    auto res = func(10);
    std::cout << "res is : " << res << std::endl;
    dlclose(myso);
}

int main() {
//    try_and_catch_abort(&do_stuff_aborted);
//    try_and_catch_abort(&do_stuff);
//    do_stuff_aborted();

//    try_and_catch_abort(&load_so);
    load_so();
    do_stuff();
}

```


```c++
#include <iostream>

extern "C" int hello(int num) {
    std::cout << "num is : " << num << std::endl;

    abort();
    return num - 1;
}
```

```
cmake_minimum_required(VERSION 3.24)
project(untitled)

set(CMAKE_CXX_STANDARD 17)

add_executable(untitled main.cpp)
target_link_libraries(untitled dl)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(demo_so SHARED demo_so.cpp)
```

[setjump and  longjump 用法](https://www.cnblogs.com/hazir/p/c_setjmp_longjmp.html)
