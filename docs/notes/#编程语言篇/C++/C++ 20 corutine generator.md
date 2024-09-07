
cmakeList.txt
```
cmake_minimum_required(VERSION 3.25)
project(co_generator)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS -fcoroutines)

add_executable(co_generator main.cpp)
```

main.cpp
```cpp
#include <iostream>
#include <coroutine>
#include <utility>
#include <tuple>

struct Generator {

    struct promise_type {
        Generator get_return_object() {
            return {handle::from_promise(*this)};
        }

        auto initial_suspend() noexcept {
            return std::suspend_never{};
        }

        auto final_suspend() noexcept {
            return std::suspend_always{};
        }

        void unhandled_exception() {
            std::terminate();
        }

        void return_void() {

        }

        auto yield_value(int value) {
            current_value_ = value;
            return std::suspend_always{};
        }

        int current_value_;
    };

    using handle = std::coroutine_handle<promise_type>;

    void next() {
        return coro_handle_.resume();
    }

    bool done() {
        return coro_handle_.done();
    }

    int corrent_value() {
        return coro_handle_.promise().current_value_;
    }

    Generator(std::coroutine_handle<promise_type> rhs) noexcept:
            coro_handle_(std::exchange(rhs, {})) {
    }

    ~Generator() {
        if (coro_handle_) {
            coro_handle_.destroy();
        }
    }

private:
    handle coro_handle_;

};


Generator fibo() {
    int a = 1, b = 1;
    while (a < 1000000) {
        co_yield a;
        std::tie(a, b) = std::make_tuple(b, a + b);
    }

    co_return;
}


int main() {
    std::cout << "Hello, World!" << std::endl;

    for (auto f = fibo(); !f.done(); f.next()) {
        std::cout << f.corrent_value() << std::endl;
    }
    return 0;
}

```

gcc g++ version
```
g++-10 (Ubuntu 10.3.0-1ubuntu1~20.04) 10.3.0
```

result
```
1
1
2
3
5
8
13
21
34
55
89
144
233
377
610
987
1597
2584
4181
6765
10946
17711
28657
46368
75025
121393
196418
317811
514229
832040
```