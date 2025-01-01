[TOC]

[`std::barrier`](https://en.cppreference.com/w/cpp/thread/barrier "`std::barrier`") 和 [`std::latch`](https://en.cppreference.com/w/cpp/thread/latch "`std::latch`") 是 C++20 引入的同步原语，用于多线程编程中的线程协调。

> The class template std::barrier provides a thread-coordination mechanism that blocks a group of threads of known size until all threads in that group have reached the barrier. Unlike std::latch, barriers are reusable: once a group of arriving threads are unblocked, the barrier can be reused. Unlike std::latch, barriers execute a possibly empty callable before unblocking threads.

### `std::barrier`

`std::barrier` 是一个可以多次使用的同步原语，允许一组线程在某个点进行同步。它的主要用途是在线程需要在某个阶段等待其它线程完成某些操作时使用。`std::barrier` 的一个显著特点是它可以被重用。

#### 使用示例

```cpp
#include <iostream>
#include <barrier>
#include <thread>
#include <vector>
#include <syncstream>

void task(std::barrier<> &sync_point, int id)
{

    std::osyncstream(std::cout) << "Task " << id << " is starting.\n";

    // 模拟工作

    std::this_thread::sleep_for(std::chrono::milliseconds(100 * id));

    std::osyncstream(std::cout) << "Task " << id << " is waiting at the barrier.\n";

    sync_point.arrive_and_wait();

    std::osyncstream(std::cout) << "Task " << id << " has passed the barrier.\n";
}

int main()
{

    const int num_threads = 5;

    std::barrier sync_point(num_threads);

    std::vector<std::thread> threads;

    for (int i = 0; i < num_threads; ++i)
    {

        threads.emplace_back(task, std::ref(sync_point), i);
    }

    for (auto &t : threads)
    {

        t.join();
    }

    return 0;
}
```

```
g++ main.cpp -std=c++20
```

```
Task 2 is starting.
Task 1 is starting.
Task 0 is starting.
Task 3 is starting.
Task 4 is starting.
Task 0 is waiting at the barrier.
Task 1 is waiting at the barrier.
Task 2 is waiting at the barrier.
Task 3 is waiting at the barrier.
Task 4 is waiting at the barrier.
Task 4 has passed the barrier.
Task 3 has passed the barrier.
Task 2 has passed the barrier.
Task 1 has passed the barrier.
Task 0 has passed the barrier.
```

### `std::latch`

`std::latch` 是一个一次性使用的同步原语，用于等待一组线程完成任务。它在初始化时设置一个计数器，线程可以调用 `count_down()` 来减少计数器，当计数器达到零时，等待的线程将继续执行。

#### 使用示例

```cpp
#include <iostream>

#include <latch>

#include <thread>

#include <vector>

#include <syncstream>

void task(std::latch &sync_point, int id)
{

    std::osyncstream(std::cout) << "Task " << id << " is starting.\n";

    // 模拟工作

    std::this_thread::sleep_for(std::chrono::milliseconds(100 * id));

    std::osyncstream(std::cout) << "Task " << id << " is done.\n";

    sync_point.count_down();
}

int main()
{

    const int num_threads = 5;

    std::latch sync_point(num_threads);

    std::vector<std::thread> threads;

    for (int i = 0; i < num_threads; ++i)
    {

        threads.emplace_back(task, std::ref(sync_point), i);
    }

    sync_point.wait(); // 等待所有线程完成

    for (auto &t : threads)
    {

        t.join();
    }

    std::osyncstream(std::cout) << "All tasks have completed.\n";

    return 0;
}
```

```
 Task 0 is starting.
Task 1 is starting.
Task 2 is starting.
Task 3 is starting.
Task 0 is done.
Task 4 is starting.
Task 1 is done.
Task 2 is done.
Task 3 is done.
Task 4 is done.
All tasks have completed.
```

### 总结

- `std::barrier` 用于可重复使用的同步，允许线程在多个阶段进行协调。
- `std::latch` 用于一次性同步，通常用于等待一组线程完成某个任务。
