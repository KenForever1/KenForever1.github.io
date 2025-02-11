
## std::make_move_iterator实际用法

先来看一个例子，
```c++
void
ProfileDataCollector::AddData(
    InferenceLoadMode& id, std::vector<RequestRecord>&& request_records)
{
  auto it = FindExperiment(id);

  if (it == experiments_.end()) {
    Experiment new_experiment{};
    new_experiment.mode = id;
    // 移动request_records，避免拷贝
    new_experiment.requests = std::move(request_records);
    experiments_.push_back(new_experiment);
  } else {
    it->requests.insert(
        it->requests.end(), std::make_move_iterator(request_records.begin()),
        std::make_move_iterator(request_records.end()));
  }
}
```
上面这段代码，截取自[triton-inference-server/perf_analyzer](https://github.com/triton-inference-server/perf_analyzer/blob/main/src/profile_data_collector.cc#L67-L83)，将vector中的requst内容移动的方式插入requests中，避免了拷贝。

使用std::make_move_iterator将request_records中的元素移动到目标位置，以避免不必要的拷贝。

!!! Tip:
    triton-inference-server/perf_analyzer是TritonServer的性能分析器命令行工具，它可以通过在你尝试不同的优化策略时测量性能变化，帮助你优化在 Triton 推理服务器上运行的模型的推理性能。

## std::make_move_iterator是什么？

```c++
template< class Iter >
std::move_iterator<Iter> make_move_iterator( Iter i );
```

std::make_move_iterator 是 C++ 标准库中的一个工具函数，用于将普通迭代器转换为移动迭代器。移动迭代器允许在迭代时使用移动语义，从而避免不必要的拷贝操作，提升性能。

make_move_iterator是一个方便的函数模板，它为给定的迭代器i构造一个std::move_iterator，其类型从参数的类型推导得出。

!!! Tip:
    cpp学习中，详细用法都可以去cppreference官网查看。比如：https://en.cppreference.com/w/cpp/iterator/make_move_iterator

```c++
#include <iomanip>
#include <iostream>
#include <iterator>
#include <list>
#include <string>
#include <vector>
 
auto print = [](const auto rem, const auto& seq)
{
    for (std::cout << rem; const auto& str : seq)
        std::cout << std::quoted(str) << ' ';
    std::cout << '\n';
};
 
int main()
{
    std::list<std::string> s{"one", "two", "three"};
 
    std::vector<std::string> v1(s.begin(), s.end()); // copy
 
    std::vector<std::string> v2(std::make_move_iterator(s.begin()),
                                std::make_move_iterator(s.end())); // move
 
    print("v1 now holds: ", v1);
    print("v2 now holds: ", v2);
    print("original list now holds: ", s);
}
```

```
v1 now holds: "one" "two" "three" 
v2 now holds: "one" "two" "three" 
original list now holds: "" "" ""
```

## std::move_iterator是什么？

```c++
template< class Iter >
class move_iterator;
```

std::move_iterator是一种迭代器适配器，其行为与基础迭代器完全相同。不同之处在于解引用将基础迭代器返回的值转换为**右值**。如果std::move_iterator迭代器用作输入迭代器，其效果是从源位置**移动**值，而不是复制值。

```c++
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <ranges>
#include <string>
#include <string_view>
#include <vector>
 
void print(const std::string_view rem, const auto& v)
{
    std::cout << rem;
    for (const auto& s : v)
        std::cout << std::quoted(s) << ' ';
    std::cout << '\n';
};
 
int main()
{
    std::vector<std::string> v{"this", "_", "is", "_", "an", "_", "example"};
    print("Old contents of the vector: ", v);
    std::string concat;
    for (auto begin = std::make_move_iterator(v.begin()),
              end = std::make_move_iterator(v.end());
         begin != end; ++begin)
    {
        std::string temp{*begin}; // moves the contents of *begin to temp
        concat += temp;
    }
 
    // Starting from C++17, which introduced class template argument deduction,
    // the constructor of std::move_iterator can be used directly:
    // std::string concat = std::accumulate(std::move_iterator(v.begin()),
    //                                      std::move_iterator(v.end()),
    //                                      std::string());
 
    print("New contents of the vector: ", v);
    print("Concatenated as string: ", std::ranges::single_view(concat));
}
```
可以看到v中的string都被move掉了，新的内容就是空字符串了。
```
Old contents of the vector: "this" "_" "is" "_" "an" "_" "example"
New contents of the vector: "" "" "" "" "" "" ""
Concatenated as string: "this_is_an_example"
```

再看一个unique_ptr的例子：
```c++
#include <iostream>
#include <vector>
#include <iterator>
#include <memory>

int main() {
    std::vector<std::unique_ptr<int>> v1;
    v1.push_back(std::make_unique<int>(1));
    v1.push_back(std::make_unique<int>(2));

    std::vector<std::unique_ptr<int>> v2;

    // 使用移动迭代器将v1的元素移动到v2
    v2.assign(std::make_move_iterator(v1.begin()), std::make_move_iterator(v1.end()));

    // v1中的元素已被移动，现在为空

    for(const auto &item : v1){
        if(item == nullptr){
            std::cout << "v1 null" << std::endl;
        }
    }

    for(const auto &item : v2){
        if(item == nullptr){
            std::cout << "v2 null" << std::endl;
        }else{
            std::cout << "v2 item not null" << std::endl;
        
        }
    }
    std::cout << "v1 size: " << v1.size() << '\n';  // 输出: v1 size: 2
    std::cout << "v2 size: " << v2.size() << '\n';  // 输出: v2 size: 2

    return 0;
}
```
v1中的指针被移动后，就变成了nullptr了。移动到v2中了。
```
v1 null
v1 null
v2 item not null
v2 item not null
v1 size: 2
v2 size: 2
```

从上面的介绍可以看到，使用std::make_move_iterator通过移动的方式，以避免不必要的拷贝。std::make_move_iterator 用于创建移动迭代器，帮助在需要时通过移动语义高效转移资源，避免不必要的拷贝，特别适合处理不可拷贝或移动成本较低的对象。
