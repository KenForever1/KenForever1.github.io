## Shared Ptr
cpp标准库的shared_ptr以及上面的样例代码都不是线程安全的。如果两个线程同时操作一个SharedPtr对象，那么很可能会导致内存错误。
以FB的folly库的AtomicSharedPtr为代表，实现了原子变更“对象指针+引用计数+alias对象”的功能，真正实现线程安全的原子引用计数对象管理。
如果要实现一个AtomicSharedPtr，需要解决的一个问题就是如何用一个原子操作同时变更指针+引用计数。好在x64平台的虚拟内存地址有个机制是地址的高16位都是0，可以利用这16位做引用计数，就可以基于64位的CAS实现同时变更指针+引用计数的功能了。这也是folly中PackedSyncPtr基本原理。基于这个功能，就可以实现一个线程安全的AtomicSharedPtr，用来在多线程环境下管理对象的生命周期了。
## 动态库的可见性
[http://zh0ngtian.tech/posts/d979a13.html](http://zh0ngtian.tech/posts/d979a13.html)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
```
void __attribute__ ((visibility ("default"))) visible_fun();
```
## conan
一个c++ 包管理器，python编写，使用pip安装。
```
pip install conan
```
[https://docs.conan.io/2/tutorial/consuming_packages.html](https://docs.conan.io/2/tutorial/consuming_packages.html)
conan1 和conan2 版本使用上存在变换，conan1对CMakeLists.txt文件是侵入式的，使用中需要在其中加入：
```
include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()
```
conan2不需要，推荐使用conan2。
conan2 有两种conanfile，分别是conanfile.txt和conanfile.py，py文件提供了python接口，可以操作build、source、package等阶段，自带cmake工具操作接口。
使用：
```
conan install . --output-folder=build --build=missing
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build .
```
## 降低编译时间
[https://blog.zaleos.net/giving-ccache-distcc-a-spin](https://blog.zaleos.net/giving-ccache-distcc-a-spin/)
Ccache & Distcc
## c++STL 并行
[https://0cch.com/2021/04/05/parallel-algorithm-in-stl/](https://0cch.com/2021/04/05/parallel-algorithm-in-stl/)
std::execution::par_unseq c++17

## vector

- clear vector
- 1. 第一种，使用clear方法清空所有元素。然后使用shrink_to_fit方法把capacity和size（0）对齐，达到释放内存的作用
```shell
#include <iostream>
#include <vector>
int main(int argc, char const *argv[])
{
    std::vector<int> vi;
    vi.reserve(1024);
    for (int i = 0; i < 1024; i++) vi.push_back(i);
    std::cout << vi.size() << " " << vi.capacity() << std::endl;    //1024 1024
    vi.clear(); 
    std::cout << vi.size() << " " << vi.capacity() << std::endl;    //0 1024
    vi.shrink_to_fit(); 
    std::cout << vi.size() << " " << vi.capacity() << std::endl;    //0 0
}
```

- 2. 第二种，使用swap方法；
```shell
#include <iostream>
#include <vector>
int main(int argc, char const *argv[])
{
    std::vector<int> vi;
    vi.reserve(1024);
    for (int i = 0; i < 1024; i++) vi.push_back(i);
    std::cout << vi.size() << " " << vi.capacity() << std::endl;    //1024 1024
    std::vector<int>().swap(vi); //使用临时量（size =0, capacity=0）和vi交换，临时量会立即析构
    std::cout << vi.size() << " " << vi.capacity() << std::endl;    //0 0
}
```
## Core Guidline
[https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#S-class](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#S-class)

## inline namespace (c++11)
[https://blog.csdn.net/craftsman1970/article/details/82872497](https://blog.csdn.net/craftsman1970/article/details/82872497)
可以应用于库的版本升级

## trivial_view class
struck_pack源码阅读--通过trivial_veiw减少内存拷贝
[https://github1s.com/alibaba/yalantinglibs/blob/main/include/ylt/struct_pack/trivial_view.hpp](https://github1s.com/alibaba/yalantinglibs/blob/main/include/ylt/struct_pack/trivial_view.hpp)
trivial_view通过内部的指针实现，提供了原始数据的一个视图，减少参数传递中的数据拷贝。
```cpp
 * struct Data {
 *   int x[10000],y[10000],z[10000];
 * };
 *
 * struct Proto {
 *   std::string name;
 *   Data data;
 * };

* struct ProtoView {
 *   std::string_view name;
 *   struct_pack::trivial_view<Data> data;
 * };
```

```cpp
template <typename T, typename>
struct trivial_view {
 private:
  const T* ref;

 public:
  trivial_view(const T* t) : ref(t){};
  trivial_view(const T& t) : ref(&t){};
  trivial_view(const trivial_view&) = default;
  trivial_view(trivial_view&&) = default;
  trivial_view() : ref(nullptr){};

  trivial_view& operator=(const trivial_view&) = default;
  trivial_view& operator=(trivial_view&&) = default;

  using value_type = T;

  const T& get() const {
    assert(ref != nullptr);
    return *ref;
  }
  const T* operator->() const {
    assert(ref != nullptr);
    return ref;
  }
};
```
在deserilize时，实现了memory_reader从序列化的内存数据中读取对象的field。
```cpp
struct memory_reader {
  const char *now;
  const char *end;
  constexpr memory_reader(const char *beg, const char *end) noexcept
      : now(beg), end(end) {}
  bool read(char *target, size_t len) {
    if SP_UNLIKELY (now + len > end) {
      return false;
    }
    memcpy(target, now, len);
    now += len;
    return true;
  }
  const char *read_view(size_t len) {
    if SP_UNLIKELY (now + len > end) {
      return nullptr;
    }
    auto ret = now;
    now += len;
    return ret;
  }
  bool ignore(size_t len) {
    if SP_UNLIKELY (now + len > end) {
      return false;
    }
    now += len;
    return true;
  }
  std::size_t tellg() { return (std::size_t)now; }
};
```
在反序列化的时候，如果是trival_view类型就调用read_view，否则调用read。从上面的read_view实现可以看出，相比read的实现，read_view没有发生内存拷贝操作，只是指针指向位置的改变。
```cpp
if constexpr (is_trivial_view_v<type>) {
      static_assert(view_reader_t<Reader>,
                    "The Reader isn't a view_reader, can't deserialize "
                    "a trivial_view<T>");
      const char *view = reader_.read_view(sizeof(typename T::value_type));
	...
}else{
    ...
    if SP_UNLIKELY (!reader_.read((char *)&item, sizeof(type))) {
    	return struct_pack::errc::no_buffer_space;
    }
    ...
}
```
## 源码阅读--iguana如何实现enum reflection
我们知道通过反射可以在运行时获取类名、类型等一些信息，然而c++语言本身没有提供类似java这种反射机制。在阅读iguana开源库时，看到了一种EnumRefection的实现，分享给大家。
通过模板传入类型T，然后获取__PRETTY_FUNCTION__字符串，通过int默认类型找到T类型所在的开始结束位置，然后通过字符串截取，把string信息拿出来，以实现反射。
定义的函数get_raw_name，通过int进行模板实例化以后返回的字符串是：
std::string_view get_raw_name() [T = int]，可以看到这个字符串中就保存了int的类型信息。
下面看一下iguana库源码中的实现，iguana是一个c++实现的序列化库：
源码地址：[https://github1s.com/alibaba/yalantinglibs/blob/main/include/ylt/thirdparty/iguana/enum_reflection.hpp#L9-L53](https://github1s.com/alibaba/yalantinglibs/blob/main/include/ylt/thirdparty/iguana/enum_reflection.hpp#L9-L53)
（注：通过在github.com中加入1s，如上面的链接，可以通过这个工具打开网页版的vscode进行源码阅读）
```cpp
template <typename T>
constexpr std::string_view get_raw_name() {
    #ifdef _MSC_VER
    return __FUNCSIG__;
    #else
    return __PRETTY_FUNCTION__;
    #endif
}

template <auto T>
constexpr std::string_view get_raw_name() {
    #ifdef _MSC_VER
    return __FUNCSIG__;
    #else
    return __PRETTY_FUNCTION__;
    #endif
}

template <typename T> // 用于类型
inline constexpr std::string_view type_string() {
    constexpr std::string_view sample = get_raw_name<int>();
    constexpr size_t pos = sample.find("int"); // 找到类型名开始位置
    constexpr std::string_view str = get_raw_name<T>();
    constexpr auto next1 = str.rfind(sample[pos + 3]); // 从右往前找类型结束后的字符，找到类型名结束的位置
    #if defined(_MSC_VER)
    constexpr auto s1 = str.substr(pos + 6, next1 - pos - 6);
    #else
    constexpr auto s1 = str.substr(pos, next1 - pos);
    #endif
    return s1;
}

template <auto T> // 用于值
inline constexpr std::string_view enum_string() {
    constexpr std::string_view sample = get_raw_name<int>();
    constexpr size_t pos = sample.find("int");
    constexpr std::string_view str = get_raw_name<T>();
    constexpr auto next1 = str.rfind(sample[pos + 3]);
    #if defined(__clang__) || defined(_MSC_VER)
    constexpr auto s1 = str.substr(pos, next1 - pos);
    #else
    constexpr auto s1 = str.substr(pos + 5, next1 - pos - 5);
    #endif
    return s1;
}
```
下面通过一个Person结构体，以及enum Status进行简单测试：
```cpp
struct Person{
    std::string name;    
    int age;
};

enum class Status{
    Good,
    Bad
};
int main(){
    
    constexpr std::string_view sample = get_raw_name<int>();
    std::cout << sample << std::endl;
    
    constexpr std::string_view sample1 = get_raw_name<Status::Good>();
    std::cout << sample1 << std::endl;
    
    std::cout << type_string<float>() << std::endl;
    std::cout << enum_string<12>() << std::endl;
    
    std::cout << type_string<Person>() << std::endl;
    std::cout << enum_string<Status::Good>() << std::endl;
}
```
可以看到实现了值和类型的反射：
```cpp
std::string_view get_raw_name() [T = int]
std::string_view get_raw_name() [T = Status::Good]
float
12
Person
Status::Good
```
运行工具：[https://cpp.sh/](https://cpp.sh/)
## 补码与类型转换
```cpp
#include <stdio.h>

int main(){
    char a = 0b01111111;
    // char a = 0x7f;
    
    unsigned int b  = (unsigned int)a;
    printf("%d\n", b); // 127
    return 0;
}
```

```cpp
#include <stdio.h>

int main(){
    { 
        char a = 0b11111111;
        // char a = 0xff;
        
        unsigned int b  = (unsigned int)a;
        printf("%d\n", b); // -1
    }

    {
        char a = 0b11111111;
        // char a = 0xff;
        
        int b  = (int)a;
        printf("%d\n", b); // -1
        return 0;
    }
    
    {
        unsigned char a = 0b11111111;
        // char a = 0xff;
        
        int b  = a;
        printf("%d\n", b); // 255
        return 0;
    }
    return 0;
}

```
char类型是有符号类型，当a的值是0xff时，最高位时1，根据补码原理，转换成unsigned int和int 时值为负数，即-1。
而unsigned char类型是无符号类型，转换成unsigned int和int 时值为正数255。

## 获取map表
通过查阅map表，可以了解程序代码的函数地址信息。
以gcc为例：
```cpp
gcc test.c -Xlinker -Map=my.map
```

## 六大金刚
[https://modern-cpp.readthedocs.io/zh_CN/latest/gang_of_6.html#id7](https://modern-cpp.readthedocs.io/zh_CN/latest/gang_of_6.html#id7)
```
// Example program
#include <iostream>
struct Foo {
   Foo(int a) : p{new int(a)} {}

   Foo(Foo const& rhs) : p{new int(*rhs.p)} {
        printf("Foo(Foo const& rhs) ");
    }
   auto operator=(Foo const& rhs) -> Foo& {
       printf("operator=(Foo const& rhs) ");
     delete p; p = new int{*rhs.p};
     return *this;
   }

   Foo(Foo&& rhs) : p{rhs.p} { 
       printf(" Foo(Foo&& rhs) ");
       rhs.p = nullptr; }
   auto operator=(Foo&& rhs) -> Foo& {
       printf("operator=(Foo&& rhs)");
       
     delete p; p = rhs.p; rhs.p = nullptr;
     return *this;
   }

   ~Foo() { delete p; }

private:
   int* p;
};


struct Bar : Foo {
  using Foo::Foo;
  
//   ~Bar() = default;

//   ~Bar() { /* do something */ } // 自定义了析构函数，就不会生成默认的Move构造函数，但是会生成Copy构造函数。执行会打印：Foo(Foo const& rhs)  
// ，注释以后执行会打印： Foo(Foo&& rhs)
};
int main()
{
Bar bar{10};
Bar bar2{std::move(bar)};
// Bar bar2(std::move(bar));
}
```


## C++ 变量初始化
```
#include <iostream>
class Config{
    public:
    Config(){
        std::cout << "call Config ...\n";
    }

    Config(int _a){
        std::cout << "call Config _a\n";
        a = _a;
    }
    private:
    int a;
};

int  main(){

    int a;
    Config config(a);

    // Config config1(int a);
    // config1(a);

    // 错误初始化变量，编译器会认为是声明了一个函数
    // warning: empty parentheses were disambiguated as a function declaration [-Wvexing-parse]
    Config config2();
    
    // 正确初始化变量
    Config config3;

    return 0;
}
```