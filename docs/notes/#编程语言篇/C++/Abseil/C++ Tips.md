## [https://abseil.io/tips/](https://abseil.io/tips/5)
## StringView
### 用法
```cpp
// C Convention
void TakesCharStar(const char* s);

// Old Standard C++ convention
void TakesString(const std::string& s);

// string_view C++ conventions
void TakesStringView(absl::string_view s);    // Abseil
void TakesStringView(std::string_view s);     // C++17
```
可以通过StringView提供一个接口，代替const char *和const std::string &。
StringView 只是对存在的一个字符串缓存区buffer提供了一个视图（view），只包括了一个指向这部分字符串的指针和长度，并不包括字符串buffer。
StringView是不能够修改buffer的，就像const char *不能修改指向的内存数据一样。如果要修改StringView指向的data，可以使用string(string_view_obj)转换成string对象，然后进行修改。
因此对StringView的拷贝只是浅拷贝，不涉及buffer内存拷贝，效率较高。
因为StringView不拥有buffer的所有权，因此需要buffer的生命周期长于StringView对象的生命周期。

tips：

- const stringView 中的const只是修饰stringView指向buffer的指针可不可以变化，stringview始终不能改变buffer的内容。
- Adding string_view into an existing codebase is not always the right answer: changing parameters to pass by string_view can be inefficient if those are then passed to a function requiring a std::string or a NUL-terminated const char*. It is best to adopt string_view starting at the utility code and working upward, or with complete consistency when starting a new project.

使用StringView的好处：作为函数参数，可以减少数据拷贝和strlen（）函数的调用。

### 源码
#### 成员变量
```cpp
  const char* ptr_;
  size_type length_;
};
```
保存了一个指针和长度，并不持有string的内存。
#### 构造函数的一个小技巧
```cpp
template <typename Allocator>
string_view(  // NOLINT(runtime/explicit)
const std::basic_string<char, std::char_traits<char>, Allocator>& str
ABSL_ATTRIBUTE_LIFETIME_BOUND) noexcept
// This is implemented in terms of `string_view(p, n)` so `str.size()`
// doesn't need to be reevaluated after `ptr_` is set.
// The length check is also skipped since it is unnecessary and causes
// code bloat.
: string_view(str.data(), str.size(), SkipCheckLengthTag{}) {}

// Implicit constructor of a `string_view` from a `const char*` and length.
constexpr string_view(const char* data, size_type len)
: ptr_(data), length_(CheckLengthInternal(len)) {}
```

```cpp
private:
  // The constructor from std::string delegates to this constructor.
  // See the comment on that constructor for the rationale.
  struct SkipCheckLengthTag {};
  string_view(const char* data, size_type len, SkipCheckLengthTag) noexcept
      : ptr_(data), length_(len) {}
```
可以看到，通过定义结构体SkipCheckLengthTag，调用到了第二个代码块中的构造函数，省去了CheckLengthInternal的操作。
#### Str Len计算
```cpp
  static constexpr size_type StrlenInternal(const char* str) {
#if defined(_MSC_VER) && _MSC_VER >= 1910 && !defined(__clang__)
    // MSVC 2017+ can evaluate this at compile-time.
    const char* begin = str;
    while (*str != '\0') ++str;
    return str - begin;
#elif ABSL_HAVE_BUILTIN(__builtin_strlen) || \
    (defined(__GNUC__) && !defined(__clang__))
    // GCC has __builtin_strlen according to
    // https://gcc.gnu.org/onlinedocs/gcc-4.7.0/gcc/Other-Builtins.html, but
    // ABSL_HAVE_BUILTIN doesn't detect that, so we use the extra checks above.
    // __builtin_strlen is constexpr.
    return __builtin_strlen(str);
#else
    return str ? strlen(str) : 0;
#endif
  }
```
根据宏定义条件判断，选择不同的计算方法，进行优化。
#### find函数优化
```cpp
class LookupTable {
 public:
  // For each character in wanted, sets the index corresponding
  // to the ASCII code of that character. This is used by
  // the find_.*_of methods below to tell whether or not a character is in
  // the lookup table in constant time.
  explicit LookupTable(string_view wanted) {
    for (char c : wanted) {
      table_[Index(c)] = true;
    }
  }
  bool operator[](char c) const { return table_[Index(c)]; }

 private:
  static unsigned char Index(char c) { return static_cast<unsigned char>(c); }
  bool table_[UCHAR_MAX + 1] = {};
};

}  // namespace

string_view::size_type string_view::find_first_of(
    string_view s, size_type pos) const noexcept {
  if (empty() || s.empty()) {
    return npos;
  }
  // Avoid the cost of LookupTable() for a single-character search.
  if (s.length_ == 1) return find_first_of(s.ptr_[0], pos);
  LookupTable tbl(s);
  for (size_type i = pos; i < length_; ++i) {
    if (tbl[ptr_[i]]) {
      return i;
    }
  }
  return npos;
}
```
根据查找字符的长度，如果长度==1，直接查找，不等于1，通过LookupTable进行索引，快速查找。
## operator和strCat
```cpp
std::string foo = LongString1();
std::string bar = LongString2();
std::string foobar = foo + bar;

std::string foo = LongString1();
std::string bar = LongString2();
std::string foobar = absl::StrCat(foo, bar);
```
foo+bar会调用+ operator函数。
```cpp
std::string foo = LongString1();
std::string bar = LongString2();
std::string baz = LongString3();
std::string foobarbaz = foo + bar + baz;

std::string foo = LongString1();
std::string bar = LongString2();
std::string baz = LongString3();
std::string foobarbaz = absl::StrCat(foo, bar, baz);
```
foo + bar + baz不存在调用三个对象相加的operator函数，实际上过程如下：
```
std::string temp = foo + bar;
std::string foobarbaz = std::move(temp) + baz;
```
会发生临时对象的生成，然后在调用+ operator。在c++ 11里面std::move(temp) + baz等效std::move(temp.append(baz))，不会创建第二个临时对象。但是如果temp的内存空间不够，append的时候就会申请新内存，然后进行拷贝，因此比较低效。
absl的strCat函数实现可以计算一共需要多少空间，然后reserve，再形成新的字符串，更加高效。
## absl StrSplit
```cpp
// Splits on commas. Stores in vector of string_view (no copies).
std::vector<absl::string_view> v = absl::StrSplit("a,b,c", ',');

// Splits on commas. Stores in vector of string (data copied once).
std::vector<std::string> v = absl::StrSplit("a,b,c", ',');

// Splits on literal string "=>" (not either of "=" or ">")
std::vector<absl::string_view> v = absl::StrSplit("a=>b=>c", "=>");

// Splits on any of the given characters (',' or ';')
using absl::ByAnyChar;
std::vector<std::string> v = absl::StrSplit("a,b;c", ByAnyChar(",;"));

// Stores in various containers (also works w/ absl::string_view)
std::set<std::string> s = absl::StrSplit("a,b,c", ',');
std::multiset<std::string> s = absl::StrSplit("a,b,c", ',');
std::list<std::string> li = absl::StrSplit("a,b,c", ',');

// Equiv. to the mythical SplitStringViewToDequeOfStringAllowEmpty()
std::deque<std::string> d = absl::StrSplit("a,b,c", ',');

// Yields "a"->"1", "b"->"2", "c"->"3"
std::map<std::string, std::string> m = absl::StrSplit("a,1,b,2,c,3", ',');
```
## RVO 返回值优化
所有现代C++编译器都默认执行RVO，即使在调试构建中也是如此，即使是对于非内联函数也是如此。
编译器会进行RVO的情况：
```cpp
class SomeBigObject {
 public:
  SomeBigObject() { ... }
  SomeBigObject(const SomeBigObject& s) {
    printf("Expensive copy …\n", …);
    …
  }
  SomeBigObject& operator=(const SomeBigObject& s) {
    printf("Expensive assignment …\n", …);
    …
    return *this;
  }
  ~SomeBigObject() { ... }
  …
};

SomeBigObject SomeBigObject::SomeBigObjectFactory(...) {
  SomeBigObject local;
  …
  return local;
}
// No message about expensive operations:
SomeBigObject obj = SomeBigObject::SomeBigObjectFactory(...);
```
不会RVO的情况：

- 1、调用函数使用一个已经存在的变量接收返回值
```cpp
// RVO won’t happen here; prints message "Expensive assignment ...":
obj = SomeBigObject::SomeBigObjectFactory(s2);
```

- 2、被调用函数有多个返回值变量
```cpp
// RVO won’t happen here:
static SomeBigObject NonRvoFactory(...) {
  SomeBigObject object1, object2;
  object1.DoSomethingWith(...);
  object2.DoSomethingWith(...);
  if (flag) {
    return object1;
  } else {
    return object2;
  }
}
```
但是，如果返回值是一个变量，在多个地方返回，也会进行RVO。例如：
```cpp
// RVO will happen here:
SomeBigObject local;
if (...) {
  local.DoSomethingWith(...);
  return local;
} else {
  local.DoSomethingWith(...);
  return local;
}
```
不声明变量，返回临时变量，RVO也起作用。
```cpp
// RVO works here:
SomeBigObject SomeBigObject::ReturnsTempFactory(...) {
  return SomeBigObject::SomeBigObjectFactory(...);
}
```
```cpp
// No message about expensive operations:
EXPECT_EQ(SomeBigObject::SomeBigObjectFactory(...).Name(), s);
```
## absl Substitute
```cpp
std::string GetErrorMessage(absl::string_view op, absl::string_view user,
                            int id) {
  return absl::Substitute("Error in $0 for user $1 ($2)", op, user, id);
}
```
## absl Status，强制调用者处理错误
```cpp
absl::Status Foo();

void CallFoo1() {
  Foo();
}
```
以上函数使用absl Status作为函数的返回值，会强制函数调用者（caller）对错误进行处理，否则编译出错。
```cpp
void CallFoo2() {
  Foo().IgnoreError();
}

void CallFoo3() {
  if (!status.ok()) std::abort();
}

void CallFoo4() {
  absl::Status status = Foo();
  if (!status.ok()) LOG(ERROR) << status;
}
```
使用Status，可以保证对错误的处理，即使调用IgnoreError函数，也可以使Code ReViewer不困惑，他会认为调用者知道这里可能会存在错误，但是错误无关紧要，不需要特殊处理。
当对错误如何处理不清楚时，通过返回Status，将状态信息向上传递，由上层进行处理。
It forces engineers to decide how to handle errors, and explicitly documents that in compilable code. 
## std invoke
从 C++17 起，C++提供了`std::invoke<>()`[1]来统一所有的 callback 形式

- Function
- Function Ptr
- lamda
- Functor
- Member Func
```cpp
template <typename Iter, typename Callable, typename... Args>
void foreach (Iter current, Iter end, Callable op, Args const &... args) {
    while (current != end) {     // as long as not reached the end of the elements
        std::invoke(op,            // call passed callable with
            args...,       // any additional args
            *current);     // and the current element
        ++current;
    }
}
```
[https://www.zhihu.com/column/c_1306966457508118528](https://www.zhihu.com/column/c_1306966457508118528)
## 函数模版不能偏特化

## 在类模板中，类模板函数只有在被调用时才会被instantiate。

## 模板元编程例子
[https://github1s.com/abseil/abseil-cpp/blob/HEAD/absl/base/internal/invoke.h](https://github1s.com/abseil/abseil-cpp/blob/HEAD/absl/base/internal/invoke.h)
invoke的实现，采用了模板元编程，通过SFINCE原理进行匹配，匹配函数和参数的类型然后执行不同的invoke方法。
void_t结合SFINAE技术进行类型选择推断与编译期表达式合法性检测
[https://zhuanlan.zhihu.com/p/377561143](https://zhuanlan.zhihu.com/p/377561143)