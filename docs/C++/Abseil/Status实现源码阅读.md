## Status and StatusOr
[https://github1s.com/abseil/abseil-cpp/blob/HEAD/absl/status/status.h#L849-L862](https://github1s.com/abseil/abseil-cpp/blob/HEAD/absl/status/status.h#L849-L862)
### Status 实现
Status抽象了很多状态，可以存储enum status、message、以及Payload。
分为两种模式，inline：不包括message和Payload，非inline。
Status和普通的enum不同，在内部存储的数据不是直接的enum数据，而是经过转换的，比如在Status中通过“｜2”操作在二进制位表示是否MoveBy。通过“&1”表示是否inline。
设计很巧妙，通过Status的数据成员rep_的两种不同表示实现。
```
  // Status supports two different representations.
  //  - When the low bit is set it is an inlined representation.
  //    It uses the canonical error space, no message or payload.
  //    The error code is (rep_ >> 2).
  //    The (rep_ & 2) bit is the "moved from" indicator, used in IsMovedFrom().
  //  - When the low bit is off it is an external representation.
  //    In this case all the data comes from a heap allocated Rep object.
  //    rep_ is a status_internal::StatusRep* pointer to that structure.
  uintptr_t rep_;
```

```
inline bool Status::IsMovedFrom(uintptr_t rep) { return (rep & 2) != 0; }

inline uintptr_t Status::MovedFromRep() {
  return CodeToInlinedRep(absl::StatusCode::kInternal) | 2;
}

inline uintptr_t Status::CodeToInlinedRep(absl::StatusCode code) {
  return (static_cast<uintptr_t>(code) << 2) + 1;
}

inline absl::StatusCode Status::InlinedRepToCode(uintptr_t rep) {
  assert(IsInlined(rep));
  return static_cast<absl::StatusCode>(rep >> 2);
}
```
### StatusOr 实现
StatusOr和Rust中的Result很相似，存在Error和Ok两种状态，Ok状态时还包括了数据Data，data可以通过模版指定类型，比如StatusOr<Food>。
```
template <typename T>
class StatusOr : private internal_statusor::StatusOrData<T>,
                 private internal_statusor::CopyCtorBase<T>,
                 private internal_statusor::MoveCtorBase<T>,
                 private internal_statusor::CopyAssignBase<T>,
                 private internal_statusor::MoveAssignBase<T> {
...
                 }
```
StatusOr存储数据Data是通过模版继承实现的，继承了StatusOrData类，在StatusOrData中定义了如下数据分别表示数据和Status。
```
  // status_ will always be active after the constructor.
  // We make it a union to be able to initialize exactly how we need without
  // waste.
  // Eg. in the copy constructor we use the default constructor of Status in
  // the ok() path to avoid an extra Ref call.
  union {
    Status status_;
  };

  // data_ is active iff status_.ok()==true
  struct Dummy {};
  union {
    // When T is const, we need some non-const object we can cast to void* for
    // the placement new. dummy_ is that object.
    Dummy dummy_;
    T data_;
  };
```
为什么要设计union呢？

- Status为什么要union？
- data_为什么要union？

这里举一个简单的例子，会报错如下：
```
#include <iostream>
#include <string>

int main()
{
  const int a = 4;
  void *p = &a;
  std::cout << p << std::endl;
}
```
```
main.cpp:8:9: error: cannot initialize a variable of type 'void *' with an rvalue of type 'const int *'
  void *p = &a;
        ^   ~~
1 error generated.
```
而通过union，dummy_和data_指向同一个内存区域，那么就可以通过dummy取地址，传递void *类型供PlacementNew函数使用，避免了T类型是const的情况。
通过new构造数据和Status：这里使用了PlacementNew的用法。
placement new就是**在用户指定的内存位置上**（这个内存是已经预先分配好的）构建新的对象，因此这个构建过程不需要额外分配内存，只需要调用对象的构造函数在该内存位置上构造对象即可。在已分配好的内存上进行对象的构建，构建速度快，已分配好的内存可以反复利用，有效的避免内存碎片问题。
（引用：[https://blog.csdn.net/qq_41453285/article/details/103547699](https://blog.csdn.net/qq_41453285/article/details/103547699)）
```
// Construct the value (ie. data_) through placement new with the passed
// argument.
template <typename... Arg>
void MakeValue(Arg&&... arg) {
  internal_statusor::PlacementNew<T>(&dummy_, std::forward<Arg>(arg)...);
}

// Construct the status (ie. status_) through placement new with the passed
// argument.
template <typename... Args>
void MakeStatus(Args&&... args) {
  internal_statusor::PlacementNew<Status>(&status_,
                                          std::forward<Args>(args)...);
}

// Construct an instance of T in `p` through placement new, passing Args... to
// the constructor.
// This abstraction is here mostly for the gcc performance fix.
template <typename T, typename... Args>
ABSL_ATTRIBUTE_NONNULL(1) void PlacementNew(void* p, Args&&... args) {
  new (p) T(std::forward<Args>(args)...);
}
```

数据析构函数中释放内存。
```
  ~StatusOrData() {
    if (ok()) {
      status_.~Status();
      data_.~T();
    } else {
      status_.~Status();
    }
  }

```