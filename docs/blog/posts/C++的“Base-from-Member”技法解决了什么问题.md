---
title: C++的“Base-from-Member”技法解决了什么问题？
date: 2025-03-12
authors: [KenForever1]
categories: 
  - C++
labels: []
comments: true
---

# C++的“Base-from-Member”技法解决了什么问题？

> C++ has indeed become too "expert friendly" -- Bjarne Stroustrup

Bjarne Stroustrup的说得很对，因为专家们对C++语言中的惯用法非常熟悉。随着程序员理解的惯用法的增加，该语言对他们来说变得更加友好。
这篇文章介绍的“[Base-from-Member](https://en.wikibooks.org/wiki/More_C%2B%2B_Idioms/Base-from-Member)”就是c++中的一种惯用法（idioms）。通过掌握现代 C++ 惯用法，提升这方面的知识，C++对我们来说就更加友好。

## “Base-from-Member”技法的目的

“Base-from-Member”技法的目的是，解决C++在继承中，如何初始化一个依赖派生类（子类）成员变量的基类。

<!-- more -->

先来看看，为什么会存在这个问题。
在C++中，有一个法则就是，先完成所有的基类初始化，然后再是派生类（子类）的所有成员变量初始化。

为什么呢？
这是因为派生类的成员可能会使用对象的基类部分。因此，所有的基类必须在派生类的成员之前被初始化。
但是，有些情况下，我们需要派生类中可用的数据成员去初始化我们的基类。
不知道你发现没，这就产生了冲突。这上面提到的规则相矛盾，因为传递给基类构造函数的参数（派生类的一个成员）必须完全初始化。这就产生了一个循环初始化问题。

### 通过一个例子发现问题

我们通过一个例子看一下这个问题：

```c++
#include <streambuf>
#include <ostream>

namespace std {
  class streambuf;
  class ostream {
    explicit ostream(std::streambuf * buf);
    //...
  };
}

// A customization of streambuf.
class fdoutbuf : public std::streambuf
{
public:
    explicit fdoutbuf( int fd );
    //...
};

class fdostream : public std::ostream
{
protected:
    fdoutbuf buf;
public:
    explicit fdostream( int fd )
        : buf( fd ), std::ostream( &buf )
        // This is not allowed: buf can't be initialized before std::ostream.
        // std::ostream needs a std::streambuf object defined inside fdoutbuf.
    {}
};
```

这个例子来源于[Boost libraries](https://en.wikipedia.org/wiki/Boost_(C%2B%2B_libraries))。
讲的是：
+ 通过继承streambuf类，实现自定义的fdoutbuf类。
+ fdoutbuf类声明了一个fdostream的成员变量。用来初始化fdostream的基类std::ostream。
+ std::ostream的初始化需要fdoutbuf类型的成员变量。
这不就产生了上面的基类和子类成员变量的循环初始化问题了吗？

### 如何解决循环初始化问题呢？

今天将的这个技法“base-from-member”，使用到了一个规则。就是基类初始化是按照声明的顺序进行的。比如：

```c++

class A;

class B;

class C: A, B {

    M m;
}

```
C类的基类初始化顺序是：先A，后B，然后再是C的成员m。

那么我们就可以添加一个新的类MC, C类去继承MC类，MC类负责初始化M m这个成员。并且继承顺序上MC在A、B的前面。

```c++

class MC{
    M m;
}

class C: MC, A, B{}

```

在这个技巧中，添加一个新类只是为了初始化派生类中导致问题的成员。这个新类在所有其他基类之前被引入到基类列表中。因为新类在需要完全构造参数的基类之前出现，所以它首先被初始化，然后可以像往常一样传递引用。

通过这种方式解决后的代码：

```c++
#include <streambuf>
#include <ostream>

class fdoutbuf : public std::streambuf
{
public:
    explicit fdoutbuf(int fd);
    //...
};

struct fdostream_pbase // A newly introduced class.
{
    fdoutbuf sbuffer; // The member moved 'up' the hierarchy.
    explicit fdostream_pbase(int fd)
        : sbuffer(fd)
    {}
};

class fdostream
    : protected fdostream_pbase // This class will be initialized before the next one.
    , public std::ostream
{
public:
    explicit fdostream(int fd)
        : fdostream_pbase(fd),   // Initialize the newly added base before std::ostream.
          std::ostream(&sbuffer) //  Now safe to pass the pointer.
    {}
    //...
};

int main()
{
  fdostream standard_out(1);
  standard_out << "Hello, World\n";
  return 0;
}
```
在解决后的代码中，加入了一个新的类fdostream_pbase，我们把fdostream类中的成员变量sbuffer移动到它中了。然后让fdostream先继承自fdostream_pbase。然后再继承std::ostream，并且把sbuffer指针传递给std::ostream。
从而，保证了sbuffer先初始化，然后再初始化std::ostream，这样指针可以安全传递给构造函数了。

你学会了这个技法了吗？别着急，我们看一个实战，在实际的代码中，可以构造一个[模板类detail::BaseFromMember](https://github.com/CVCUDA/CV-CUDA/blob/v0.2.0-alpha/src/nvcv_types/include/nvcv/detail/BaseFromMember.hpp)。

### detail::BaseFromMember模板类解决

模板类detail::BaseFromMember类这个例子来源于CVCUDA项目中的源码。

使用方法，可以通过BaseFromMember<Bar>定义包含Bar类型成员的基类。如果Far类也是一个基类，它的初始化依赖于Bar类型的成员，那么就可以使用Foo(BaseFromMember<Bar>::member)。如果依赖多个成员，这些成员是同一个类型，那么就可以给他们一个ID区分。

```c++
struct Bar
{
};

struct Foo
{
    Foo(Bar &, Bar * = nullptr);
};

struct A
    : BaseFromMember<Bar>,
      Foo
{
    using MemberBar = BaseFromMember<Bar>;

    A()
        : Foo(MemberBar::member)
    {
    }
};

struct B
    : BaseFromMember<Bar, 0>,
      BaseFromMember<Bar, 1>,
      Foo
{
    using MemberBar0 = BaseFromMember<Bar, 0>;
    using MemberBar1 = BaseFromMember<Bar, 1>;

    B()
        : Foo(MemberBar0::member, MemberBar1::member)
    {
    }
};
```

看看这个模板类具体怎么实现的：

```c++
template <class T, int ID = 0>
class BaseFromMember
{
public:
    T member;
};

template <class T, int ID>
class BaseFromMember<T &, ID>
{
public:
    BaseFromMember(T &m)
        : member(m)
    {
    }

    T &member;
};
```
看了代码是不是很简单，定义了一个模板类，声明了T &member的成员，采用特化的方式指定默认ID为0，也可以指定参数ID。

看一个CVCUDA中具体的例子：
```c++
class TensorDataAccessStrided
    // declaration order is important here
    : private detail::BaseFromMember<TensorShapeInfo>
    , public detail::TensorDataAccessStridedImpl
{
public:
    static bool IsCompatible(const ITensorData &data)
    {
        return dynamic_cast<const ITensorDataStrided *>(&data) != nullptr;
    }

    static detail::Optional<TensorDataAccessStrided> Create(const ITensorData &data)
    {
        if (auto *dataStrided = dynamic_cast<const ITensorDataStrided *>(&data))
        {
            return TensorDataAccessStrided(*dataStrided);
        }
        else
        {
            return detail::NullOpt;
        }
    }

    TensorDataAccessStrided(const TensorDataAccessStrided &that)
        : MemberShapeInfo(that)
        , detail::TensorDataAccessStridedImpl(that, MemberShapeInfo::member)
    {
    }

private:
    using MemberShapeInfo = detail::BaseFromMember<TensorShapeInfo>;

    TensorDataAccessStrided(const ITensorDataStrided &data)
        : MemberShapeInfo{*TensorShapeInfo::Create(data.shape())}
        , detail::TensorDataAccessStridedImpl(data, MemberShapeInfo::member)
    {
    }
};
```

感谢您的关注于支持！！