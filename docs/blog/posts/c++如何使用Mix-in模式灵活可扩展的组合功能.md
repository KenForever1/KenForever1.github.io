---
title: c++如何使用Mix-in模式灵活可扩展的组合功能
date: 2025-03-09
authors: [KenForever1]
categories: 
  - C++
labels: []
comments: true
---


## Mix-in 是什么？

在讨论 mix-in 是什么之前，我们先看它试图解决什么问题？

假设你在开发一款程序的过程中，有一堆想法或概念要建模。它们可能在某种程度上相关，但在很大程度上是正交的——这意味着它们可以独立存在。
你有以下方式选择：

<!-- more -->

- 现在你可以通过继承来建模，并让这些概念中的每一个都从某个公共接口类派生。然后在实现该接口的派生类中提供具体方法。这种方法的问题是，这种设计没有提供任何清晰直观的方法来将这些具体的类组合在一起。

- mix-in 的方式。mix-in 的想法是提供一堆原始类，其中每个类都对一个基本的正交概念进行建模，并且能够将它们组合在一起，以仅使用你想要的功能组成更复杂的类——有点像乐高积木。原始类本身旨在用作构建块。这是可扩展的，因为以后你可以在不影响现有类的情况下向集合中添加其他原始类。

## Mix-in 如何实现？

Mix-in 通常以模板类的形式实现，以便在编译时将功能混入到目标类中。它允许增加类的功能而不需要修改类层次结构。
常用于实现类似于多重继承的功能，但避免了多重继承的复杂性。

在 C++的实现中，一种实现方式是使用模板和继承。这里的基本思想是通过模板参数提供这些构建块，从而将它们连接在一起。然后，例如通过 typedef 将它们链接在一起，以形成一个包含所需功能的新类型。

以你的示例为例，假设我们要在其之上添加重做功能。它可能如下所示：

以在 Number 类上，添加 Undo（撤销功能）和 Redo（再做一次）为例，例子来源于[what-are-mixins-as-a-concept](https://stackoverflow.com/questions/18773367/what-are-mixins-as-a-concept "what-are-mixins-as-a-concept")。实现如下：

```cpp
#include <iostream>
using namespace std;

struct Number
{
  typedef int value_type;
  int n;
  void set(int v) { n = v; }
  int get() const { return n; }
};

template <typename BASE, typename T = typename BASE::value_type>
struct Undoable : public BASE
{
  typedef T value_type;
  T before;
  void set(T v) { before = BASE::get(); BASE::set(v); }
  void undo() { BASE::set(before); }
};

template <typename BASE, typename T = typename BASE::value_type>
struct Redoable : public BASE
{
  typedef T value_type;
  T after;
  void set(T v) { after = v; BASE::set(v); }
  void redo() { BASE::set(after); }
};

typedef Redoable< Undoable<Number> > ReUndoableNumber;

int main()
{
  ReUndoableNumber mynum;
  mynum.set(42); mynum.set(84);
  cout << mynum.get() << '\n';  // 84
  mynum.undo();
  cout << mynum.get() << '\n';  // 42
  mynum.redo();
  cout << mynum.get() << '\n';  // back to 84
}
```

这里只是举例说明 mix-in 的用法，这个例子实际使用需要考虑边界情况和特殊用法。例如，在从未设置数字的情况下执行 undo 操作可能不会像你期望的那样表现。

## 通过类继承的方式实现的 mix-in 的一种方法

Mixin(Mix in) 是一种将若干功能独立的类通过继承的方式实现模块复用的 C++模板编程技巧。

```
template<typename... Mixins>
class MixinClass : public Mixins... {
  public:
    MixinClass() :  Mixins...() {}
  // ...
};
```

通过这种方式，我们可以将多个功能独立的类组合成一个类，从而实现模块复用。

```cpp
// point_example.cpp

#include <string>
#include <iostream>


template <typename... Mixins>
class Point : public Mixins... {
 public:
  double x, y;
  Point() : Mixins()..., x(0.0), y(0.0) {}
  Point(double x, double y) : Mixins()..., x(x), y(y) {}
};

class Label {
 public:
  std::string label;
  Label() : label("") {}
};

class Color {
 public:
  unsigned char red = 0, green = 0, blue = 0;
};

using MyPoint = Point<Label, Color>;

int main(){
  MyPoint p(1.0, 2.0);

  p.label = "MyPoint";
  p.red = 255;
  p.green = 0;
  p.blue = 0;

  std::cout << p.label << std::endl;
  std::cout << "(" << p.x << ", " << p.y << ")" << std::endl;
  std::cout << "Color: (" << (int)p.red << ", " << (int)p.green << ", " << (int)p.blue << ")" <<std::endl;
  return 0;
}

// MyPoint
// (1, 2)
// Color: (255, 0, 0)
```

上面的方式，在 Point 的功能基础上，加入了 Label 和 Color 的功能。除此之外还可以加入其他功能。

例子参考了[C++编程技巧：Mixin](https://zhuanlan.zhihu.com/p/460825741 "C++编程技巧：Mixin")
