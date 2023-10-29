## FTADLE
```cpp
#include <iostream>

template <typename FormatSink>
void AbslStringify(FormatSink& sink, int a) {
    sink << "int stringify : " << a << std::endl;
}

template <typename FormatSink>
void AbslStringify(FormatSink& sink, float a) {
    sink << "float stringify : " << a << std::endl;
}

struct PointStringify {
template <typename FormatSink>
friend void AbslStringify(FormatSink& sink, const PointStringify& p) {
    // sink.Append("(");
    // sink.Append(absl::StrCat(p.x));
    // sink.Append(", ");
    // sink.Append(absl::StrCat(p.y));
    // sink.Append(")");
    sink << " x : " << p.x << " y : " << p.y << std::endl;
}

double x = 10.0;
double y = 20.0;
};

struct AStringify {
template <typename FormatSink>
friend void AbslStringify(FormatSink& sink, const AStringify& p) {
    sink << p.a << std::endl;
}

std::string a = "hello i am a";
};

namespace b {
    void func(const std::string& s) {}
    namespace internal {
        void func(int a) {}
        namespace deep {
            void test() {
                std::string s = "hello";
                b::func(s);
            }
        }  // namespace deep
    }  // namespace internal
}  // namespace b

int main() {
    b::internal::deep::test();

    int i = 111;
    AbslStringify(std::cout, i);

    float b = 111.222;
    AbslStringify(std::cout, b);

    PointStringify p;
    AbslStringify(std::cout, p);

    AStringify a;
    AbslStringify(std::cout, a);
    return 0;
}

```
```cpp
int stringify : 111
float stringify : 111.222
 x : 10 y : 20
hello i am a
```
## ADL
argument-dependent lookup ADL 出现在 C++98/C++03 中，是对函数表达式中非限定函数名查找的规则，比如查找operator的重载函数。非限定字面意思是没有做任何限制，比如没有指定namespace。通过ADL方式可以依据函数参数的namespace进行函数名查找。
依赖 ADL 有可能会导致语义问题，这也是为什么有的时候需要在函数前面加::，或者一般推荐使用 xxx::func，而不是 using namespace xxx 。因为前者是 qualified name，没有 ADL 的过程。