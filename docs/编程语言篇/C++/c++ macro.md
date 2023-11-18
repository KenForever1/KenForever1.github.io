


```c++
#include <iostream>

#define GET_NTH_ARG(_1, _2, _3, _4, _5, _6, _7, _8, n, ...) n
//#define GET_ARG_COUNT(...) GET_NTH_ARG(__VA_ARGS__, 8, 7, 6, 5, 4, 3, 2, 1)
#define GET_ARG_COUNT(args...) GET_NTH_ARG(args, 8, 7, 6, 5, 4, 3, 2, 1)

#define REPEAT_1(f, i, arg) f(i, arg)
#define REPEAT_2(f, i, arg, ...) f(i, arg) REPEAT_1(f, i+1, __VA_ARGS__)
#define REPEAT_3(f, i, arg, ...) f(i, arg) REPEAT_2(f, i+1, __VA_ARGS__)
#define REPEAT_4(f, i, arg, ...) f(i, arg) REPEAT_3(f, i+1, __VA_ARGS__)
#define REPEAT_5(f, i, arg, ...) f(i, arg) REPEAT_4(f, i+1, __VA_ARGS__)
#define REPEAT_6(f, i, arg, ...) f(i, arg) REPEAT_5(f, i+1, __VA_ARGS__)
#define REPEAT_7(f, i, arg, ...) f(i, arg) REPEAT_6(f, i+1, __VA_ARGS__)
#define REPEAT_8(f, i, arg, ...) f(i, arg) REPEAT_7(f, i+1, __VA_ARGS__)

#define hello_f(_i, arg) puts(arg);


//struct Point {
//    template<typename, size_t>
//    struct FIELD;
//
//    static constexpr size_t _filed_count_ = 2;
//    double x;
//
//    template<typename T>
//    struct FIELD<T, 0> {
//        T &obj;
//
//        auto value() -> decltype(auto) {
//            return (obj.x);
//        }
//
//        static constexpr const char *name() {
//            return "x";
//        }
//    };
//
//    double y;
//    template<typename T>
//    struct FIELD<T, 1> {
//        T &obj;
//
//        auto value() -> decltype(auto) {
//            return (obj.y);
//        }
//
//        static constexpr const char *name() {
//            return "y";
//        }
//    };
//};


#define PARE(...) __VA_ARGS__
#define PAIR(x) PARE x

#define EAT(...)
#define STRIP(x) EAT x

#define PASTE(x, y) CONCATE(x, y)
#define CONCATE(x, y) x##y

#define STRING(x) "x"

#define FIELD_EACH(i, arg) \
    PAIR(arg);              \
    template <typename T>  \
    struct FIELD<T, i> {   \
        T &obj;            \
        auto value() -> decltype(auto) {return (obj.STRIP(arg));} \
        static constexpr const char * name() {return STRING(STRIP(arg));} \
    };

#define DEFINE_SCHEMA(st, ...) \
    struct st {                \
        template <typename, size_t> struct FIELD; \
        static constexpr size_t _filed_count_ = GET_ARG_COUNT(__VA_ARGS__); \
        PASTE(REPEAT_, GET_ARG_COUNT(__VA_ARGS__)) (FIELD_EACH, 0, __VA_ARGS__)\
    };

DEFINE_SCHEMA(Point, (double) x, (double) y)

int main() {
//    std::cout << GET_ARG_COUNT(a, b, c) << std::endl;
//    REPEAT_3(hello_f, 0, "hello", "wold", "!")

    return 0;
}

```