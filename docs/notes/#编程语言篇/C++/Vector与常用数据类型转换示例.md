### Convert Functions

一个各种常用数据类型和Vector互相转换的例子：
```c++
#include <vector>
#include <memory>
#include <cstring>
template<typename T>
T VecToType(const std::vector<unsigned char> &vec) {
    if (sizeof(T) != vec.size()) {
        throw std::runtime_error("vec2type failed: sizeof(T) != vec.size()");
    }
    T t;
    std::memcpy(&t, vec.data(), sizeof(T));
    return std::move(t);
}

template<typename T>
std::vector<unsigned char> TypeToVec(const T &data) {
    std::vector<unsigned char> ret;
    ret.resize(sizeof(data));
    memcpy(ret.data(), &data, sizeof(data));
    return ret;
}

template<class T>
static inline std::vector<unsigned char> toVector(const T &data) {
    std::vector<unsigned char> ret;
    ret.resize(sizeof(T));
    memcpy(ret.data(), &data, sizeof(T));
    return ret;
}

template<>
inline std::vector<unsigned char> toVector(const std::string &data) {
    return {data.begin(), data.end()};
}


template<typename T>
static inline T toType( std::vector<unsigned char> data) {
    return *reinterpret_cast<T *>(data.data());
}

template<>
inline std::string toType(std::vector<unsigned char> data) {
    return {data.begin(), data.end()};
}

```
