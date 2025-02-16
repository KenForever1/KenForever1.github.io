---
comments: true
---
C++ 条件编译

1. 首先，在CMakeLists.txt中添加以下代码，以定义新的CFLAG：
    

```bash
option(NEW_CFLAG_ENABLE "Enable new CFLAG" OFF)  
if(NEW_CFLAG_ENABLE)  
  add_compile_options(-DNEW_CFLAG)  
endif()  
```

2. 接着，在代码中使用条件编译（#ifdef）来检查是否定义了该CFLAG：
    

```bash
#ifdef NEW_CFLAG  
  // 你的代码  
#endif  

```
3. 在使用CMake编译代码时，可以选择启用或禁用新的CFLAG，只需设置NEW_CFLAG_ENABLE的值即可：
    

```bash
cmake -B build -S . -DNEW_CFLAG_ENABLE=ON

cmake --build build
```

这样，当启用了新的CFLAG时，编译器将会定义NEW_CFLAG宏，你的代码中的条件编译就会成立。