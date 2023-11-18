```
nm -DC libpaddle_inference.so | grep paddle::AnalysisConfig::EnableXpu
0000000003632660 T paddle::AnalysisConfig::EnableXpu(int, bool, bool, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, bool, bool)
```

```
undefined reference to `paddle::AnalysisConfig::EnableXpu(int, bool, bool, std::string const&, std::string const&, bool, bool)'
```
以gcc5.1为界限，之前使用的是std::string，之后如果编译时使用用了stdc++11标准，默认是std::__cxx::string。gcc5.1主要更改了std::string copy on write实现和std::list 的size实现，为了兼容旧版本，通过宏定义，add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)控制，为0使用之前的，为1使用新的。
在一个项目中，新ABI和旧ABI不能同时使用。如果报错：

```
undefined func(std::__cxx11:string ....)
```
一般是因为，加载的func函数的这个so动态库使用了旧的ABI编译，而当前项目使用了新的ABI编译。需要重新编译so库。
也可以通过nm命令查看符号，如果显示结果有[abi:cxx11], 就代表so是使用的新ABI编译的。

```
0000000001b24120 T paddle_infer::Predictor::GetInputTensorShape[abi:cxx11]()
```
如果要设置add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)以及set (CMAKE_CXX_STANDARD 11)，注意检测conan的cmake文件conan_toolchain.cmake，这里面可能也对这些选项进行了设置。以及其他地方有没有设置。
参考：
[https://www.cnblogs.com/oloroso/p/11307804.html](https://www.cnblogs.com/oloroso/p/11307804.html)
[https://www.cnblogs.com/stdxxd/p/16491854.html](https://www.cnblogs.com/stdxxd/p/16491854.html)
[https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_dual_abi.html](https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_dual_abi.html)
## gflags
[https://github.com/gflags/gflags/issues/203](https://github.com/gflags/gflags/issues/203)
[https://stackoverflow.com/questions/33394934/converting-std-cxx11string-to-stdstring](https://stackoverflow.com/questions/33394934/converting-std-cxx11string-to-stdstring)
还和clang gcc，以及使用的libc++.so 或者libstdc++.so有关
**如果明明别人提供第三方库也是GCC 5.3 编译的，你本地也是指定的GCC 5.3版本，但是还是有ABI问题，那我猜你是不是本地默认的GCC版本是4.x，或者你链接了libstdc++.a静态库？我就遇到了这个情况。**
**我本地默认GCC版本是4.8，在另一个目录/buildtools/gcc5.3安装了GCC 5.3。在编译时指定了CMAKE_CXX_COMPILER为GCC5.3的对应目录。但是编译时依然会报ABI的问题。被这个问题折磨了一天后，又和第三方对齐了一下cmake版本，然后将cmake升级为最新版本，它居然多了个报警：**

```
CMake Warning at CMakeLists.txt:75 (add_executable):
  Cannot generate a safe linker search path for target MyBinary because
  files in some directories may conflict with libraries in implicit
  directories:
 
    link library [libstdc++.a] in /build_tools/gcc-5.3.21/lib64 may be hidden by files in:
      /usr/lib/gcc/x86_64-redhat-linux/4.8.5
 
  Some of these libraries may not be found correctly.
```
[C++ ABI 问题定位解决_abi问题-CSDN博客](https://blog.csdn.net/qq_35985044/article/details/128602855)
系统上存在多个gcc版本，导致链接错了so文件，也会产生这个问题