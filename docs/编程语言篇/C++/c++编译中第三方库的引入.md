在c++开发过程中，避免不了引入第三方库，比如视频读取、图片编辑需要引入opencv库，opencv的视频处理需要ffmpeg（也可以选择gstreamer等引擎），而json处理，需要引入jsoncpp等json库。
当程序需要编译打包成可执行文件以及动态链接库，然后到另一台机器上运行时，可能会出现一些问题。比如：

1. 动态链接库版本不匹配，在开发机上使用了jsoncpp1.9.5版本，但是在部署机上使用了jsoncpp1.7.4版本，那么程序在执行时，动态链接就会报错找不到libjsoncpp.so.xx的文件。opencv等库同样也存在这样的问题，比如opencv3和opencv4版本的不同。
2. 当遇到版本不匹配时，往往选择卸载当前版本，安装匹配的版本。但是，如果没有使用正确的卸载方法，比如只是去/usr/local/lib中删除了.so和.a文件，/usr/local/include或者其他目录中原来的头文件还存在。然后安装了匹配版本在某个目录，编译程序时，如果首先找到的还是原来的头文件，那么就会报错：未定义的xxx（undefined），这个时候不是因为没有找到so（是因为找错了头文件）。解决方式:删除之前版本的头文件。
3. 在不同的开发平台上，安装第三方库的方式不一样。以ubuntu开发为例，在ubuntu20.04以及以后版本中往往可以直接采用apt方式安装（比如:apt install opencv），而在ubuntu18.04以及以下版本中，往往需要源码编译安装（sudo apt install python3-opencv的方式安装的功能不全）。但是使用源码进行编译安装编译耗时且麻烦，多个版本切换不方便。这个时候需要使用包管理工具，c++平台的包管理工具比如conan。
4. 在使用opencv时，如果出现cv::videocapture时，发现open函数不能打开视频，可以考虑ffmpeg安装错误，或者开发机上存在多个ffmpeg版本，这个时候首先需要解决ffmpeg的版本冲突。查看机器上的ffmpeg情况，最简单的暂时将冲突的ffmpeg目录改一个名字。

```
whereis ffmpeg
which ffmpeg
```
前面提到了使用包管理工具，在windows上可以使用vcpkg，在linux平台上可以使用conan，类似python的pip包管理工具。我只使用过conan，因此对conan做一下使用过程中的分享。

- conan包管理工具

conan可以在项目中很方便的切换多个版本第三方库，而所有的第三方库都只会存在一次在本地的一个统一目录下，比如.conan目录中。而且使用connan search可以搜索到有哪些包可以使用，避免了源码编译耗费太多时间的问题。另一个好处是conan可以很方便的和cmake等编译构建系统结合，特别是conan2.0不需要侵入CmakeLists.txt文件，只需要单独增加一个conanfile.txt或者conanfile.py文件。

```
conan search "opencv"
结果：
conancenter
  opencv
    opencv/2.4.13.7
    opencv/3.4.12
    opencv/3.4.17
    opencv/4.1.2
    opencv/4.5.0
    opencv/4.5.1
    opencv/4.5.2
    opencv/4.5.3
    opencv/4.5.5
```
conanfile.txt文件示例：

```
[requires]
jsoncpp/1.9.5
opencv/3.4.12

[generators]
CMakeDeps
CMakeToolchain
```
在conan2.0中不需要对CmakeLists.txt做任何更改，首先根据conanfile下载依赖:

```
conan install . --output-folder=build --build=missing
```
然后再编译cmake时，通过-D选项使用build文件下conan生成的依赖文件:

```
$ cd build
$ cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release
$ cmake --build .
```
参考：[https://docs.conan.io/2/tutorial/consuming_packages/build_simple_cmake_project.html](https://docs.conan.io/2/tutorial/consuming_packages/build_simple_cmake_project.html)
当然，conan还有很多高级的功能，比如使用conanfile.py可以自定义编译的逻辑，比如区分不同的平台处理不同的编译逻辑等。在不同程序语言中，有很多类似的工具，比如类似rust中cargo toml文件定义三方库的使用以及build.rs定义编译过程。

- CPM：Cmake依赖管理

[https://github.com/cpm-cmake/CPM.cmake](https://github.com/cpm-cmake/CPM.cmake)
```
cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

# create project
project(MyProject)

# add executable
add_executable(main main.cpp)

# add dependencies
include(cmake/CPM.cmake)

CPMAddPackage("gh:fmtlib/fmt#7.1.3")
CPMAddPackage("gh:nlohmann/json@3.10.5")
CPMAddPackage("gh:catchorg/Catch2@3.2.1")

# link dependencies
target_link_libraries(main fmt::fmt nlohmann_json::nlohmann_json Catch2::Catch2WithMain)
```