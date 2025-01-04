## 1 安装ffmpeg

### 1.1 下载ffmpeg

下班ffmpeg小于4.4版本，这里下载ffmpeg 4.1版本。

### 1.2 编译ffmpeg
编译的bin、so等文件，就放在ffmpeg源码根目录。
```bash
cd ffmpeg

./configure --enable-shared --disable-x86asm --libdir=. --prefix=. --disable-static

make -j8

make install

mkdir ffmpeg-4.1

cd ffmpeg-4.1 && mkdir lib && cd ..

mv lib*.so* ffmpeg-4.1/lib

mv bin ffmpeg-4.1/

mv include ffmpeg-4.1/

mv share ffmpeg-4.1/

mv pkgconfig ffmpeg-4.1/lib/

```

```bash
$ ls ffmpeg-4.1

bin  ffmpeg-config.cmake  FfmpegConfig.cmake  include  lib  share
```

### 1.3 编写ffmpeg-config.cmake文件

编写ffmpeg-config.cmake文件是为了让opencv编译时，正确的找到这个ffmpeg的位置。FfmpegConfig.cmake文件的内容和ffmpeg-config.cmake文件相同。命名规则安装cmake find_package的命令方式。

```bash
set(ffmpeg_path "${CMAKE_CURRENT_LIST_DIR}")

message("ffmpeg_path: ${ffmpeg_path}")

set(FFMPEG_EXEC_DIR "${ffmpeg_path}/bin")

set(FFMPEG_LIBDIR "${ffmpeg_path}/lib")

set(FFMPEG_INCLUDE_DIRS "${ffmpeg_path}/include")

# library names

set(FFMPEG_LIBRARIES

    ${FFMPEG_LIBDIR}/libavformat.so

    ${FFMPEG_LIBDIR}/libavdevice.so

    ${FFMPEG_LIBDIR}/libavcodec.so

    ${FFMPEG_LIBDIR}/libavutil.so

    ${FFMPEG_LIBDIR}/libswscale.so

    ${FFMPEG_LIBDIR}/libswresample.so

    ${FFMPEG_LIBDIR}/libavfilter.so

)

# found status

set(FFMPEG_libavformat_FOUND TRUE)

set(FFMPEG_libavdevice_FOUND TRUE)

set(FFMPEG_libavcodec_FOUND TRUE)

set(FFMPEG_libavutil_FOUND TRUE)

set(FFMPEG_libswscale_FOUND TRUE)

set(FFMPEG_libswresample_FOUND TRUE)

set(FFMPEG_libavfilter_FOUND TRUE)

# library versions, 注意这几个变量，一定要设置为全局CACHE变量

set(FFMPEG_libavutil_VERSION 56.31.100 CACHE INTERNAL "FFMPEG_libavutil_VERSION") # info

set(FFMPEG_libavcodec_VERSION 58.54.100 CACHE INTERNAL "FFMPEG_libavcodec_VERSION") # info

set(FFMPEG_libavformat_VERSION 58.29.100 CACHE INTERNAL "FFMPEG_libavformat_VERSION") # info

set(FFMPEG_libavdevice_VERSION 58.8.100 CACHE INTERNAL "FFMPEG_libavdevice_VERSION") # info

set(FFMPEG_libavfilter_VERSION 7.57.100 CACHE INTERNAL "FFMPEG_libavfilter_VERSION") # info

set(FFMPEG_libswscale_VERSION 5.5.100 CACHE INTERNAL "FFMPEG_libswscale_VERSION") # info

set(FFMPEG_libswresample_VERSION 3.5.100 CACHE INTERNAL "FFMPEG_libswresample_VERSION") # info

set(FFMPEG_FOUND TRUE)

set(FFMPEG_LIBS ${FFMPEG_LIBRARIES})

status("    #################################### FFMPEG:"       FFMPEG_FOUND         THEN "YES (find_package)"                       ELSE "NO (find_package)")

status("      avcodec:"      FFMPEG_libavcodec_VERSION    THEN "YES (${FFMPEG_libavcodec_VERSION})"    ELSE NO)

status("      avformat:"     FFMPEG_libavformat_VERSION   THEN "YES (${FFMPEG_libavformat_VERSION})"   ELSE NO)

status("      avutil:"       FFMPEG_libavutil_VERSION     THEN "YES (${FFMPEG_libavutil_VERSION})"     ELSE NO)

status("      swscale:"      FFMPEG_libswscale_VERSION    THEN "YES (${FFMPEG_libswscale_VERSION})"    ELSE NO)

status("      avresample:"   FFMPEG_libavresample_VERSION THEN "YES (${FFMPEG_libavresample_VERSION})" ELSE NO)
```

## 2 安装Opencv

### 2.1 编译opencv

编写build_with_ffmpeg4-1.sh文件，编译opencv

```bash
#!/bin/bash

OPENCV_INSTALL_DIR=`pwd`/opencv_install

mkdir -p ${OPENCV_INSTALL_DIR}

FFMPEG_FIND_DIR=`pwd`/compile_opencv/ffmpeg-4.1/ffmpeg-4.1/

BUILD_DIR=build_ffmpeg4.1

mkdir ${BUILD_DIR}

cd ${BUILD_DIR} || return

pwd

cmake .. \

  -D CMAKE_BUILD_TYPE=Release \

  -D CMAKE_INSTALL_PREFIX=${OPENCV_INSTALL_DIR} \

  -D BUILD_TESTS=OFF \

  -D BUILD_PERF_TESTS=OFF \

  -D WITH_CUDA=OFF \

  -D WITH_VTK=OFF \

  -D WITH_MATLAB=OFF \

  -D BUILD_DOCS=OFF \

  -D BUILD_opencv_python3=OFF \

  -D BUILD_opencv_python2=OFF \

  -D WITH_IPP=OFF \

  -D BUILD_SHARED_LIBS=ON \

  -D BUILD_opencv_apps=OFF \

  -D WITH_CUDA=OFF \

  -D WITH_OPENCL=OFF \

  -D WITH_VTK=OFF \

  -D WITH_MATLAB=OFF \

  -D BUILD_DOCS=OFF \

  -D BUILD_opencv_python3=OFF \

  -D BUILD_opencv_python2=OFF \

  -D BUILD_JAVA=OFF \

  -D BUILD_FAT_JAVA_LIB=OFF \

  -D WITH_PROTOBUF=OFF \

  -D WITH_QUIRC=OFF \

  -D WITH_FFMPEG=ON \

  -D OPENCV_GENERATE_PKGCONFIG=ON \

  -D OPENCV_FFMPEG_USE_FIND_PACKAGE=ON \

  -D OPENCV_FFMPEG_SKIP_BUILD_CHECK=ON \

  -D FFMPEG_DIR=${FFMPEG_FIND_DIR}

make -j8

make install
```

编译opencv，编译好的文件会安装到当前目录下的opencv_install。

```bash
bash ./build_with_ffmpeg4.2.2_osx.sh
```

在编译时可以先把make和make install注释了，查看opencv编译时，FFmpeg是否是YES状态。在运行时，也可以通过代码检测FFmpeg状态。

```cpp
std::cout << cv::getBuildInformation();
```

## 3 打包ffmpeg和opencv作为第三方库

现在将编译好的ffmpeg和opencv放在同一个目录下，作为第三方库给其他主机使用。

### 3.1 创建3rd_party

```bash
$ mkdir 3rd_party

$ mv ffmpeg-4.1 3rd_party

$ mv opencv_install/ 3rd_party/opencv4.5.5

$ ls

ffmpeg-4.1  opencv-4.5.5
```

### 3.2 使用ffmpeg和opencv

现在假设一个项目名叫ProjA，那么将3rd_party复制到ProjA根目录，然后创建一个build.sh。为了避免找到主机上的其他ffmpeg，需要设置LD_LIBRARY_PATH，将${FFMPEG_LIB_DIR}加入进去。

```bash
#!/bin/bash

current_dir=$(cd `dirname $0`; pwd)

export OPENCV_LIB_DIR=`pwd`/3rd_party/opencv-4.5.5/lib64/cmake/opencv4;

FFMPEG_LIB_DIR=`pwd`/3rd_party/ffmpeg-4.1/lib;

GCC_VERSION="8.2"

export CC=/opt/compiler/gcc-${GCC_VERSION}/bin/gcc

export CXX=/opt/compiler/gcc-${GCC_VERSION}/bin/g++

export PATH=/opt/compiler/gcc-${GCC_VERSION}/bin${PATH:+:${PATH}}

export LD_LIBRARY_PATH=/opt/compiler/gcc-${GCC_VERSION}/lib:${FFMPEG_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export PKG_CONFIG_PATH=/opt/compiler/gcc-${GCC_VERSION}/lib/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}

build_folder=${current_dir}/build

clean=$1

if [ -d ${build_folder} ]; then

    if [ "${clean}" == "clean" ]; then

        rm -rf ${build_folder}/*

    fi

else

    mkdir ${build_folder}

fi

cd ${build_folder} && cmake ..

make -j4

make install

exit $?
```

参考：

[[推理部署]🤓opencv+ffmpeg编译打包全解指南](https://zhuanlan.zhihu.com/p/472115312?utm_id=0)