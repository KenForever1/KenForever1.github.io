## gdb更好的显示c++ STL数据结构的值
### 开启pretty-print功能
在使用vscode调试时，不能像clion一样直接查看STL容器中的内容，比如String，只能看到指针和长度等信息，不能直接看到字符串。
[【工具】——VSCODE调试C++时无法显示Vector，map等容器的值_农夫山泉2号的博客-CSDN博客](https://blog.csdn.net/u011622208/article/details/132083666)

```
"setupCommands": [
    {
        "description": "Test",
        "text": "python import sys;sys.path.insert(0, '/usr/share/gcc-8/python');from libstdcxx.v6.printers import register_libstdcxx_printers;register_libstdcxx_printers(None)",
        "ignoreFailures": false
    },
    {
        "description": "Enable pretty-printing for gdb",
        "text": "-enable-pretty-printing",
        "ignoreFailures": true
    }
]
```
但是可能会出现问题：Python scripting is not supported in this copy of GDB
解决：
1.查看当前gdb是否支持pretty-print功能，两种方法，方法一，在gdb终端：

```
（gdb）info pretty-print
```
如果支持，会看到以下类似的输出：

```
global pretty-printers:
  builtin
    mpx_bound128
```
方法二：
如果是交叉编译的gdb，请将$(which gdb)换成实际的gdb路径。

```
readelf -d $(which gdb) | grep python
```
如果没有打印类似pythonxxx的内容，说明不支持，需要重新编译。

1. 重新编译

```
sudo apt install python3 python3-dev
```
编译脚本如下：

```
#!/bin/bash

cd gdb-8.1.1/

./configure --program-prefix=`aarch64-mix210-linux-` \
    --target=aarch64-mix210-linux \
	--prefix=`pwd`/out \ 
    --with-python="/usr/bin/python3"

# 增加了--with-python，添加python支持
make -j2 
make install
```

### 使用gdb脚本调试自定义print STL命令
如果不在vscode中调试呢，如何更好的打印stl容器中的内容呢？
首先，引入gdbinit脚本，可以定义.gdbinit文件，默认启动时会去$home目录中找，如果存在这个文件就加载使用。也可以通过gdb -x .gdbinit文件指定。
一些重复性的调试内容或者显示的设定，以及command的扩展，都可以通过.gdbinit文件来设置。command命令支持定义函数和函数说明（description）。
打印stl容器中的内容，就可以通过gdbinit脚本实现。比如打印stl容器的gdb脚本，[http://www.yolinux.com/TUTORIALS/src/dbinit_stl_views-1.03.txt](http://www.yolinux.com/TUTORIALS/src/dbinit_stl_views-1.03.txt)。

```
wget http://www.yolinux.com/TUTORIALS/src/dbinit_stl_views-1.03.txt -o .gdbinit
```
然后将其放入$home目录下，在这个脚本中扩充了pvector、pstring、pqueue、pmap等命令，比如存在一个vector变量file_name_list。

```
(gdb) ptype file_name_list
type = std::vector<std::string>
(gdb) pvector file_name_list # 就会打印vector内容
```
在vscode中执行需要加-exec命令。

```
(gdb) -exec ptype file_name_list
type = std::vector<std::string>
(gdb) -exec pvector file_name_list # 就会打印vector内容
```