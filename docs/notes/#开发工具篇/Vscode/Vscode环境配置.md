## vscode 开发c++存在两种插件模式:
(1) vscode c++: 微软官方
(2) clangd + codeLLDB：更强悍
其他插件：
Cmake用于编译项目。
Remote Explorer：连接远程开发机器，连接docker，连接docker比Clion更友好，可以进入已经存在的Container，而Clion会新启动一个Container。
常用快捷键：
Ctrl+Shift+P：功能面板，输入Cmake，就会跳出插件支持的功能，比如：Build，Run。
Ctrl+R：切换项目
Ctrl+B：打开/关闭目录树
Ctrl-Alt+-: 回到上一个位置
调试：
Ctrl+Shift+D点左边栏的debug按钮，然后应该会自动生成一个launch.json。可以安装一个调试脚本自动生成插件，然后在基础上修改，如修改可执行文件位置，gdb执行文件等。还可以配置远程gdbserver，vscode连接gdbserver进行调试。
参考：[https://zhangjk98.xyz/vscode-c-and-cpp-develop-and-debug-setting/](https://zhangjk98.xyz/vscode-c-and-cpp-develop-and-debug-setting/)。
常见问题：

1. 采用c/c++插件时，如果代码不能跳转到定义位置，一般是没有打开Intelligence，参考：右键代码无跳转，[Vscode右键无代码跳转问题](https://zhuanlan.zhihu.com/p/563906474)。
2. 采用clangd插件时，代码不能跳转定义位置，[vscode 中用clang遇到问题：clang(pp_file_not_found)](https://www.jianshu.com/p/bc78efb11c61)，clangd 插件设置中加入参数： --compile-commands-dir=${workspaceFolder}/build/ 帮你找到compile_commands.json。该文件通过设置Cmake插件生成，默认打开。
3. 中文乱码问题，打开设置中的自动文件格式检测，[关于VS Code 中文显示乱码_vscode 乱码_Sean_gGo的博客-CSDN博客](https://blog.csdn.net/gongxun1994/article/details/80356031)。vim中也有类似的设置，vim中加入： set encoding=utf-8 fileencodings=ucs-bom,utf-8,cp936。

## 快捷键
[https://cloud.tencent.com/developer/article/1885451](https://cloud.tencent.com/developer/article/1885451)
Ctrl-d：一个一个选择匹配到的，多光标选择修改
Ctrl-u：Ctrl-d对应的逆操作
Ctrl-g：跳转到指定行号位置
Ctrl-w：关闭Tab
Ctrl-k-w：关闭所有Tab
Ctrl-Shit-a：注释选中的一段代码
Ctrl-\：分割窗口
Ctrl-1：切换上一个窗口
Ctrl-2：切换下一个窗口
Ctrl-Tab：切换Tab

## Vscode结合Vim
[https://www.barbarianmeetscoding.com/boost-your-coding-fu-with-vscode-and-vim](https://www.barbarianmeetscoding.com/boost-your-coding-fu-with-vscode-and-vim)
```shell
 "vim.leader": "<Space>",
    "vim.easymotion": true,
    "vim.normalModeKeyBindingsNonRecursive": [

        {
            "before" : ["<leader>","a"],
            "commands" : [
              "workbench.view.explorer"
            ]
        },
        {
          "before": ["J"],
          "after": ["5", "j"]
        },
        {
          "before": ["K"],
          "after": ["5", "k"]
        },
        {
            "before": ["<Leader>", "j"],
            "after": ["J"]
        },
        {
            "before": ["<C-h>"],
            "after": ["<C-w>", "h"]
        },
        {
            "before": ["<C-j>"],
            "after": ["<C-w>", "j"]
        },
        {
          "before": ["<C-k>"],
          "after": ["<C-w>", "k"]
        },
        {
          "before": ["<C-l>"],
          "after": ["<C-w>", "l"]
        },
        {
          "before": ["<Leader>", "t", "t"],
          "commands": [":tabnew"]
        },
        {
          "before": ["<Leader>", "t", "n"],
          "commands": [":tabnext"]
        },
        {
          "before": ["<Leader>", "t", "p"],
          "commands": [":tabprev"]
        },
        {
          "before": ["<Leader>", "t", "o"],
          "commands": [":tabo"]
        },
        {
           "before": ["<Leader>", "p"],
           "commands": [
               "workbench.action.showCommands",
           ]
        },
        {
           "before": ["<Leader>", "t"],
            "commands": [
                "workbench.action.gotoSymbol",
            ]
        }

    ],
    "vim.handleKeys": {
        "<C-f>": false
    },
```

## Vscode如何gdb
调试普通可执行程序
调试so
Python加载so
gdb版本也可能影响

启动server：
/opt/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf# find . -name gdbserver
./arm-linux-gnueabihf/libc/usr/bin/gdbserver
cp gdbserver 到运行主机上
${current_dir}/../gdbserver :9091 ${current_dir}/bin/main arg1 arg2 
在开发机上：
/opt/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf/bin#
arm-linux-gnueabihf-gdb

```shell
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "OK_Gdb_Debugger",
            "type": "cppdbg",
            "request": "launch",
            //要调试的程序
            "program": "${workspaceFolder}/build/bin/human_detect",
            "args": [],
            "stopAtEntry": false,
            "cwd": "${workspaceFolder}",
            "environment": [],
            "externalConsole": false,
            "MIMode": "gdb",
            //gdb路径
            "miDebuggerPath": "/opt/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf/bin/arm-linux-gnueabihf-gdb",
            //远程调试，服务器地址，端口号。
            "miDebuggerServerAddress": "192.168.1.188:9091",
            "additionalSOLibSearchPath": "${workspaceFolder}/build/lib/"
            // "setupCommands": [
            //     {
            //         "description": "Enable pretty-printing for gdb",
            //         "text": "-enable-pretty-printing",
            //         "ignoreFailures": true
            //     }
            // ]
        }
    ]
}
```
错误：
不能断点so，so设置了-g 没有-O优化，cmake设置了debug编译，不能断点到so的cpp文件，报错提示：
Module containing this breakpoint has not yet loaded or the breakpoint address could not be obtained.
解决方式：
将开发机的build文件下的lib目录添加到additionalSOLibSearchPath：
"additionalSOLibSearchPath": "${workspaceFolder}/build/lib/"

args 和 env设置
```cpp
"args": [
            "xxxx1",
            "yyyy2",
            "zzzz3",
            ],
"cwd": "${workspaceFolder}",
"environment": [
    // {
    //     "name": "SHELL",
    //     "value": "/bin/bash"
    // },
    {
        "name": "LD_LIBRARY_PATH",
        "value": "${LD_LIBRARY_PATH}:${workspaceFolder}/xxxx"
    }
],
```
## Vscode配置交叉编译工具
[https://edgeai-lab.github.io/notebook/Embedded%20System/linux/compiler/cmake/vscode_camke_cross_compile/](https://edgeai-lab.github.io/notebook/Embedded%20System/linux/compiler/cmake/vscode_camke_cross_compile/)

```
vim .local/share/CMakeTools/cmake-tools-kits.json
```

```
{
    "name": "arm_corss_compiler",
    "toolchainFile": "/home/fhc/myWorkspace/cmake_project/cmake_cross_complie_base/toolchain.cmake"
}
```

```
### toolchain.cmake ###
# this is required
SET(CMAKE_SYSTEM_NAME Linux)

# specify the cross compiler
SET(CMAKE_C_COMPILER   /home/fhc/linux_driver/gcc-3.4.5-glibc-2.3.6/bin/arm-linux-gcc)
SET(CMAKE_CXX_COMPILER /home/fhc/linux_driver/gcc-3.4.5-glibc-2.3.6/bin/arm-linux-g++)

# where is the target environment
SET(CMAKE_FIND_ROOT_PATH  /home/fhc/linux_driver/gcc-3.4.5-glibc-2.3.6)

# search for programs in the build host directories (not necessary)
SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

# for libraries and headers in the target directories
SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
```

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