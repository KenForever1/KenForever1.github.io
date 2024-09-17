## 调试mlir的vscode环境配置
除了前文中在运行时指定调试打印参数，更加友好的方式可能就是gdb调试了。本节讲解一下vscode调试环境配置。
首先vscode打开llvm_project项目，采用debug模式编译项目:
```bash
cd build
cmake -G Ninja ../llvm    -DLLVM_ENABLE_PROJECTS=mlir    -DLLVM_BUILD_EXAMPLES=ON    -DLLVM_TARGETS_TO_BUILD="X86"    -DCMAKE_BUILD_TYPE=Debug    -DLLVM_ENABLE_ASSERTIONS=ON  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON

cmake --build . --target toyc-ch3
```

创建launch.json，添加以下内容：
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "OK_Gdb_Debugger",
            "type": "cppdbg",
            "request": "launch",
            //要调试的程序
            "program": "${workspaceFolder}/build/bin/toyc-ch3",
            // "args": ["${workspaceFolder}/mlir/test/Examples/Toy/Ch2/codegen.toy","-emit=mlir", "-mlir-print-debuginfo"],
            "args": [
                "${workspaceFolder}/mlir/test/Examples/Toy/Ch3/transpose_transpose.toy", "-emit=mlir", "-opt"
            ],
            "stopAtEntry": false,
            "cwd": "${workspaceFolder}",
            "environment": [],
            "externalConsole": false,
            "MIMode": "gdb",
            "miDebuggerPath": "/usr/bin/gdb",

            //gdb路径
            // "miDebuggerPath": "/opt/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf/bin/arm-linux-gnueabihf-gdb",
            // //远程调试，服务器地址，端口号。
            // "miDebuggerServerAddress": "192.168.1.188:9091",
            // "additionalSOLibSearchPath": "${workspaceFolder}/build/lib/"
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
核心是program和参数args选项，注释的内容如果是远程调试可以根据此来做一些修改。Program项指定了toyc-ch3运行bin文件，args是传递给它的参数。
然后可以debug源码了。