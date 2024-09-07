python debug有两种方法，

- pdb
- debugpy （可以配合vscode使用）
## pdb使用
使用方法和gdb类似，常用的操作命令也和gdb也几乎一样。
方法1，命令行启动：

```
python3 -m pdb ./test.py
```
方法2，程序中嵌入代码，在想要断点的地方，添加：

```
import pdb
pdb.set_trace()
```
## debugpy使用
### 直接使用debugpy
直接使用，程序中存在循环

```
python3 -m debugpy --listen :5678  ./test.py
```
如果程序中没有loop循环，程序会直接执行结束。
### vscode client连接使用
vscode launch.json配置
安装python插件

```
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            // "request": "launch",
            // "program": "${file}",
            // "python": "/usr/local/python3.9.2/bin/python3",
            // "console": "integratedTerminal",
            // "justMyCode": false,
            // "stopOnEntry": true,
            "request": "attach",
            "connect": {
                "host": "localhost",
                "port": 5678
              }
        }
    ]
}
```
启动调试py文件，
安装debugpy

```
python3 -m pip install debugpy
```

```
python3 -m debugpy --listen :5678 --wait-for-client ./onnx_infer.py
```
### vscode作为server反向调试
launch.json配置：

```
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: 远程调试",
            "type": "python",
            "request": "attach",
            "listen": {
                "host": "0.0.0.0",
                "port": 5678
            },
            "pathMappings": [
                {
                    "localRoot": "${workspaceFolder}", 
                    "remoteRoot": "."
                }
            ]
        }
    ]
}
```
在调试程序python程序的最前面，加入如下代码：

```
import debugpy
debugpy.connect(('xxx.xxx.xxx.xxx', 5678)) # 更改成vscode开发机的ip地址
```
参考：
[10分钟教程掌握Python调试器pdb](https://zhuanlan.zhihu.com/p/37294138)
[https://code.visualstudio.com/docs/python/debugging](https://code.visualstudio.com/docs/python/debugging)
[Debugpy——如何使用VSCode调试无法直接执行的Python程序](https://zhuanlan.zhihu.com/p/560405414)