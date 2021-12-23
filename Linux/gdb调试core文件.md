我在调试程序时，遇到了一种情况：在debug方式下执行程序没有问题，但是在运行模式下出现了段错误。使用下面的调试方式，使用gdb调试程序崩溃后生成core文件，成功找到报错位置。
### 程序生成core文件

```
# 设置（临时在当前terminal中设置）
ulimit -c unlimited

# 查看
ulimit -a
```

设置core文件生成目录

```
sysctl -w kernel.core_pattern=/var/crash/core.%u.%e.%p
```
此命令将coredump文件缺省会保存至/var/crash目录下，文件名称格式为core.%u.%e.%p

### gdb调试

将core文件拷贝到程序执行目录

```
gdb ./exec_file_name core_file_name
```

```
bt
```

```
frame number
```