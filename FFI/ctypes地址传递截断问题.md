### ctypes地址截断问题

在使用ctypes开发python调用c程序的接口函数时，需要在python程序和c程序之间传递地址，不管是堆上的地址还是栈上的地址，如果没有设置函数的restype和argtypes时，很可能会引起Segment Fault错误。

这是因为在默认情况下，地址可能会被截断，导致访问地址时，访问到错误的地址，出现Segment Fault。

### 解决方法

在每个需要传递地址的地方，设置restype和argstype：

如：

```
myfunc.restype = ctypes.c_uint64 
myfunc.argtypes = [ctypes.c_uint64]
```
在设置argtypes时，如果有多个参数可以只设置部分参数的argtype。

也有说可以将restype设置为ctypes.c_void_p类型。

### 建议

我最初也没有每个函数都去设置restype和argtypes，发现在已有的测试代码都成功后，就觉得没有问题了，但是后面遇到了Segment Fault。这就建议我们在使用ctypes在python和c之间传递地址时，一定要设置restype和argtypes。

两种因为没有设置restype和argtypes有趣的报错经历：
+ 相同的一个函数接口，设置restype时，我在使用函数直接调用时，没有出现Segment Fault，但是在python传递的一个给c的回调函数中调用了这个函数，出现了Segment Fault。

+ 在普通Python环境和virtual env环境中都没有出现错误，但是在conda环境中直接Segment Fault。

### 使用gdb debug

在使用python ctypes调用c函数的时候，如果出现了Segment Fault等错误时，很难调试，特别是在python传递回调函数给c使用时。

两种调试的方法：

+ 在回调函数的情况，我只使用了打印的办法，暂时没有发现更好的办法。

在c++这边可以使用如下方法，打印出16进制地址：
```
std::cout << std::hex << ptrVar << std::endl;
```

在python这边：
```
print(hex(ptrVar))
```
通过对比地址是否一样，来判断是否出现地址截断问题。

+ 使用gdb
使用gdb启动测试程序，
```
gdb -args python test_file.py
```
在gdb中执行run，执行程序，
```
run
```
当函数报错时，打印backtrace，
```
backtrace
// 或者使用缩写 bt
```
使用frame，展示栈帧
```
frame number
```

然后通过打印信息，处理解决报错问题。

注意: 在使用gdb调试的时候，c程序这边如果时so，则要使用debug模式进行编译，才能打印出详细的报错位置。