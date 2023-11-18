## Python动态参数传递
在python中的动态参数传递使用*args 和 **kargs。其中args被解析为元组，可以包括不同类型的参数，比如（12, ‘s’）。kargs被解析为字典，{"k1": "v1", "k2":233}。
在编程中遇到一个问题：
```python
def func(*args, **kargs):
    pass

func(name = "tom", age = 18)
```
上面的方式可以对func进行调用，但是如何使得name 和 age成为参数呢？
解决方法：通过定义字典d，然后func(**d)解决。
```python
d = {
    "name" = "tom",
	"age" = 18
}
func(**d)
```
这样name和age就可以以变量的形式动态传递了。例如：
```python

def func(*args, **kargs):
    print(args)

    print(kargs)
    return 0
func(1, 2, 3, a= "bbb")
func(a = "aaa", b = 1)

d1 = {
    "hello": "world",
    "abc": "def",
    "kkk": 124
}

func(**d1)


l1 = (1,2, "adfaf")
func(*l1)

a = "image"
a_v = 1234
d2 = {
    a:a_v
}

func(**d2)
```
```bash
(1, 2, 3)
{'a': 'bbb'}
()
{'a': 'aaa', 'b': 1}
()
{'hello': 'world', 'abc': 'def', 'kkk': 124}
(1, 2, 'adfaf')
{}
()
{'image': 1234}
```
## python代码移植成c++
在深度学习等领域，首先会通过python开发验证，但是在部署时，为了提高程序执行效率，需要将python实现的算法处理代码转换成为c++代码，
通过调研有几种方式，比如：

- 使用一些自动转换工具
- 在c++里面使用，Eigen库、NumCpp库等去编写python里Numpy处理逻辑
- 直接构造Vector用一维表示python Numpy多维数组，然后编写操作逻辑

下面对几种方式的优缺点进行总结：
 	自动转换工具一般可以转换典型的Python代码，但是不支持所有的。
第三方库的使用分两种情况，如果python代码逻辑简单，引入第三方库，会增加成本，比如学习成本、编译成本，而且三方库如果不熟悉产生bug，难以解决。如果python代码逻辑复杂，比如使用了很多线性代数等函数计算，这时可以极大提高效率。
在使用第三方库时，还需要考虑部署环境的C++版本要求，即编译器版本gcc、clang的版本支持情况，比如NumCpp是需要 C++17版本以上要求的，那么在一些部署环境中可能就不能使用。当然你可以选择支持c++版本更高的交叉编译环境，提高开发体验。
通过Vector以及其它C++原生数据结构编写，具有更大的可控性。在cpu眼里，多维数组都是被当作一维数组来处理，这些库只是在上层进行了封装。可以参考PaddleDetection中deploy的例子，python和C++程序如何转换的。