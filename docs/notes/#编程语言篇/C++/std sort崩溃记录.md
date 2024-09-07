[https://blog.sina.com.cn/s/blog_532f6e8f01014c7y.html](https://blog.sina.com.cn/s/blog_532f6e8f01014c7y.html)

[https://en.cppreference.com/w/cpp/named_req/Compare](https://en.cppreference.com/w/cpp/named_req/Compare)

```
#include <iostream>
#include <vector>
#include <algorithm>

void func()
{
    std::vector<int> v1;

    for (auto i = 0; i < 18; i++)
    {
        v1.push_back(1);
    }
    std::cout << "v1 addr : " << &v1[0] << std::endl;
    std::cout << "befor sort" << std::endl;
    std::sort(v1.begin(), v1.end(), [](int a, int b) { return a >= b; });
    std::cout << "after sort" << std::endl;
}

int main(int, char **)
{
    func();
}
```

```
v1 addr : 0x55e0ab6d1f30
befor sort
after sort
double free or corruption (out)
fish: Job 1, './hello/…' terminated by signal SIGABRT (Abort)
```

```
(gdb) b 14
(gdb) b 16
(gdb) r
(gdb) x/26wd 0x55555576df20
0x55555576df20: 0       0       145     0
0x55555576df30: 1       1       1       1
0x55555576df40: 1       1       1       1
0x55555576df50: 1       1       1       1
0x55555576df60: 1       1       1       1
0x55555576df70: 1       1       0       0
0x55555576df80: 0       0
(gdb) until 2
(gdb) x/26wd 0x55555576df20
0x55555576df20: 0       0       145     1
0x55555576df30: 1       0       1       1
0x55555576df40: 1       1       1       1
0x55555576df50: 1       1       1       1
0x55555576df60: 1       1       1       1
0x55555576df70: 1       1       0       0
0x55555576df80: 0       0
```
可以看到内存数据被改了，当把set *(0x55555576df2c)=0，就不会报错了。这里与vector的析构函数调用有关。
在项目代码中，如果vector中使一个class或者std::string，那么报错现象可能会是std::bad_alloc，析构std::string出错，或者析构class出错，但本质是因为std::sort的cmp函数定义不正确，导致内存被更改。