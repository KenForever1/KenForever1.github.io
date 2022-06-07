### 1.brpc
窃取队列
bthread
协程

### 2.c++并发
局部静态变量
原子可以解决双重检查锁定原有的多线程问题
单例：Scott Meyers and Andrei Alexandrescu 2004年的论文


### 3.inotify
文件监听

### io_uring

### epoll
epoll不能监听文件描述符，因为epoll_ctl函数中需要fd实现poll函数，而ext4文件系统没有实现此poll函数

### systemtap

### ebpf

### 网络
kcp quic http2


### 设计模式

单例：
https://ost.51cto.com/posts/670
https://blog.csdn.net/janeqi1987/article/details/76147312
c++ 单例模式
分为饿汉式，饱汉式

饿汉式，为了避免多线程竞争问题，采用了加锁；
(1)为了缩小加锁范围，采用了double_check机制。但是 double_check不能解决线程问题，instance = new T; 这个语句不是原子的，可以分为三个语句：
申请内存，赋值给instance，调用构造函数。如果一个A线程执行了前两步，B线程执行过了第一个if语句结果是false，拿到的instance 是构造不完整的。
（2）内存屏障，不通用，x86可以，但是有的cpu是弱一致性。
(3) 原子指针变量，加锁，std::atomic(T *); 为了优化效率，采用 acquire, release。
(4) std::call_once, std::once_flag 
(5) 静态局部变量，保证了线程之间的顺序

适配器，装饰者，观察者，外观模式
