## ebpf udst初探

ebpf udst, 静态跟踪点(tracepoint)，在用户空间也被称为USDT(用户静态定义跟踪)探针，是应用程序感兴趣的特定位置，可以在其中挂载跟踪器来检查数据和代码执行情况。

用户程序添加udst，采用libstapsdt。
```
https://github.com/sthima/libstapsdt
```

### 安装libstapsdt

参考git地址，源码编译安装

### 安装bcc

```
sudo apt-get install bpfcc-tools linux-headers-$(uname -r) 
```

### demo讲解

demo程序启动命令：
```
./demo PROVIDER_NAME PROBE_NAME
```

启动libstapsdt的demo程序，该程序将argv[1]（也就是命令行参数1）传递给了providerInit()函数，将argv[2]传递给providerAddProbe()函数。

```
  provider = providerInit(argv[1]);
  for (int idx = 0; idx < (probesCount); idx++) {
    probes[idx] = providerAddProbe(provider, argv[idx + 2], 2, uint64, int64);
  }
```
如上代码所示，probe的name为PROBE_NAME，捕获两个参数，均为uint64类型。


使用fire函数触发probe，传递i和j两个uint64值，这两个值可以在观测probe时获取到。

```
probeFire(probes[idx], i++, j--);
```

demo程序终端打印如下：
```
Firing probes...
Firing probe [0]...
Probe fired!
Firing probes...
Firing probe [0]...
Probe fired!
Firing probes...
Firing probe [0]...
Probe fired!
Firing probes...
Firing probe [0]...
```

启动bcc的trace程序进行跟踪探测点，打印捕获到的probe触发时传入的参数。
```
sudo /sbin/trace-bpfcc -p (pgrep demo) 'u::PROBE_NAME "i=%d j=%d", arg1, arg2'
```

bcc trace终端打印如下：
```
PID     TID     COMM            FUNC             -
10958   10958   demo            PROBE_NAME       i=3 j=-3
10958   10958   demo            PROBE_NAME       i=4 j=-4
10958   10958   demo            PROBE_NAME       i=5 j=-5
10958   10958   demo            PROBE_NAME       i=6 j=-6
10958   10958   demo            PROBE_NAME       i=7 j=-7
10958   10958   demo            PROBE_NAME       i=8 j=-8
10958   10958   demo            PROBE_NAME       i=9 j=-9
10958   10958   demo            PROBE_NAME       i=10 j=-10
```
