---
title: netcap调试
date: 2024-09-06
authors: [KenForever1]
categories: 
  - ebpf
  - netcap
labels: []
comments: true
---

## rust实践
+ https://github.com/jeremychone/rust-genai 一个rust比较好的库，对错误处理、包管理等有很好的示例。
<!-- more -->

## 实现
tcpdump是ip层抓包工具，通过[netcap](https://github.com/bytedance/netcap)使用xdp的方式可以抓取更加底层的包。
> tcpdump 抓包点位置固定：入向是 xdp 之后，tc 之前；出向是 tc 之后。如果网络包通过其他方式进行路径优化后，不经过这几个位置，那么 tcpdump 就抓不到了，另外 tcpdump 还无法支持进程名，pid，namespace id 等其他过滤条件。

tcpdump将pcap filter过滤描述语法，转换成classic bpf对包进行过滤，如果返回0就丢弃该包，返回1就保留该包。
+ 复用pcap的filter语法，libpcap可以将filter转换成为classic bpf。
+ 再通过将classic bpf转换成为c，c转换成为ebpf，编译成二进制。
+ 或者直接根据classic bpf指令和ebpf指令的规则，将classic bpf转换成ebpf。

通过将ebpf程序注入内核，通过map或者buffer将内核包数据传递到用户态，然后将数据进行解析。对数据解析的部分通过起一个tcpdump -r的进程，解析pcap包。

## 相关库和原理
### golang

+ libpcap 将filter转换成classic bpf
+ [cloudflare/cbpfc](github.com/cloudflare/cbpfc) 将classic bpf转换成c（c可以通过llvm编译成ebpf）, 或者将classic bpf转换成ebpf
+ [cilium/ebpf](github.com/cilium/ebpf) 提供了ebpf相关指令类
+ [iovisor/gobpf](github.com/iovisor/gobpf) 


### rust
https://github.com/mmisono/rust-cbpf/tree/master
https://github1s.com/polachok/bpfjit/blob/master/src/lib.rs
https://github.com/qmonnet/rbpf


### golang-rust ffi
可以看到在ebpf领域，golang有良好的生态，包括了丰富的库。如果rust要使用除了重写以外，还可以通过golang-rust ffi进行复用。在项目中也是，可以复用其它语言开发的组件，然后逐步替换。

https://www.ihcblog.com/rust2go/

## 参考
+ [xcap：基于 eBPF 技术的下一代内核网络抓包工具](https://mp.weixin.qq.com/s?__biz=Mzg3Mjg2NjU4NA==&mid=2247484864&idx=1&sn=a212fbf34aa041c245be58fa02ea693b
)

+ https://blog.cloudflare.com/xdpcap