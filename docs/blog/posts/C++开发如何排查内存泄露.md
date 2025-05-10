---
title: C++开发如何排查内存泄露（增长）
date: 2024-11-07
authors: [KenForever1]
categories: 
  - C++
labels: []
comments: true
---

不一定是内存泄露，也可能是内存持续分配，一直上涨，直到最后OOM。

<!-- more -->
## 工具的使用

### top
观察RES变化。

### pmap

```bash
pmap -x [pid] > tmp_time1.txt
# -k3,3n 排序依据第三列，即内存大小, 第三列是RES列
awk '{print $0}' tmp_time1.txt | sort -k3,3n | tail -n 10
```
对比不同时间点的pmap输出，找到内存分配持续上涨的模块。


```
# 提取 server_0418_2011.txt 中的地址列
awk '{print $1}' server_0418_2011.txt > addresses_0418.txt

# 使用 awk 处理 server_0419_2011.txt，打印不在 addresses_0418.txt 中的地址和对应的第三列
awk 'NR==FNR {addr[$1]; next} !($1 in addr) {print $1, $3}' addresses_0418.txt server_0419_2011.txt
```

```
awk '{print $1, $3}' server_0418_2011.txt > data1.txt
awk '{print $1, $3}' server_0419_2011.txt > data2.txt

join -o 1.1 1.2 2.2 data1.txt data2.txt | \
awk '{print $1, $3-$2, $2, $3}' | \
sort -k2,2nr | head -n 100
```

## 理念

缩小范围，减少模块，最小单元复现。

+ new、malloc没有释放
+ vector或者map一直增长

## ptmalloc换成jemalloc或者tcmalloc

内存不回收，碎片一直增加，导致oom。

定位工具。
```
malloc_trim(); 手动释放内存。


// https://zhuanlan.zhihu.com/p/682033366
valgrind --tool=massif 
```