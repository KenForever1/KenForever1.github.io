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

## 理念

缩小范围，减少模块，最小单元复现。

+ new、malloc没有释放
+ vector或者map一直增长