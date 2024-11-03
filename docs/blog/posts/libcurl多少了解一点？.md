---
title: libcurl多少了解一点？
date: 2024-11-03
authors: [KenForever1]
categories: 
  - libcurl
labels: []
---

## libcurl多少了解一点？
我们经常用一个工具叫做curl，你知道libcurl和curl的区别吗？
你知道怎么安装和使用libcurl吗？
<!-- more -->
curl是一个很常用的网络工具，比如我要下载文件，可以用wget、curl等。也可以支持Range，按照给定Header中范围下载，比如下载文件0-999，也就是1000bytes。

libcurl是一个c语言实现网络库，你能想到网络相关的FTP、HTTP2、SSL等等，都有支持实现。比如你可以通过libcurl实现一个http的库，支持GET、POST等，然后你可以在此基础上实现OpenAI应用、聊天工具等等。

看个小例子，怎么使用：
```
#include <curl/curl.h>

```

安装curl，才开始使用你也许会疑惑，我怎么安装呢？
```bash
apt install curl
```
不巧，你安装的是curl工具，不是libcurl。

```
apt install libcurl4-openssl-dev
```

有多个ssl版本，比如
gnu，gpl开源协议，就是你的应用需要开发源码，和linux等一样。
企业更多使用openssl, apache开源协议，和MIT等一样，比较弱，你使用apache开源项目开发的应用不用开源源码。
rults，纯rust开发。

```c
GET baidu.com

```

## 回调函数

设置回调函数，

通过DATA,传递用户回调函数指针