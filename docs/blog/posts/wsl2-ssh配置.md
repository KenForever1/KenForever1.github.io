---
title: wsl2 ssh配置与ai编程环境搭建
date: 2025-06-28
authors: [KenForever1]
categories: 
  - C++
labels: []
comments: true
---

采用windows + wsl2（安装ubuntu）的方式进行开发。

## ssh网络配置
comate是一个和vscode类似的AI智能编码工具，但是没有wsl直接远程连接，只能通过ssh连接。
因此，需要windows上ssh远程wsl2。

推荐配置模式：

+ wsl2采用mirror模式，采用同一个ip，同时方便了网络代理的使用。

> 如果你的windows使用了clash工具使用网络代理，那么可以暴露一个7890的http代理端口(clash自带功能)。采用mirror模式，wsl2访问windows的7890端口进行代理就很方便了，直接配置127.0.0.1:7890即可。例如：proxychains4 -f /etc/proxychains.conf

```bash
[ProxyList]
http    127.0.0.1 7890
```

+ ubuntu安装配置openssh-server, 安装启动openssh-server服务很多资料了
  
```bash
sudo apt install openssh-server
sudo systemctl start ssh
sudo systemctl enable ssh
```
修改的ssh服务配置如下，只需要修改端口号，其他保持默认
```bash
sudo vim /etc/ssh/sshd_config

Port 2212
#AddressFamily any
#ListenAddress 0.0.0.0
#ListenAddress ::
```

+ windows上连接wsl2

```bash
ssh -p 2212 username@localhost
```