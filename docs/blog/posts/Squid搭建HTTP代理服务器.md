---
title: Squid搭建HTTP代理服务器
date: 2024-10-24
authors: [KenForever1]
categories: 
  - http proxy
  - squid
labels: []
comments: true
---


背景，有一台Mac电脑可以访问外网，但是有个开发板通过网线能够和Mac互通ssh，但是板子不能连接外网。想要通过Mac启动一个代理服务器，然后开发板的可以访问外网，包括终端和docker等。
<!-- more -->
## 1 Mac安装配置Squid

```bash
brew install squid
```

修改配置文件，其他默认就可以了：
```bash
http_access allow !Safe_ports

# Deny CONNECT to other than secure SSL ports
http_access allow CONNECT !SSL_ports

# Only allow cachemgr access from localhost
http_access allow localhost manager
http_access deny manager

# And finally deny all other access to this proxy
http_access allow all

# Squid normally listens to port 3128
http_port 3128
```

完整配置文件，http和https都可以代理，不需要签发https证书。
```bash
#
# Recommended minimum configuration:
#

# Example rule allowing access from your local networks.
# Adapt to list your (internal) IP networks from where browsing
# should be allowed
acl localnet src 0.0.0.1-0.255.255.255	# RFC 1122 "this" network (LAN)
acl localnet src 10.0.0.0/8		# RFC 1918 local private network (LAN)
acl localnet src 100.64.0.0/10		# RFC 6598 shared address space (CGN)
acl localnet src 169.254.0.0/16 	# RFC 3927 link-local (directly plugged) machines
acl localnet src 172.16.0.0/12		# RFC 1918 local private network (LAN)
acl localnet src 192.168.0.0/16		# RFC 1918 local private network (LAN)
acl localnet src fc00::/7       	# RFC 4193 local private network range
acl localnet src fe80::/10      	# RFC 4291 link-local (directly plugged) machines

acl SSL_ports port 443
acl Safe_ports port 80		# http
acl Safe_ports port 21		# ftp
acl Safe_ports port 443		# https
acl Safe_ports port 70		# gopher
acl Safe_ports port 210		# wais
acl Safe_ports port 1025-65535	# unregistered ports
acl Safe_ports port 280		# http-mgmt
acl Safe_ports port 488		# gss-http
acl Safe_ports port 591		# filemaker
acl Safe_ports port 777		# multiling http

#
# Recommended minimum Access Permission configuration:
#
# Deny requests to certain unsafe ports
http_access allow !Safe_ports

# Deny CONNECT to other than secure SSL ports
http_access allow CONNECT !SSL_ports

# Only allow cachemgr access from localhost
http_access allow localhost manager
http_access deny manager

# This default configuration only allows localhost requests because a more
# permissive Squid installation could introduce new attack vectors into the
# network by proxying external TCP connections to unprotected services.
http_access allow localhost

# The two deny rules below are unnecessary in this default configuration
# because they are followed by a "deny all" rule. However, they may become
# critically important when you start allowing external requests below them.

# Protect web applications running on the same server as Squid. They often
# assume that only local users can access them at "localhost" ports.
http_access deny to_localhost

# Protect cloud servers that provide local users with sensitive info about
# their server via certain well-known link-local (a.k.a. APIPA) addresses.
http_access deny to_linklocal

#
# INSERT YOUR OWN RULE(S) HERE TO ALLOW ACCESS FROM YOUR CLIENTS
#

# For example, to allow access from your local networks, you may uncomment the
# following rule (and/or add rules that match your definition of "local"):
# http_access allow localnet

# And finally deny all other access to this proxy
http_access allow all

# Squid normally listens to port 3128
http_port 3128

# Uncomment and adjust the following to add a disk cache directory.
#cache_dir ufs /opt/homebrew/var/cache/squid 100 16 256

# Leave coredumps in the first cache dir
coredump_dir /opt/homebrew/var/cache/squid

#
# Add any of your own refresh_pattern entries above these.
#
refresh_pattern ^ftp:		1440	20%	10080
refresh_pattern -i (/cgi-bin/|\?) 0	0%	0
refresh_pattern .		0	20%	4320
```

启动squid
```bash
sudo brew services start squid
# 或者
/opt/homebrew/opt/squid/sbin/squid -f /opt/homebrew/etc/squid.conf
```


查看服务，测试代理
```bash
sudo brew services list

proxychains4 curl https://www.baidu.com
curl -x http://192.168.1.102:3128 http://www.baidu.com

```

http://cooolin.com/scinet/2020/06/21/squid-proxy-simple.html


## 2 proxychains4使用代理服务器

下载proxychains4 deb文件：
```bash
https://packages.debian.org/buster/proxychains4
sudo dpkg -i *.deb
```

配置proxychains，vim /etc/proxychains4.conf
```bash
[ProxyList]
# add proxy here ...
# meanwile
# defaults set to "tor"
#socks4 	127.0.0.1 9050
http 192.168.1.102 3128
```
## 3 docker使用代理服务器

docker配置代理：
proxychians对docker无效，因为docker命令执行时只是客户端工具，并没有代理docker-daemon程序。参考：
https://yeasy.gitbook.io/docker_practice/advanced_network/http_https_proxy

### 3.1 为 dockerd 设置网络代理

```bash
sudo mkdir -p /etc/systemd/system/docker.service.d
```
为 dockerd 创建 HTTP/HTTPS 网络代理的配置文件，文件路径是 /etc/systemd/system/docker.service.d/http-proxy.conf 。并在该文件中添加相关环境变量。

```bash
[Service]
Environment="HTTP_PROXY=http://proxy.example.com:8080/"
Environment="HTTPS_PROXY=http://proxy.example.com:8080/"
Environment="NO_PROXY=localhost,127.0.0.1,.example.com"
```

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```

### 3.2 docker容器设置代理
更改 docker 客户端配置：创建或更改 ~/.docker/config.json，并在该文件中添加相关配置。

```json
{
 "proxies":
 {
   "default":
   {
     "httpProxy": "http://proxy.example.com:8080/",
     "httpsProxy": "http://proxy.example.com:8080/",
     "noProxy": "localhost,127.0.0.1,.example.com"
   }
 }
}

```

### 3.3 docker build设置代理
docker在使用docker build时，总是报错resolve xxx.com 出错，一度以为是dns没有配置对。实际上是没有使用上代理服务器。

尝试了解决办法：

+ docker配置dns，修改/etc/docker/daemon.json，加入dns选项。

```json
{
  "data-root": "/package/docker_data",
  "registry-mirrors": ["https://docker.mirrors.ustc.edu.cn/","https://hub-mirror.c.163.com","https://registry.docker-cn.com"],
  "dns": ["8.8.8.8"]
}
```

+ 运行docker build时，指定参数和设置bash环境变量：

```bash
#!/bin/bash
export  HTTP_PROXY="http://192.168.1.102:3128"
export  HTTPS_PROXY="http://192.168.1.102:3128"

proxychains4 docker build --build-arg "HTTP_PROXY=http://192.168.1.102:3128/" --build-arg "HTTPS_PROXY=http://192.168.1.102:3128/"  -f ./Dockerfile.example -t xxx_yyy:v1 .
```

正确解决方法，在Dockerfile中加入

```bash
ENV http_proxy=http://192.168.1.102:3128
ENV https_proxy=http://192.168.1.102:3128
```

### 3.4 apt update报错时间没有对齐问题
由于开发板之前没有联网，时间可能和互联网时间没有对齐，不能apt update。
解决办法，尝试了安装ntpdate命令同步时间，还是不行。采用了如下命令：

```bash
apt-get -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false update
```