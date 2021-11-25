---
title: hexo博客搭建记录
date: 2020-11-05 19:21:32
toc: true
mathjax: true
description: 记一次blog部署过程
tags:
- hexo
- blog
categories:
- 笔记
---

## hexo博客搭建记录

最开始hexo博客是因为看到了[CosmosNing的个人博客](https://cosmosning.github.io/)，感觉效果很Nice。

开始参考他的[使用 Hexo搭建并部署个人博客](https://cosmosning.github.io/2020/03/12/shi-yong-hexo-da-jian-bing-bu-shu-ge-ren-bo-ke/)的这篇文章。但是在搭建的过程中，遇到了一些问题，这里记录下来，给需要的同学参考。

### 1、安装hexo环境遇到的网络问题
在使用ubuntu的apt命令安装nodejs和npm后，开始使用如下命令安装hexo环境。
```
npm install -g hexo-cli
```
但是，遇到了网络问题，一直卡顿，和各种网络不能连接的报错！参考了网上，代理、换淘宝源、使用cnpm等，都没能解决。
**解决方法**：
参考[Hexo官方文档](https://hexo.io/zh-cn/docs/)，从[NodeSource](https://github.com/nodesource/distributions)安装最新版的Nodejs。

```
# Using Ubuntu 
curl -sL https://deb.nodesource.com/setup_15.x | sudo -E bash -
sudo apt-get install -y nodejs
```

### 2、使用TravisCI部署时遇到问题
在使用TravisCI部署博客时，遇到了“--token” 相关的问题，花费了大量的时间，没有解决，发现在TravisCI仓库的issues里也有这个问题亟待解决。
**解决方法**
最终，选择了Actions部署Hexo。
可以参考我在".github/workflows/action.yaml"中的自动部署[配置](https://github.com/kktao/kktao.github.io)。

### 3、在使用部署时，遇到“xxx.md”文件中的某个属性不能识别
在使用部署时，遇到“xxx.md”文件中的某个属性不能识别，如我遇到了README.md中的“note”属性不能识别，产生报错。
**解决方法**：
直接将REDEME.md文件删除，即直接将产生报错的文件删除，再尝试部署。

### 4、在部署成功后，访问时显示404
在使用Actions部署成功后，但是访问时遇到了404的问题。
**解决方法**：
在your_github_name.github.io仓库的setting中，将仓库的"GithubPages"的Source设置为master分支。

