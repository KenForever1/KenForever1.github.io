---
title: 如何实现proxychains代理工具（一）
date: 2024-11-03
authors: [KenForever1]
categories: 
  - proxychains
  - 源码实现
labels: []
---

## proxychains工具实现原理

相信很多朋友都听过或者使用过proxychains工具，它是一个终端代理工具，支持HTTP proxy、Socks5、Socks4协议，同时还分为Strict、Dynamic、Random三种模式。关于它的用法可以参考之前的文章，[Squid搭建HTTP代理服务器](https://kenforever1.github.io/blog/squid%E6%90%AD%E5%BB%BAhttp%E4%BB%A3%E7%90%86%E6%9C%8D%E5%8A%A1%E5%99%A8/)。

以linux平台为例，这里的核心技术点：
<!-- more -->
### （一）LD_PRELOAD和自定义libc函数

采用了LD_PRELOAD，然后我们自定义了c库的connect函数，这样就可以让你的终端命令行工具调用到我们实现的connect函数。其它要自定义的函数也一样。
这同样也决定了proxychains不能代理静态编译的工具，只能代理加载动态库的工具，因为只有动态库，才可以LD_PRELOAD加载到我们自定义的实现。

!!! Tip: 动态库和静态库的区别？以及go语言的实现区别
    插一段，加载动态库的优点大家可能都知道，比如可以使得工具更小，也容易升级。静态库更容易分发，比如我编译了一个exe可执行文件，拷贝到另一个机器。如果动态库可能会报错缺失so，那么静态库直接编译到exe中，就更容易分发了。
    其它语言rust、go都可以编译成静态或者动态。go语言如果是采用GC(Go Compiler)编译器LD_PRELOAD会失效。因为它底层调用的自己的实现,直接调用 syscall 进行了系统调用，而不是libc库的connect实现。因此go写的程序如果要proxychains代理，需要采用GCCGO编译器编译。go的实现区别可以参考[Golang编写程序是无法使用proxychains代理](https://void-shana.moe/posts/proxychains-ng)。


### （二）dlsym RTLD_NEXT

目的是通过dlsym RTLD_NEXT获取c库的connect，记为true_connect，在需要直接调用c库的connect函数时，就可以调用true_connect了。
实现：
```c
// https://github.com/haad/proxychains/blob/master/src/libproxychains.c
static void* load_sym(char* symname, void* proxyfunc) {
	void *funcptr = dlsym(RTLD_NEXT, symname);
	......
	return funcptr;
}

#define SETUP_SYM(X) do { true_ ## X = load_sym( # X, X ); } while(0)

SETUP_SYM(connect);
SETUP_SYM(gethostbyname);
SETUP_SYM(getaddrinfo);
SETUP_SYM(freeaddrinfo);
SETUP_SYM(gethostbyaddr);
SETUP_SYM(getnameinfo);
```

### （三）dup2重定向文件描述符

dup2可以实现把文件描述符A重定向到文件描述符B。也就是说，所有对A有写入，最终都会写入到B。
ns表示client和proxy server建立的连接fd，sock是用户请求connect的参数。
```c
dup2(ns, sock); // sock是用户请求connect的参数，通过dup2函数将和代理的sock重定向给用户请求的sock，因此用户使用上了代理链访问目标地址
```

当然还有另一种实现方式，不采用dup2，而是用户自定义write和read方法，覆盖c库中的write和read方法。实现原理是，建立好和proxy的连接后，将用户read、write和proxy的read、write channel绑定上，进行通信，实现可以参考[alifarrokh/proxychains](https://github.com/alifarrokh/proxychains/blob/HEAD/src/proxychains.rs#L79)。

**完整流程**

+ 自定义connect函数，返回文件描述符fd_a
+ 利用dlsym拿到真实的connect函数，与proxy建立连接，拿到文件描述符fd_b
+ 利用dup2把fd_a重定向到fd_b
+ 发到fd_a的数据包都被发送到了proxy上
+ 数据到了proxy上面之后，剩下的任务就交给proxy了
  
参考：[proxychains是怎么工作的](https://segmentfault.com/a/1190000018194455)

## HTTP代理链是如何建立的呢？

[!HTTP代理链过程]()

以HTTP proxy协议为例，它的原理其实就是发了一个请求给代理服务器，这个HTTP请求头是"CONNECT"。

```bash
CONNECT ip:port HTTP/1.0\r\n\r\n

例如：
CONNECT google.com:443 HTTP/1.0\r\n\r\n
```

当使用 `HTTP CONNECT` 方法在客户端、代理服务器和目标服务器之间建立连接后，代理服务器的职责就是数据转发。
那请求和数据流的过程是怎么样的呢？

1. **客户端到代理服务器**:
    
    - 客户端首先向代理服务器发送一个 `CONNECT` 请求，要求代理服务器与目标服务器建立隧道。
    - 代理服务器响应 `200 Connection Established` 表示隧道建立成功。
2. **代理服务器到目标服务器**:
    
    - 代理服务器在收到 `CONNECT` 请求后，会尝试与目标服务器建立一个直接的 TCP 连接。
    - 一旦连接建立，代理服务器不再解析或修改数据，而是直接转发数据。
3. **请求和响应的转发**:
    
    - **客户端请求目标服务器**: 客户端通过已建立的隧道发送请求数据。此数据首先到达代理服务器。
    - **代理服务器转发请求**: 代理服务器将请求数据原封不动地转发给目标服务器。
    - **目标服务器响应**: 目标服务器处理请求并返回响应数据。
    - **代理服务器转发响应**: 代理服务器将响应数据原封不动地转发回客户端。

上面的过程清楚的呈现了一个代理服务器的代理建立过程。如果中间有多个代理，那么就形成了**代理链**。
在这个过程中，代理服务器扮演的是一个透明的中间人角色。它不会对经过它的数据进行任何理解或修改，只是简单地进行字节流的转发。因此，除了最初的 `CONNECT` 请求之外，客户端不需要再对代理服务器发起其他请求。所有后续的通信都是在这条建立的隧道上进行的，代理服务器只负责数据的传递。

```
client -> proxy1 -> proxy2 -> proxy3 -> proxy4 -> destination.
```

关于如何形成链条的过程，下面这篇文章讲解很清晰[proxy-chaining-how-does-it-exactly-work](https://superuser.com/questions/1213774/proxy-chaining-how-does-it-exactly-work)

> A subtlety though: After proxy1 agrees to act as a CONNECT proxy for you, it takes whatever payload you send and sends to proxy2 as if proxy1 was the author. The next request you send reaches proxy2. In the example scenario this is also a CONNECT request. proxy2 gets it from proxy1 and may even not know you exist. From its point of view proxy1 asks it to CONNECT to proxy3. At the same time proxy1 is unaware it asks anything (unless it peeks into what you send). So neither proxy "consciously" negotiates with the next. You negotiate on behalf of each one in chain. 

感谢您的阅读，关于proxychains的几种代理模式、以及关于connect函数递归调用的问题、学习源码，动手实现一个proxychains工具等内容将在下一篇文章介绍。bye！