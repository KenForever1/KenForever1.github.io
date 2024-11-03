---
title: 如何实现proxychains代理工具（二）
date: 2024-11-03
authors: [KenForever1]
categories: 
  - proxychains
  - 源码实现
labels: []
---

## proxychains代理的建立过程

### proxychains的几种代理模式你清楚吗？

`   proxychains` 通过代理服务器链路来转发网络连接的工具。它支持多种代理链模式，。包括 `DYNAMIC_TYPE`、`STRICT_TYPE` 和 `RANDOM_TYPE` 每种模式在代理选择和使用策略上有所不同。

<!-- more -->
1. **DYNAMIC_TYPE**:
    
    - **动态模式**：在这种模式下，`proxychains` 会尽量使用代理列表中可用的代理，并在代理链中动态调整代理。
    - **灵活性**：如果当前代理无法连接，`proxychains` 会尝试下一个可用代理。它不会因为某个代理不可用而导致整个连接失败。
    - **优势**：提高了连接成功率，尤其是在代理服务器不稳定或不可用的情况下。
2. **STRICT_TYPE**:
    
    - **严格模式**：`proxychains` 会按照配置文件中列出的顺序严格使用代理。
    - **固定链路**：所有代理必须按顺序连接。如果其中一个代理不可用，整个连接将失败。
    - **优势**：确保流量经过每个指定的代理服务器，这对某些需要特定路径的应用场景可能很有用。
3. **RANDOM_TYPE**:
    
    - **随机模式**：`proxychains` 会随机选择代理进行连接。
    - **随机性**：每次连接可能会使用不同的代理链，增加了路径的随机性。
    - **优势**：增加连接路径的多样性，可以用于规避某些特定的网络限制。

如果需要通过特定的代理路径来访问资源，可以选择 `STRICT_TYPE`；如果需要更高的连接成功率和灵活性，可以选择 `DYNAMIC_TYPE`；而 `RANDOM_TYPE` 则适合需要随机化流量路径的场景。

[!HTTP代理链过程]()

### Dynamic模式的源码分析
一切仿佛都没有源码更加清晰了, 看一些Dynamic模式的源码：
```c
case DYNAMIC_TYPE:
			calc_alive(pd, proxy_count);
			offset = 0;
			do {
				if(!(p1 = select_proxy(FIFOLY, pd, proxy_count, &offset)))
					goto error_more;
			} while(SUCCESS != start_chain(&ns, p1, DT) && offset < proxy_count); // client和第一个proxy server建立tcp连接ns
			for(;;) {
				p2 = select_proxy(FIFOLY, pd, proxy_count, &offset);
				if(!p2)
					break;
				if(SUCCESS != chain_step(ns, p1, p2)) {
					PDEBUG("GOTO AGAIN 1\n");
					goto again;
				}
				p1 = p2; // 建立代理链的过程，不停p1相当于一个指针，不停的向后移动, 建立的过程是在和proxy1的ns连接上，发送HTTP CONNCET请求给proxy2，这样建立了proxy1到proxy2的转发。然后循环，建立proxy2和proxy3的转发...
			}
			//proxychains_write_log(TP);
			p3->ip = target_ip;
			p3->port = target_port;
			if(SUCCESS != chain_step(ns, p1, p3)) // 最后代理链的最后一个代理proxy和target建立连接，比如代理服务器和google.com
				goto error;
			break;

dup2(ns, sock); // sock是用户请求connect的参数，通过dup2函数将和代理的sock复制给用户请求的sock，因此用户使用上了代理链访问目标地址
```

start_chain函数的实现, 实际就是调用了timed_connect函数，和代理服务器建立tcp连接。
```c
static int start_chain(int *fd, proxy_data * pd, char *begin_mark) {
	struct sockaddr_in addr;
	char ip_buf[16];
    ....
	*fd = socket(PF_INET, SOCK_STREAM, 0);
	memset(&addr, 0, sizeof(addr));
	addr.sin_family = AF_INET;
	addr.sin_addr.s_addr = (in_addr_t) pd->ip.as_int;
	addr.sin_port = pd->port;
	if(timed_connect(*fd, (struct sockaddr *) &addr, sizeof(addr))) {
		pd->ps = DOWN_STATE;
		goto error1;
	}
    ......
}
```
chain_step函数的实现，调用了tunnel_to函数，实际就是根据类型去建立代理链。
```c
static int chain_step(int ns, proxy_data * pfrom, proxy_data * pto) {
    ......
	retcode = tunnel_to(ns, pto->ip, pto->port, pfrom->pt, pfrom->user, pfrom->pass);
    ......
	return retcode;
}

static int tunnel_to(int sock, ip_type ip, unsigned short port, proxy_type pt, char *user, char *pass) {
    ......
    switch (pt) {
		case RAW_TYPE: {
			return SUCCESS;
		}
		break;
		case HTTP_TYPE:{
        }
        case SOCKS4_TYPE:{}
        case SOCKS5_TYPE:{}
    }
    ......
}
```
其中HTTP PROXY建立过程就是发送CONNECT请求，接受Response，成功以后用户的请求就由代理服务器转发处理了。是不是很简单！
```c
case HTTP_TYPE:{
        snprintf((char *) buff, sizeof(buff), "CONNECT %s:%d HTTP/1.0\r\n", dns_name,
                ntohs(port));
        strcat((char *) buff, "\r\n");
        len = strlen((char *) buff);
        if(len != (size_t) send(sock, buff, len, 0))
            goto err;

        len = 0;
        // read header byte by byte.
        while(len < BUFF_SIZE) {
            if(1 == read_n_bytes(sock, (char *) (buff + len), 1))
                len++;
            else
                goto err;
            if(len > 4 &&
                buff[len - 1] == '\n' &&
                buff[len - 2] == '\r' && buff[len - 3] == '\n' && buff[len - 4] == '\r')
                break;
        }

        // if not ok (200) or response greather than BUFF_SIZE return BLOCKED;
        if(len == BUFF_SIZE || !(buff[9] == '2' && buff[10] == '0' && buff[11] == '0'))
            return BLOCKED;

        return SUCCESS;
    }
    break;
```

## 关于connect函数递归调用的问题？

在开发时，如果使用了比较高级的网络封装库，这时要注意会递归调用, 因为在网络库中直接调用的connect函数。比如重写了自己的connect函数，在使用网络库的connect_time_out函数中，可能是直接封装的c库中的connect函数，那么就会无限递归。最后函数栈就Segment fault了。
这里就需要我们自己写一个connect_time_out函数，在里面调用dlsym获取到的true_connect，避免了递归调用。

在rust中，不能直接使用TcpStream::connect_timeout，同样也不能用nix::sys::socket::connect。

> We can't use nix::sys::socket::connect since it would call our hooked connect function and recurse infinitely.

```rust
let stream = TcpStream::connect_timeout(&socket_addr, Duration::from_secs(5))?;
```
而需要自定义实现如下：
```c
// https://github1s.com/haad/proxychains/blob/master/src/core.c#L204-L240
static int timed_connect(int sock, const struct sockaddr *addr, socklen_t len) {
	int ret, value;
	socklen_t value_len;
	struct pollfd pfd[1];
	pfd[0].fd = sock;
	pfd[0].events = POLLOUT;
	fcntl(sock, F_SETFL, O_NONBLOCK);
	ret = true_connect(sock, addr, len);
	PDEBUG("\nconnect ret=%d\n", ret);

	if(ret == -1 && errno == EINPROGRESS) {
		ret = poll_retry(pfd, 1, tcp_connect_time_out);
		PDEBUG("\npoll ret=%d\n", ret);
		if(ret == 1) {
			value_len = sizeof(socklen_t);
			getsockopt(sock, SOL_SOCKET, SO_ERROR, &value, &value_len);
			PDEBUG("\nvalue=%d\n", value);
		} else {
			ret = -1;
		}
	}
    ......
	fcntl(sock, F_SETFL, !O_NONBLOCK);
	return ret;
}
```

同样的逻辑，rust实现如下：
```rust
// https://github.com/mlvl42/proxyc/blob/HEAD/libproxyc/src/util.rs#L2-L29
pub fn timed_connect(fd: RawFd, addr: &SockAddr, timeout: usize) -> Result<(), Error> {
    let c_connect = CONNECT.expect("Cannot load symbol 'connect'");

    let mut fds = [PollFd::new(fd, PollFlags::POLLOUT)];
    let mut oflag = OFlag::empty();

    oflag.toggle(OFlag::O_NONBLOCK);
    match fcntl(fd, FcntlArg::F_SETFL(OFlag::O_NONBLOCK)) {
        Ok(_) => (),
        Err(e) => error!("fcntl NONBLOCK error: {}", e),
    };

    let res = unsafe {
        let (ptr, len) = addr.as_ffi_pair();
        c_connect(fd, ptr, len)
    };

    if let (-1, Errno::EINPROGRESS) = (res, errno()) {
        let ret = poll_retry(&mut fds, timeout)?;

        match ret {
            1 => {
                match getsockopt(fd, SocketError)? {
                    0 => (),
                    _ => return Err(Error::Socket),
                };
            }
            _ => return Err(Error::Connect("poll_retry".into())),
        };
    }

    oflag.toggle(OFlag::O_NONBLOCK);
    match fcntl(fd, FcntlArg::F_SETFL(oflag)) {
        Ok(_) => (),
        Err(e) => error!("fcntl BLOCK error: {}", e),
    };
    ......
}
```
参考实现：[proxyc](https://github.com/mlvl42/proxyc/blob/HEAD/libproxyc/src/util.rs#L2-L29)


## 学习源码，动手实现一个proxychains工具

如果你学会了，也想大展拳脚一番，这里整理了几个比较清晰的实现，可以参考。

### c语言实现

+ [haad/proxychains](https://github.com/haad/proxychains/blob/master/src/core.c)

### rust实现

#### mlvl42/proxyc

[mlvl42/proxyc](https://github.com/mlvl42/proxyc/blob/HEAD/libproxyc/src/core.rs)和**haad/proxychains** 实现逻辑相同，如果你想学习如何用rust实现c的功能可以读读这个源码。比如如何dlsym、如何设置非阻塞fd、如何编译so自定义connect函数。

如何实现动态库中的初始化逻辑？
```rust
static ONCE: std::sync::Once = std::sync::Once::new();
/// This is called when our dynamic library is loaded, so we setup our internals
/// here.
#[no_mangle]
#[link_section = ".init_array"]
static LD_PRELOAD_INIT: extern "C" fn() = self::init;
extern "C" fn init() {
    ONCE.call_once(|| {
        let config = &*core::CONFIG;
        std::env::set_var("RUST_LOG", config.log_level.to_string());
        pretty_env_logger::init();
        debug!("init pid: {}", std::process::id());
        info!("chain_type: {:?}", config.chain_type);
        info!("proxies:");
        for p in &config.proxies {
            info!("\t{}", p);
        }
    });
}
```
".init_array"这个段是 ELF（Executable and Linkable Format）文件格式的一部分，专门用于存放初始化函数。在程序启动时，操作系统会调用这个段中的所有函数，以执行一些初始化操作。
这种模式通常用于实现动态库中的初始化逻辑，在使用 LD_PRELOAD 进行函数钩挂（hooking）时。通过将初始化代码放入 .init_array 中，可以确保在共享库加载时自动调用这些初始化函数，而不需要在主程序中显式地调用它们。


#### alifarrokh/proxychains

[alifarrokh/proxychains](https://github.com/alifarrokh/proxychains/blob/HEAD/src/connection.rs#L16
) 采用rust风格实现，自定义了connect、write、read函数，通过channel连接用户读写和proxy代理读写，而没有采用dup2(new_fd, fd)的方式。核心逻辑如下：

```rust
tokio::spawn(async move {
                    let connection = unsafe { (*CONNECTIONS).get_mut(&fd) }.unwrap();
                    let target = connection.target_addr.clone();
                    let (connection_reader, connection_writer) = connection.split();

                    let stream = ProxyChains::connect(target, config()).await;
                    match stream {
                        Ok(mut stream) => {
                            let (mut reader, mut writer) = stream.split();
                        let _ = futures::join!(
                            copy(connection_reader, &mut writer),
                            copy(&mut reader, connection_writer)
                        );
                        },
                        Err(e) => eprintln!("Failed to create proxychains. {}", e.to_string()),
                    }
                });
```
可以学习到rust异步stream，以及channel通信的写法等。

#### KernelErr/proxychain-rs

[KernelErr/proxychain-rs](https://github1s.com/KernelErr/proxychain-rs)将http代理可以转换成socks5代理。
```bash
proxychain -i socks5://127.0.0.1:9000 -o http://127.0.0.1:8123
```

感谢您的阅读！