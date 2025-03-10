---
title: 从Deepseek开源库3FS中学习fuse的使用-如何开发一个文件系统（一）
date: 2025-03-10
authors: [KenForever1]
categories: 
  - C++
labels: []
comments: true
---

## 背景介绍

Deepseek开源了一系列AI infra的相关的项目，其中包括了[deepseek-ai/3FS](https://github.com/deepseek-ai/3FS)。

> A high-performance distributed file system designed to address the challenges of AI training and inference workloads.

文件系统是任何操作系统的支柱，负责管理数据的存储和检索方式。传统上，开发文件系统是一项复杂而艰巨的任务，需要对内核编程有深入的了解。然而，有了 FUSE（用户空间文件系统），这项任务变得更加容易和通用。

<!-- more -->

FUSE（用户空间文件系统）是用户空间程序向 Linux 内核导出文件系统的接口。FUSE 项目由两个组件组成：fuse内核模块（在常规内核存储库中维护）和libfuse用户空间库（在此存储库中维护）。libfuse 为与 FUSE 内核模块通信提供参考实现。
FUSE 文件系统通常实现为与 libfuse 链接的独立应用程序。libfuse 提供挂载文件系统、卸载文件系统、从内核读取请求以及发送响应的功能。libfuse 提供两种 API：“高级” 同步 API 和 “低级” 异步 API。在这两种情况下，来自内核的传入请求都使用回调传递给主程序。当使用高级 API 时，回调可以处理文件名和路径而不是索引节点，并且当回调函数返回时请求处理完成。当使用低级 API 时，回调必须处理索引节点，并且必须使用单独的一组 API 函数显式发送响应。

FUSE（用户空间文件系统）是 Linux 中的一个软件层，它允许非特权用户在不编辑内核源代码的情况下创建自己的文件系统。它由三个主要组件组成：
fuse.ko - FUSE 内核模块，它提供了 FUSE 的接口。
libfuse - 用户空间库，它提供了处理与 FUSE 内核模块通信所需的 API，允许用户空间应用程序实现自定义文件系统逻辑。
fusermount - 一个挂载工具。

### 重要的数据结构、函数、宏定义

#### fuse_args

这个结构用于处理传递给 FUSE 文件系统的命令行参数。

```c++
struct fuse_args {
	int argc; // number of arguments
	char **argv; // argument vector, NULL terminated
	int allocated; // is argv allocated?
};
```

通过宏定义初始化，

```c++
#define FUSE_ARGS_INIT(argc, argv) { argc, argv, 0 }
```

在3fs中传递参数，

```c++
// https://github1s.com/deepseek-ai/3FS/blob/main/src/fuse/FuseMainLoop.cc#L47
std::vector<std::string> fuseArgs;
fuseArgs.push_back(programName);
if (allowOther) {
fuseArgs.push_back("-o");
fuseArgs.push_back("allow_other");
fuseArgs.push_back("-o");
fuseArgs.push_back("default_permissions");
}
fuseArgs.push_back("-o");
fuseArgs.push_back("auto_unmount");
fuseArgs.push_back("-o");
fuseArgs.push_back(fmt::format("max_read={}", maxbufsize));
fuseArgs.push_back(mountpoint);
fuseArgs.push_back("-o");
fuseArgs.push_back("subtype=hf3fs");
fuseArgs.push_back("-o");
fuseArgs.push_back("fsname=hf3fs." + clusterId);
std::vector<char *> fuseArgsPtr;
for (auto &arg : fuseArgs) {
fuseArgsPtr.push_back(const_cast<char *>(arg.c_str()));
}
struct fuse_args args = FUSE_ARGS_INIT((int)fuseArgsPtr.size(), fuseArgsPtr.data());
```

#### fuse_cmdline_opts

```c++
struct fuse_cmdline_opts {
	int singlethread;
	int foreground;
	int debug;
	int nodefault_subtype;
	char *mountpoint;
	int show_version;
	int show_help;
	int clone_fd;
	unsigned int max_idle_threads; 
	unsigned int max_threads; // This was added in libfuse 3.12
};
```

这个结构体用于存储从参数解析出的命令行选项。它有助于根据用户输入管理和配置 FUSE 文件系统。此结构可以使用fuse_parse_cmdline函数进行填充。



在3fs中使用示例：
```c++
struct fuse_cmdline_opts opts;
struct fuse_loop_config *config = fuse_loop_cfg_create();
SCOPE_EXIT { fuse_loop_cfg_destroy(config); };

if (fuse_parse_cmdline(&args, &opts) != 0) {
return 1;
}
```

#### fuse_session_new

创建一个新的low-level会话。此函数接受大多数与文件系统无关的挂载选项。

```c++
struct fuse_session *fuse_session_new(struct fuse_args *args,const struct fuse_lowlevel_ops *op,size_t op_size, void *userdata);
```
使用方法：
```c++
d.se = fuse_session_new(&args, &ops, sizeof(ops), NULL);
if (d.se == nullptr) {
return 1;
}
```
#### fuse_set_signal_handlers

此函数为信号SIGHUP、SIGINT和SIGTERM安装信号处理程序，这些信号处理程序将尝试卸载文件系统。如果已经为这些信号中的任何一个安装了信号处理程序，则不会替换它。此函数在成功时返回零，在失败时返回 -1。

```c++
int fuse_set_signal_handlers(struct fuse_session *se);
```

#### fuse_lowlevel_ops

这个结构代表了低级文件系统操作。在3fs项目中，就是通过这个结构注册了自己实现的回调函数。

注册回调函数：
```c++
const fuse_lowlevel_ops hf3fs_oper = {
    .init = hf3fs_init,
    .destroy = hf3fs_destroy,
    .lookup = hf3fs_lookup,
    .forget = hf3fs_forget,
    .getattr = hf3fs_getattr,
    .setattr = hf3fs_setattr,
    .readlink = hf3fs_readlink,
    .mknod = hf3fs_mknod,
    .mkdir = hf3fs_mkdir,
    .unlink = hf3fs_unlink,
    .rmdir = hf3fs_rmdir,
    .symlink = hf3fs_symlink,
    .rename = hf3fs_rename,
    .link = hf3fs_link,
    .open = hf3fs_open,
    .read = hf3fs_read,
    .write = hf3fs_write,
    .flush = hf3fs_flush,
    .release = hf3fs_release,
    .fsync = hf3fs_fsync,
    .opendir = hf3fs_opendir,
    //    .readdir = hf3fs_readdir,
    .releasedir = hf3fs_releasedir,
    //.fsyncdir = hf3fs_fsyncdir,
    .statfs = hf3fs_statfs,
    .setxattr = hf3fs_setxattr,
    .getxattr = hf3fs_getxattr,
    .listxattr = hf3fs_listxattr,
    .removexattr = hf3fs_removexattr,
    .create = hf3fs_create,
    .ioctl = hf3fs_ioctl,
    .readdirplus = hf3fs_readdirplus,
};
```

结构体定义：
```c++
struct fuse_lowlevel_ops {
	// Called when libfuse establishes communication with the FUSE kernel module.
	void (*init) (void *userdata, struct fuse_conn_info *conn);

	// Cleans up filesystem, called on filesystem exit.
	void (*destroy) (void *userdata);

	// Look up a directory entry by name and get its attributes.
	void (*lookup) (fuse_req_t req, fuse_ino_t parent, const char *name);

	// Can be called to forget about an inode
	void (*forget) (fuse_req_t req, fuse_ino_t ino, uint64_t nlookup);

	// Called to get file attributes
	void (*getattr) (fuse_req_t req, fuse_ino_t ino,
			 struct fuse_file_info *fi);

	// Called to set file attributes
	void (*setattr) (fuse_req_t req, fuse_ino_t ino, struct stat *attr,
			 int to_set, struct fuse_file_info *fi);

	// Called to read the target of a symbolic link
	void (*readlink) (fuse_req_t req, fuse_ino_t ino);

	// Called to create a file node
	void (*mknod) (fuse_req_t req, fuse_ino_t parent, const char *name,
		       mode_t mode, dev_t rdev);

	// Called to create a directory
	void (*mkdir) (fuse_req_t req, fuse_ino_t parent, const char *name,
		       mode_t mode);

	// Called to remove a file
	void (*unlink) (fuse_req_t req, fuse_ino_t parent, const char *name);

	// Called to remove a directory
	void (*rmdir) (fuse_req_t req, fuse_ino_t parent, const char *name);

	// Called to create a symbolic link
	void (*symlink) (fuse_req_t req, const char *link, fuse_ino_t parent,
			 const char *name);

	// Called to rename a file or directory
	void (*rename) (fuse_req_t req, fuse_ino_t parent, const char *name,
			fuse_ino_t newparent, const char *newname,
			unsigned int flags);

	// Called to create a hard link
	void (*link) (fuse_req_t req, fuse_ino_t ino, fuse_ino_t newparent,
		      const char *newname);

	// Called to open a file
	void (*open) (fuse_req_t req, fuse_ino_t ino,
		      struct fuse_file_info *fi);

	// Called to read data from a file
	void (*read) (fuse_req_t req, fuse_ino_t ino, size_t size, off_t off,
		      struct fuse_file_info *fi);

	// Called to write data to a file
	void (*write) (fuse_req_t req, fuse_ino_t ino, const char *buf,
		       size_t size, off_t off, struct fuse_file_info *fi);

	// Called on each close() of the opened file, for flushing cached data
	void (*flush) (fuse_req_t req, fuse_ino_t ino,
		       struct fuse_file_info *fi);

	// Called to release an open file (when there are no more references to an open file i.e all file descriptors are closed and all memory mappings are unmapped)
	void (*release) (fuse_req_t req, fuse_ino_t ino,
			 struct fuse_file_info *fi);

	// Called to synchronize file contents
	void (*fsync) (fuse_req_t req, fuse_ino_t ino, int datasync,
		       struct fuse_file_info *fi);

	// Called to open a directory
	void (*opendir) (fuse_req_t req, fuse_ino_t ino,
			 struct fuse_file_info *fi);

	// Called to read directory entries
	void (*readdir) (fuse_req_t req, fuse_ino_t ino, size_t size, off_t off,
			 struct fuse_file_info *fi);

	// Called to release an open directory
	void (*releasedir) (fuse_req_t req, fuse_ino_t ino,
			    struct fuse_file_info *fi);

	// Called to synchronize directory contents
	void (*fsyncdir) (fuse_req_t req, fuse_ino_t ino, int datasync,
			  struct fuse_file_info *fi);

	// Called to get file system statistics
	void (*statfs) (fuse_req_t req, fuse_ino_t ino);

	// Called to set an extended attribute
	void (*setxattr) (fuse_req_t req, fuse_ino_t ino, const char *name,
			  const char *value, size_t size, int flags);

	// Called to get an extended attribute
	void (*getxattr) (fuse_req_t req, fuse_ino_t ino, const char *name,
			  size_t size);

	// Called to list extended attribute names
	void (*listxattr) (fuse_req_t req, fuse_ino_t ino, size_t size);

	// Called to remove an extended attribute
	void (*removexattr) (fuse_req_t req, fuse_ino_t ino, const char *name);

	// Called to check file-access permissions
	void (*access) (fuse_req_t req, fuse_ino_t ino, int mask);

	// Called to create and open a file
	void (*create) (fuse_req_t req, fuse_ino_t parent, const char *name,
			mode_t mode, struct fuse_file_info *fi);

	// Called to get a file lock
	void (*getlk) (fuse_req_t req, fuse_ino_t ino,
		       struct fuse_file_info *fi, struct flock *lock);

	// Called to set a file lock
	void (*setlk) (fuse_req_t req, fuse_ino_t ino,
		       struct fuse_file_info *fi,
		       struct flock *lock, int sleep);

	// Called to map a block index within file to a block index within device
	void (*bmap) (fuse_req_t req, fuse_ino_t ino, size_t blocksize,
		      uint64_t idx);

	// The ioctl handler
#if FUSE_USE_VERSION < 35
	void (*ioctl) (fuse_req_t req, fuse_ino_t ino, int cmd,
		       void *arg, struct fuse_file_info *fi, unsigned flags,
		       const void *in_buf, size_t in_bufsz, size_t out_bufsz);
#else
	void (*ioctl) (fuse_req_t req, fuse_ino_t ino, unsigned int cmd,
		       void *arg, struct fuse_file_info *fi, unsigned flags,
		       const void *in_buf, size_t in_bufsz, size_t out_bufsz);
#endif

	// Called to poll a file for I/O readiness.
	void (*poll) (fuse_req_t req, fuse_ino_t ino, struct fuse_file_info *fi,
		      struct fuse_pollhandle *ph);

	// Called to write a buffer to a file.
	void (*write_buf) (fuse_req_t req, fuse_ino_t ino,
			   struct fuse_bufvec *bufv, off_t off,
			   struct fuse_file_info *fi);

	// Called to reply to a retrieve operation.
	void (*retrieve_reply) (fuse_req_t req, void *cookie, fuse_ino_t ino,
				off_t offset, struct fuse_bufvec *bufv);

	// Called to forget multiple inodes
	void (*forget_multi) (fuse_req_t req, size_t count,
			      struct fuse_forget_data *forgets);

	// Called to acquire, modify or release a file lock
	void (*flock) (fuse_req_t req, fuse_ino_t ino,
		       struct fuse_file_info *fi, int op);

	//  Called to allocate space to a file
	void (*fallocate) (fuse_req_t req, fuse_ino_t ino, int mode,
		       off_t offset, off_t length, struct fuse_file_info *fi);

	// Called to read a directory entry with attributes 
	void (*readdirplus) (fuse_req_t req, fuse_ino_t ino, size_t size, off_t off,
			 struct fuse_file_info *fi);

	// To copy a range of data from one file to another
	void (*copy_file_range) (fuse_req_t req, fuse_ino_t ino_in,
				 off_t off_in, struct fuse_file_info *fi_in,
				 fuse_ino_t ino_out, off_t off_out,
				 struct fuse_file_info *fi_out, size_t len,
				 int flags);

	// The lseek operation, for specifying new file offsets past the current end of the file.
	void (*lseek) (fuse_req_t req, fuse_ino_t ino, off_t off, int whence,
		       struct fuse_file_info *fi);
};
```

### fuse_session_loop

进入单线程、阻塞式事件循环。如果预先注册了信号处理程序，则可以通过信号终止该循环。

```c++
int fuse_session_loop(struct fuse_session *se);
```

在3fs中使用示例：

```c++
  int ret = -1;
  if (opts.singlethread) {
    ret = fuse_session_loop(d.se);
  } else {
    fuse_loop_cfg_set_clone_fd(config, opts.clone_fd);
    fuse_loop_cfg_set_idle_threads(config, d.maxIdleThreads);
    fuse_loop_cfg_set_max_threads(config, d.maxThreads);
    ret = fuse_session_loop_mt(d.se, config);
  }

  return ret ? 1 : 0;
```

#### fuse_session_unmount

这个函数确保文件系统被卸载。

```c++
void fuse_session_unmount(struct fuse_session *se);
```

```c++
std::stack<std::function<void()>> onStopHooks;
onStopHooks.push([&] { fuse_session_unmount(d.se); });

SCOPE_EXIT {
    while (!onStopHooks.empty()) {
        onStopHooks.top()();
        onStopHooks.pop();
    }
};
```

!!! 注：这里的SCOPE_EXIT宏定义在folly库中哈！

```C++
// https://github.com/facebook/folly/blob/main/folly/ScopeGuard.h
/**
 * Capture code that shall be run when the current scope exits.
 *
 * The code within SCOPE_EXIT's braces shall execute as if the code was in the
 * destructor of an object instantiated at the point of SCOPE_EXIT.
 *
 * Variables used within SCOPE_EXIT are captured by reference.
 *
 * @def SCOPE_EXIT
 */
#define SCOPE_EXIT                                        \
  auto FB_ANONYMOUS_VARIABLE_ODR_SAFE(SCOPE_EXIT_STATE) = \
      ::folly::detail::ScopeGuardOnExit() + [&]() noexcept
```

#### fuse_reply_*

这些类型的函数（例如，fuse_reply_entry、fuse_reply_open 等）用于从用户空间文件系统实现将响应发送回 FUSE 内核模块。每个fuse_reply_*类型的函数对应一种特定类型的可以发送的响应，具体取决于正在执行的操作。

### 如何使用FUSE开发一个简单的文件系统

使用低级 API 开发一个简单的文件系统，支持在目录中创建、读取、写入文件和列出文件。

未完待续...


## 参考

+ https://sh4dy.com/2024/06/24/fuse_01/
+ https://github.com/osxfuse/fuse/blob/master/README.md
