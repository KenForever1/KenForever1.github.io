---
title: 使用aya和c混合编写uprobe程序
date: 2024-09-06
authors: [KenForever1]
categories: 
  - rust
  - ebpf
labels: []
---
使用aya支持CO-RE（编译一次，在不同版本的内核上运行程序）。aya 库（用户空间侧）可以用来加载有 relocations 的 BPF 程序，但用 Rust 语言编写 BPF 程序的 aya-ebpf 库（内核侧）。所以如果无法避免 relocations 等问题时，就需要结合c写内核，rust写用户空间加载。(比如：还不支持 relocations 相关的 bpf_core_read 或编译器的 __builtin_preserve_access_index 函数）。[参考](https://github.com/lx200916/kill_probe_eBPF/issues/1)
<!-- more -->
## 使用C编译CORE EBPF内核程序

要使用c编译CORE EBPF内核程序，依赖头文件。
+ 需要先生成vmlinux.h文件
+ 需要libbpf-dev, #include <btf/btf-helper.h>

### 生成vmlinux.h文件
生成vmlinux.h文件采用bpftool工具，根据vmlinux文件生成vmlinux.h文件。vmlinux文件包含内核的调试信息。
```
$ ls  -alh  /sys/kernel/btf/vmlinux
-r--r--r-- 1 root root 4.9M 8月  13 22:19 /sys/kernel/btf/vmlinux
```
生成vmlinux.h文件：
```
$ bpftool btf dump file /sys/kernel/btf/vmlinux format c > vmlinux.h
```

(1) 如果你的bptfool报错需要apt安装xxx版本的linux-tools，可以如下操作：
```
apt install linux-tools-generic
```
然后将/usr/lib/linux-tools/xxx-generic/设置到环境变量PATH中，或者直接使用绝对路径执行/usr/lib/linux-tools/xxx-generic/bpftool。

(2) 如果报错：可能是内核编译时没有开启BTF功能，可以检查/boot/config-xxx文件，是否有CONFIG_DEBUG_INFO_BTF=y。暂时没有直接解决，我换了一台linux就可以生成了。
```
$ bpftool btf dump file /sys/kernel/btf/vmlinux > vmlinux.h
Error: failed to load BTF from /sys/kernel/btf/vmlinux: Invalid argument
```
### 安装libbpf-dev
```
$ sudo apt install libbpf-dev
```
插一个小曲儿。在linux，如果你想看到安装的包位置，
+ 采用dpkg安装的，使用dpkg -L <package>
+ 采用apt安装的，使用 apt-file list libbpf-dev
```
sudo apt install apt-file
sudo apt-file update
apt-file list libbpf-dev
```

### 编写c EBPF内核程序
uprobe_test.c 文件
```c
// clang-format off
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
// clang-format on

char _license[] SEC("license") = "GPL";

SEC("uprobe/test_text_64_64_reloc")
int test_text_64_64_reloc(struct pt_regs *ctx) {
  return 0;
}
```


```bash
#!/bin/bash
clang -target bpf -D__AARCH64__ -mlittle-endian \
        -I/usr/include -I/home/ken/ \
        -O2 -c $(pwd)/uprobe_test.c -o $(pwd)/ebpf_program.o
```

```bash
$ llvm-objdump -d ebpf_program.o

ebpf_program.o: file format elf64-bpf

Disassembly of section uprobe/test_text_64_64_reloc:

0000000000000000 <test_text_64_64_reloc>:
       0:       b7 00 00 00 00 00 00 00 r0 = 0x0
       1:       95 00 00 00 00 00 00 00 exit
```

### aya加载内核程序

