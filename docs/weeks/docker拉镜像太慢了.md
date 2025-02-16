---
comments: true
---
# docker镜像加速方法

在使用docker时，找到的有效加速方法：

docker pull dockerpull.org/openmmlab/lmdeploy:latest-cu12

原文：
https://blog.csdn.net/x1131230123/article/details/143502374

# docker --cap-add=SYS_PTRACE

```
docker run --cap-add=SYS_PTRACE <flags> <stub image>

docker run --cap-add=SYS_PTRACE -p9999:80 --name=nginx-rf-test   
  
docker.io/library/nginx:latest-rfstub
```

To enable runtime observation, SYS_PTRACE Linux kernel capability must be added.

https://man7.org/linux/man-pages/man7/capabilities.7.html

> **CAP_SYS_PTRACE**
  •  Trace arbitrary processes using [ptrace(2)](https://man7.org/linux/man-pages/man2/ptrace.2.html);
  •  apply [get_robust_list(2)](https://man7.org/linux/man-pages/man2/get_robust_list.2.html) to arbitrary processes;
  •  transfer data to or from the memory of arbitrary
	 processes using [process_vm_readv(2)](https://man7.org/linux/man-pages/man2/process_vm_readv.2.html) and
	 [process_vm_writev(2)](https://man7.org/linux/man-pages/man2/process_vm_writev.2.html);
  •  inspect processes using [kcmp(2)](https://man7.org/linux/man-pages/man2/kcmp.2.html).

在Docker中，`--cap-add=SYS_PTRACE` 是一个用于增加容器的能力（capability）选项。`SYS_PTRACE` 是Linux系统中的一种能力，它允许进程使用`ptrace`系统调用来跟踪和控制其他进程。

# git 如何修改commit的author

    git config --global  --list
    git config --global user.name "Your Name"
    git config --global user.email you@example.com

After doing this, you may fix the identity used for this commit with:

    git commit --amend --reset-author