## 基础应用篇

### docker修改文件，主机无权限

```
1.在容器外用id命令获得自己的用户ID和组ID：

$ id
uid=1001(us001) gid=1001(us001) groups=1001(us001),978(docker)
2.进入容器，将root用户的文件夹，改成自己的，注意后面的 * 表示所有的文件及文件夹

$ chown -R 1001:1001 *
```
### docker 打包镜像

[docker打包镜像](https://learn.lianglianglee.com/%e4%b8%93%e6%a0%8f/Kubernetes%e5%85%a5%e9%97%a8%e5%ae%9e%e6%88%98%e8%af%be/04%20%e5%88%9b%e5%bb%ba%e5%ae%b9%e5%99%a8%e9%95%9c%e5%83%8f%ef%bc%9a%e5%a6%82%e4%bd%95%e7%bc%96%e5%86%99%e6%ad%a3%e7%a1%ae%e3%80%81%e9%ab%98%e6%95%88%e7%9a%84Dockerfile.md)

### docker运行命令示例

```
docker run --gpus all --privileged=true --network=host --workdir=/workspace -v /home/xxx:/workspace -v /dev/shm -v /etc/profile.d:/etc/profile.d -v /etc/ld.so.conf.d:/etc/ld.so.conf.d --tmpfs /dev/shm:exec -e LOCAL_USER_ID=id --name my_dev -it d806c5ed0d21 /bin/bash
```

## Docker进程监控--内存篇

查看内存是否泄露，可以关注：

- memory.stat[rss]
- free 命令的available（包括free字段和page cache字段）
- top命令 进程对应的RES值
- cat /proc/1442/status | grep RSS
- vmstat 2 （每隔2s打印free值）
- **cat /proc/1442/status | grep VmSize (vmpeak：进程历史上使用的最大虚拟内存大小，单位为字节。vmsize：进程当前使用的虚拟内存大小，单位为字节**。)

如果一个进程一直在malloc数据，但是没有使用，且没有free内存，应该关注虚拟内存是否上涨。
判断程序或者容器是否OMM-kill，可以关注：

- journalctl -k 
- docker inspect 中的status

```
docker stats $容器名
docker inspect $容器名
查看内存情况，进入容器memory cgroup目录
find /sys/fs/cgroup/memory/ -name *container_id*
cd /sys/fs/cgroup/memory/container_id

memory.limit_in_bytes 控制组里所有进程可使用内存的最大值
memory.oom_control 当控制组中的进程内存使用达到上限值时，这个参数能够决定会不会触发 OOM Killer

# 每两秒钟，比较memory.usage_in_bytes文件数据变化，-d 比较不同 -n 2 每隔2秒更新一次
watch -n 2 -d cat memory.usage_in_bytes

memory.usage_in_bytes = memory.stat[rss] + memory.stat[cache] + memory.kmem.usage_in_bytes
```
我们通过查看内核的日志，使用用 journalctl -k 命令，或者直接查看日志文件 /var/log/message，我们会发现当容器发生 OOM Kill 的时候，内核会输出下面的这段信息，大致包含下面这三部分的信息：
第一个部分就是容器里每一个进程使用的内存页面数量。在"rss"列里，**"rss'是 Resident Set Size 的缩写，指的就是进程真正在使用的物理内存页面数量。**
从这里我们发现，Page Cache 内存对我们判断容器实际内存使用率的影响，目前 Page Cache 完全就是Linux 内核的一个自动的行为，只要读写磁盘文件，只要有空闲的内存，就会被用作 Page Cache。
所以，判断容器真实的内存使用量，我们不能用 Memory Cgroup 里的memory.usage_in_bytes（rss + Page Cache），而需要用memory.stat里的 rss 值。
这个很像我们用free命令查看节点的可用内存，不能看"free"字段下的值，而要看包括 Page Cache 之后的"available"字段下的值。free 和 available 的区别在于，free 表示系统中真正空闲的内存，而 available 表示系统当前可以用来分配给新进程的内存，包括了部分用作缓存的内存。
使用top命令查看：对应进程的VIRT对应的是malloc等方式申请的虚拟内存，RES和RSS相同，代表的是实际使用的物理页面大小。
参考：
[【精选】Docker 容器内存：我的容器为什么被杀了？_docker oomkilled-CSDN博客](https://blog.csdn.net/qq_34556414/article/details/120507761)
[【精选】09 | Page Cache：为什么我的容器内存使用量总是在临界点?_res pagecache-CSDN博客](https://blog.csdn.net/u011458874/article/details/120839563)