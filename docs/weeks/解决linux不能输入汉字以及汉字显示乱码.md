---
comments: true
---
# 解决linux不能输入汉字以及汉字显示乱码

```bash
locale-gen zh_CN.UTF-8
locale
export LC_ALL="zh_CN.UTF-8"

export LANG="zh_CN.UTF-8"

export LANGUAGE=zh_CN:zh

```

```bash
apt-get install -y locales locales-all
```

# exec env设置环境变量和export的区别是啥？

`exec env LD_LIBRARY_PATH=` 和 `export LD_LIBRARY_PATH=` 是两种在 Unix/Linux 系统中设置环境变量的方法，但它们的作用和使用场景有所不同。以下是两者的区别：

1. **作用范围**：
   - `exec env LD_LIBRARY_PATH=value command`：这是在执行一个特定命令时临时设置环境变量 `LD_LIBRARY_PATH`。此环境变量的设置仅对该命令及其子进程有效，而不会影响当前 shell 或其他进程。`env` 命令用于在一个新的环境中运行指定的命令。
   - `export LD_LIBRARY_PATH=value`：这是在当前 shell 会话中设置环境变量 `LD_LIBRARY_PATH`。这样设置的环境变量对当前 shell 及其所有子进程有效，直到该 shell 会话结束或者手动更改/删除该变量。

2. **使用场景**：
   - 使用 `exec env LD_LIBRARY_PATH=value command` 的场景是当你希望只在执行某个特定命令时修改环境变量，而不对整个 shell 会话或其他命令产生影响。这种方法适用于临时的、一次性的配置。
   - 使用 `export LD_LIBRARY_PATH=value` 适用于你希望在整个 shell 会话期间、多次执行命令时都使用相同的环境变量配置。这种方法适用于持久的、会话范围的配置。

3. **影响范围**：
   - `exec env` 仅影响所执行的命令本身，而 `export` 会影响到当前 shell 会话的所有后续命令。

总之，选择使用哪种方式取决于你希望环境变量的作用范围和持续时间。如果你只需要为单个命令设置环境变量，请使用 `exec env`；如果你需要在整个 shell 会话中都使用同一设置，请使用 `export`。

# pip-env工具的使用

https://www.cnblogs.com/c2soft/articles/17802918.html

install Pytorch
```python
[[source]]
name = "pytorch"
url = "https://download.pytorch.org/whl/"
verify_ssl = true


[packages]
torch = {index = "pytorch",version = "==1.9.0"}
torchvision = {index ="pytorch", version= "==0.10.0"}
torchaudio = {index ="pytorch", version= "==0.9.0"}

[requires]
python_version = "3.7"
```

pip-env install

https://stackoverflow.com/questions/63974588/how-to-install-pytorch-with-pipenv-and-save-it-to-pipfile-and-pipfile-lock/68336073#68336073

virtual-env

https://linux.cn/article-13174-1.html