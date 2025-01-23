
## 安装nvim新版本，比如nvim-0.10.3
可以先查看apt是否支持最新你需要的版本：

```bash
sudo apt install package_name=package_version
# 你如何知道某个软件包有哪些可用的版本？可以使用这个命令：
apt list --all-versions package_name
```

下载发行版本，手动安装：

```bash
proxychains wget https://github.com/neovim/neovim/releases/download/v0.10.3/nvim-linux64.tar.gz

curl -LO https://github.com/neovim/neovim/releases/latest/download/nvim-linux64.tar.gz
sudo rm -rf /opt/nvim
sudo tar -C /opt -xzf nvim-linux64.tar.gz

export PATH="$PATH:/opt/nvim-linux64/bin"
```

如果是fish：
```bash
fish_add_path /opt/nvim-linux64/bin/
```

## 我的基础配置

可以通过管理哈。


### 相关问题解决：

1. 如果提示某个language-server没有安装？

在[nvim-lspconfig](https://github.com/neovim/nvim-lspconfig/blob/master/doc/configs.md)中找对应的语言，根据说明安装language-server即可。

比如[lua_ls](https://github.com/neovim/nvim-lspconfig/blob/master/doc/configs.md#lua_ls)根据[lua_ls install](https://luals.github.io/#neovim-install)安装。

```bash

```

