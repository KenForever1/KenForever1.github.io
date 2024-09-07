```
# get bash script dir
dirname "$0"
```

[numToStr](https://github.com/numToStr/dotfiles/blob/master/scripts/.dotscripts/install)

dotfiles 链接管理软件
[stow](https://blog.swineson.me/zh/use-gnu-stow-to-manage-dot-started-config-files-in-your-home-directory/)

[nvim config](https://github.com/nshen)

[zx script](https://github.com/google/zx)

### install fish

```
sudo apt-add-repository ppa:fish-shell/release-3
sudo apt update
sudo apt install fish
```

[从 Zsh 迁移到 Fish，感觉还不错](https://zhuanlan.zhihu.com/p/441328829)


[tmux config](https://zuorn.gitee.io/year/06/20/tmux-conf/#more)



[从Goland转到Neovim](https://jimyag.cn/posts/20d50b9d/#%E5%AE%89%E8%A3%85)


packer 崩溃临时解决方法：
删除packer以及安装的插件，重新安装

```
cd ~/.local/share/nvim/site/pack/packer/start
rm -rf ./*

git clone --depth 1 https://github.com/wbthomason/packer.nvim\
                           ~/.local/share/nvim/site/pack/packer/start/packer.nvim
```