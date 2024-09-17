---
title: 用两句命令行配置了一个ai编码的vim编辑器
date: 2024-09-17
authors: [KenForever1]
categories: 
  - vim
  - ai
labels: []
---

## 用两句命令行配置了一个ai编码的vim编辑器

<!-- more -->
## vim安装
先看看界面效果：

![AI辅助编码](https://mmbiz.qpic.cn/mmbiz_gif/VYOcrWSWmc2agHEy1SL2yND5MsS1IyxUcZVLiabVZypANZTIkgbJS5MfncVIwIl8ny5YoDPfBicd9CoH6AtgNdFQ/640?wx_fmt=gif&amp;from=appmsg)

![快速查找文件<Space-f>](https://mmbiz.qpic.cn/mmbiz_png/VYOcrWSWmc2agHEy1SL2yND5MsS1IyxUPQNgNORZD9E6ibW7jpc1or5bEGpria57RAichCwpQUw3mQ5jlDmc0nr7w/640?wx_fmt=png&amp;from=appmsg)

![窗口分割](https://mmbiz.qpic.cn/mmbiz_png/VYOcrWSWmc2agHEy1SL2yND5MsS1IyxU1Z0BL8GEuBUY3MDJNG76M0AxbG0WJOeujocuMxLiaBfLp7SmJBeY1cQ/640?wx_fmt=png&amp;from=appmsg)

![支持命令](https://mmbiz.qpic.cn/mmbiz_png/VYOcrWSWmc2agHEy1SL2yND5MsS1IyxUnfWPJLQ7PFBsJMsbFvj7D9gxQhibIpiaFhqiaZukVGIdoRePB4YMjFBwQ/640?wx_fmt=png&amp;from=appmsg)


你想知道我怎么配置的吗？
```bash
sudo mv /usr/bin/vim /usr/bin/vim_old
sudo ln -s /opt/helix-24.07-x86_64-linux/hx /usr/bin/vim
```
​是不是超简单！
对的，没错，我玩了一下[helix](https://helix-editor.com/)，采用rust开发的一款启发于vim和Kakoune的编辑器。对于喜欢vim的功能，但是不想维护插件配置和解决插件冲突的人来说，可以尝试使用helix。
我试着用helix编辑文件，不用于项目开发。项目开发中vscode对我更加方便一些，便捷的远程开发以及可以使用vscode提供的ai编程工具，不用自己折腾。
helix和vim按键基本相同，区别可以参考文章[Migrating from Vim](https://github.com/helix-editor/helix/wiki/Migrating-from-Vim)，界面挺好看的。

## ai编程
为了在helix上尝试一些gpt辅助编程，找到了[helix-gpt](https://github.com/KenForever1/helix-gpt)开源项目。我添加了对通义千问的调用，对一些bug进行了修复。ai编程提示的效果，和模型的相关性特别高，比如收费模型"qwen-max"相对"qwen1.5-1.8b-chat"的提示是又快又准。看看效果：

![AI辅助编码](https://mmbiz.qpic.cn/mmbiz_gif/VYOcrWSWmc2agHEy1SL2yND5MsS1IyxUcZVLiabVZypANZTIkgbJS5MfncVIwIl8ny5YoDPfBicd9CoH6AtgNdFQ/640?wx_fmt=gif&amp;from=appmsg)
## 配置
配置使用helix-gpt，
```bash
vim ~/.config/helix/languages.toml
```
```toml
[language-server.gpt]
command = "helix-gpt"
args = ["--handler", "qianwen", "--logFile", "/home/ken/helix-gpt.log"]

[language-server.ts]
command = "typescript-language-server"
args = ["--stdio"]
language-id = "javascript"

[[language]]
name = "typescript"
language-servers = [
    "gpt",
    "ts"
]
```

附helix简单配置：
```bash
cat ~/.config/helix/config.toml
```

```toml
theme = "onedark"

[editor]
line-number = "relative"
mouse = false
true-color = true

[editor.cursor-shape]
insert = "bar"
normal = "block"
select = "underline"

[editor.file-picker]
hidden = false

[editor.soft-wrap]
enable = true
```
