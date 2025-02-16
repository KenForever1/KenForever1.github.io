---
title: 实现一个neovim插件转换unicode字符到中文显示
date: 2024-12-15
authors: [KenForever1]
categories: 
  - vim
labels: []
comments: true
---

上一篇讲了如何通过nvim-oxi实现一个neovim插件nvim_rotate_chars来轮转字符，这一篇通过采用相关技术实现了一个unicode字符表示转换成汉字显示。这些工具在日常开发和工作中都可以用到，自己实现一个也可以学到新东西练练代码手感。

之前在文章[unicode编码和utf-8转换不同语言实现的差别？以及locale杂谈](./unicode编码和utf-8转换不同语言实现的差别？以及locale杂谈.md)中讲到过由于系统的locale设置不正确，遇到unzip解压出来的中文文件名称都变成了unicde的16进制字符表示。当然也介绍了如何使用python、c++以及rust实现转换。在这里我们将实现移植到了neovim插件中，打开文件一键转换。

<!-- more -->

实现的效果是这样：

![](https://raw.githubusercontent.com/KenForever1/CDN/main/unicode_converter.gif)

## 简介

使用 `nvim-oxi` 编写一个将选中的 `\uXXXX` Unicode 编码转换为汉字字符的 Neovim 插件[KenForever1/nvim_unicode_converter](https://github.com/KenForever1/nvim_unicode_converter)。

## 实现细节

### 核心功能实现

在 `src/lib.rs` 中实现插件的核心功能，通过nvim-oxi提供的api获取选中的当前行，通过正则表达式regex匹配\uxxxx格式，转换替换。

```rust
use nvim_oxi as oxi;
use oxi::{api, Dictionary, Function, Object};
use regex::Regex;

#[oxi::plugin]
fn nvim_unicode_converter() -> Dictionary {
    let convert_unicode = Function::from_fn(|()| {
        // 获取当前选中的文本
        let current_selection = api::get_current_line().unwrap();

        // 正则表达式匹配 \uXXXX 格式
        let re = Regex::new(r"\\u([0-9a-fA-F]{4})").unwrap();

        // 替换匹配的部分
        let converted = re.replace_all(&current_selection, |caps: &regex::Captures| {
            let code_point = u32::from_str_radix(&caps[1], 16).unwrap();
            std::char::from_u32(code_point).unwrap().to_string()
        });

        // 将结果替换到当前行
        let _ = api::set_current_line(converted);
    });

    Dictionary::from_iter([("convert_unicode", Object::from(convert_unicode))])
}
```
toml配置:
```
[package]
name = "nvim_unicode_converter"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
nvim-oxi = { version = "0.5.1", features = ["neovim-0-10"] }
regex = "1.11.1"
```
### neovim插件lua配置

查看runtimepath 获取 rplugin 的路径
```bash
:lua print(vim.inspect(vim.api.nvim_list_runtime_paths()))
```

要在 Neovim 中使用这个插件，需要在你的 `init.lua`（或 `init.vim`）中加载这个 Rust 插件。假设你已经将编译好的库放在 `~/.config/nvim/rplugin/rust` 中：

```lua

-- 绑定命令或按键调用插件
vim.api.nvim_set_keymap('v', '<leader>u', ':lua require("nvim_unicode_converter").convert_unicode()<CR>', { noremap = true, silent = true })
```

```bash
1801\u4e07\u91cc\u957f\u57ce\u6c38\u4e0d\u5012_123445_5_3811-3826s_000004.jpg
```
在 Neovim 中，选中包含 `\uXXXX` 的文本，然后按下你绑定的快捷键（如 `<leader>u`），即可将选中的 Unicode 编码转换为实际字符。

通过这篇文章，您学会了rust和lua配合给neovim实现一个自定义的插件，又多了一个武器技能。
感谢您的阅读！！！开源地址：[KenForever1/nvim_unicode_converter](https://github.com/KenForever1/nvim_unicode_converter)。