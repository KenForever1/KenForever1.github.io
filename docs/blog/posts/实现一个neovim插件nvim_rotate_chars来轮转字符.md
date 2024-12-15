---
title: 实现一个neovim插件nvim_rotate_chars来轮转代码
date: 2024-12-15
authors: [KenForever1]
categories: 
  - vim
labels: []
---

文章介绍了如果实现一个轮转代码的neovim插件，简单的策略就是a->b, b->c, ..., z->a。
试想一下你的代码经过轮转以后会是怎么样，轮转一个位置、n个位置呢？配上快捷键，在别人看你代码的时候，轮转一下字符，向左转了还能支持向反方向转回来。需要实现如下功能:

+ 支持指定轮转的步数、方向（即向左向右）；
+ 支持unicode字符，比如中文字符的改变；
+ neovim配置快捷键和command命令行输入；
+ 采用rust和lua编写，基于nvim-oxi，和传统rpc不同，直接调用c-api bind。

听起来还有点传统的代码加密算法的意思，旋转加密或者叫轮转机加密。

<!-- more -->

## 简介

[KenForever1/nvim-rotate-chars](https://github.com/KenForever1/nvim-rotate-chars)采用[nvim-oxi](https://github.com/noib3/nvim-oxi)开发的nvim插件，用于旋转选中的文本字符。nvim-oxi提供了Neovim编辑器的Rust API绑定。
传统的插件方法比如：Vimscript or Lua是通过RPC channels和neovim进行通信，采用API绑定的方式。主要是避免了序列化开销，可以在同一个进程中直接调用api绑定回调函数，以及编程时提供了方便的api提示和函数解释。优势的话，相比lua可以使用rust丰富的crate库，还有编译时检查。

> nvim-oxi provides safe and idiomatic Rust bindings to the rich API exposed by the Neovim text editor.The project is mostly intended for plugin authors, although nothing's stopping end users from writing their Neovim configs in Rust.

如何写一个插件，首先需要创建一个lib，lib.rs内容：
```rust
use nvim_oxi::{Dictionary, Function, Object};

#[nvim_oxi::plugin]
fn calc() -> Dictionary {
    let add = Function::from_fn(|(a, b): (i32, i32)| a + b);

    let multiply = Function::from_fn(|(a, b): (i32, i32)| a * b);

    let compute = Function::from_fn(
        |(fun, a, b): (Function<(i32, i32), i32>, i32, i32)| {
            fun.call((a, b)).unwrap()
        },
    );

    Dictionary::from_iter([
        ("add", Object::from(add)),
        ("multiply", Object::from(multiply)),
        ("compute", Object::from(compute)),
    ])
}
```
原理就是然后通过指定cdylib，编译成c ABI的so。neovim就可以加载lua require("so_name"), 就可以调用插件中实现的calc函数了。详细参考：[examples](https://github.com/noib3/nvim-oxi/tree/main/examples)。
```
[lib]
crate-type = ["cdylib"]

[dependencies]
nvim-oxi = "0.3"
```
### nvim-roate-chars 插件

实现的功能，比如：
```
aaa
bbb
ccc
```
向右旋转1个字符，结果为：
```
bbb
ccc
ddd
```

效果：

+ 快捷键触发

![](https://raw.githubusercontent.com/KenForever1/CDN/main/rotate_chars_1.gif)

+ command触发

![](https://raw.githubusercontent.com/KenForever1/CDN/main/rotate_chars_2.gif)


## 使用方法

### 使用方法1

通过快捷键`<leader-r>`进行旋转。根据提示输入需要旋转的方向（true代表向右， false代表向左）和字符数。
例如：visual mode下将选中的行，向右旋转1个字符。
```
Enter direction and rotation number: true 1
```

### 使用方法2

通过Command输入进行旋转。在 Neovim 中进入视觉模式，选中一段文本，然后执行命令：
```
:RotateChars [direction] [steps]
```
visual mode下将选中的行，向右旋转1个字符。
```bash
:'<,'> RotateChars true 1
```
向左旋转1个字符。
```bash
:'<,'> RotateChars false 1
```

## 实现一些细节

### 采用rayon并行高效旋转字符

命令参数：命令 RotateChars 接受两个可选参数：
+ direction：旋转方向，可以是 "left" 或 "right"，默认 "right"。
+ steps：旋转的位数，默认为 1。
字符轮转逻辑：

rotate_char 函数：判断字符是否在 a-z, A-Z, 0-9 范围内，如果是进行轮转，否则保持不变。

```rust
use rayon::prelude::*;

/// 假设 rotate_char 是一个高效的旋转字符函数
#[inline]
fn rotate_char(c: u8, direction: bool, steps: u8) -> u8 {
    let steps = steps % 26;
    // 例如，一个简单的 Caesar cipher 实现（假设输入是 ASCII 字母）
    if (b'a'..=b'z').contains(&c) {
        let base = b'a';
        ((c - base + if direction { steps } else { 26 - steps }) % 26) + base
    } else if (b'A'..=b'Z').contains(&c) {
        let base = b'A';
        ((c - base + if direction { steps } else { 26 - steps }) % 26) + base
    } else {
        c
    }
}

/// 高效旋转内容的函数
pub fn rotate_content(content: &[String], direction: bool, steps: u8) -> Vec<String> {
    content
        .par_iter() // 使用 Rayon 进行并行处理
        .map(|line| {
            let bytes = line.as_bytes();
            let mut rotated = String::with_capacity(bytes.len());
            for &c in bytes {
                rotated.push(rotate_char(c, direction, steps) as char);
            }
            rotated
        })
        .collect()
}
```

### 旋转字符逻辑

通过调用nvim-oxi::api中的函数获取当前Buffer，以及get_lines、set_lines。
```rust
/// 插件的主要逻辑函数
fn rotate_chars(
    buffer: &mut api::Buffer,
    row_range: std::ops::Range<usize>,
    // col_range: std::ops::Range<usize>,
    direction: bool,
    steps: usize,
) -> oxi::Result<()> {
    // 获取选中的范围lines [start_row, end_row)
    let content = buffer.get_lines(row_range.to_owned(), false)?;

    let content_string_list = content
        .enumerate()
        .map(|(_, s)| s.to_string())
        .collect::<Vec<_>>();

    // let rotated_content = rotate_content(&content_string_list, direction, steps as u8);

    let rotated_content = rotate_unicode_content(&content_string_list, direction, steps as u8, false);

    // 替换选中的内容
    buffer.set_lines(row_range, false, rotated_content)?;

    Ok(())
}
```

## neovim中lua配置实现

### lua配置快捷键

```lua
function processString(input)
    local boolStr, numStr = input:match("^(%S+)%s+(%S+)$")
    
    if not boolStr or not numStr then
        error("Input string does not contain exactly two parts.")
    end
    
    local boolValue
    if boolStr:lower() == "true" then
        boolValue = true
    elseif boolStr:lower() == "false" then
        boolValue = false
    else
        error("Invalid boolean string: " .. boolStr)
    end
    
    local numValue = tonumber(numStr)
    if not numValue then
        error("Invalid number string: " .. numStr)
    end
    
    return boolValue, numValue
end

vim.keymap.set('v', '<Leader>r', function()
  -- 请求用户输入数字参数
  vim.ui.input({ prompt = 'Enter direction and rotation number: ' }, function(input)
    print("bool_val:" .. input)
    local bool_val, number_arg = processString(input)

    print("bool_val:" .. tostring(bool_val))
    --print("bool_val:" .. number_arg)

    if number_arg == nil then
      -- 如果输入不是有效数字，则提示错误并退出
      print("Error: Please enter a valid number.")
      return
    end

    -- 获取选区的起始和结束行
    vim.cmd([[ execute "normal! \<ESC>" ]])
    local mode = vim.fn.visualmode()
    local start_line = vim.fn.getpos("'<")[2] - 1
    local end_line = vim.fn.getpos("'>")[2]

    print("start " .. start_line .. "end " .. end_line)
    -- 调用插件的函数，将输入的数字作为参数
    require("nvim_rotate_chars").RotateCharsWithRange(bool_val, number_arg, start_line, end_line)
  end)
end, { noremap = true, silent = true, desc = "Rotate selected characters" })

```

### lua配置快捷键和Command

```lua
vim.api.nvim_create_user_command(
  'RotateChars',
  function(opts)
    -- 检查是否提供了正确数量的参数
    if #opts.fargs < 2 then
      print("Usage: :RotateChars <boolean> <number>")
      return
    end

    -- 将第一个参数解析为布尔值
    local bool_arg = opts.fargs[1] == "true"

    -- 尝试将第二个参数解析为数字
    local num_arg = tonumber(opts.fargs[2])
    if not num_arg then
      print("Error: The second argument must be a number.")
      return
    end

    -- 调用插件的函数
    require("nvim_rotate_chars").RotateChars(bool_arg, num_arg)
  end,
  { range = 2, nargs = "+" }
)

```

通过这篇文章，您学会了rust和lua配合给neovim实现一个自定义的插件，又多了一个武器技能。
感谢您的阅读！！！开源地址：[KenForever1/nvim-rotate-chars](https://github.com/KenForever1/nvim-rotate-chars)。