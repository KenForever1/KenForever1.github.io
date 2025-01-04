---
title: 如何使用 GNU gettext 实现多语言支持机制
date: 2025-01-04
authors: [KenForever1]
categories: 
  - GNU
labels: [GNU]
---

故事的起因是，我看到了一个项目，[kawaii-gcc](https://github.com/Bill-Haku/kawaii-gcc)。

<!-- more -->

这个项目是干嘛的呢？Make your GCC compiler kawaii.
就是让你的gcc编译器输出或者输出变得："卡哇伊"，可爱！

看一下效果。

![](https://raw.githubusercontent.com/KenForever1/CDN/main/kawayi1.png)

![](https://raw.githubusercontent.com/KenForever1/CDN/main/kawayi.png)

然后我去看了源码，发现呢？src文件夹里面放在三个.po文件，分别对应了英文于中文、日文的翻译。为了弄懂它！于是有了这篇文章。

在当今全球化的软件开发环境中，多语言支持已成为一项不可或缺的功能。无论是面向全球用户的应用程序，还是需要本地化的小型工具，实现多语言支持都能显著提升用户体验。本文将深入探讨如何使用 **GNU gettext** 这一强大的工具链，为你的软件添加多语言支持机制。

---

## 什么是 GNU gettext？

GNU gettext 是一套用于软件国际化和本地化的工具链，广泛应用于开源项目和商业软件中。它的核心思想是将程序中的文本字符串提取出来，存储到独立的翻译文件中，从而实现多语言支持。通过 gettext，开发者可以轻松管理不同语言的翻译，而无需修改源代码。

---

## `.po` 文件：多语言支持的核心

### 什么是 `.po` 文件？
`.po` 文件（Portable Object 文件）是 GNU gettext 的核心组成部分，用于存储源语言和目标语言的翻译对照表。它是一个纯文本文件，包含了需要翻译的字符串及其对应的翻译内容。

### `.po` 文件的作用
`.po` 文件的主要作用是将程序中的文本字符串与目标语言的翻译关联起来。通过编辑 `.po` 文件，开发者可以为软件添加对多种语言的支持，从而让用户能够以自己熟悉的语言使用软件。

---

## `.po` 文件的结构

一个典型的 `.po` 文件由以下几部分组成：

### 文件头信息
文件头信息包含了元数据，如项目名称、语言、字符编码等。这些信息帮助翻译工具和开发者更好地管理翻译文件。

```po
msgid ""
msgstr ""
"Project-Id-Version: My Project 1.0\n"
"Report-Msgid-Bugs-To: \n"
"POT-Creation-Date: 2024-10-01 12:00+0000\n"
"PO-Revision-Date: 2024-10-02 14:00+0000\n"
"Last-Translator: John Doe <john@example.com>\n"
"Language-Team: French <team@example.com>\n"
"Language: fr\n"
"MIME-Version: 1.0\n"
"Content-Type: text/plain; charset=UTF-8\n"
"Content-Transfer-Encoding: 8bit\n"
```

### 翻译条目
每个翻译条目包含两个部分：
- `msgid`：源语言文本（通常是英语）。
- `msgstr`：目标语言的翻译。

```po
msgid "Hello, world!"
msgstr "Bonjour, le monde!"
```

### 注释
注释以 `#` 开头，用于为翻译人员提供上下文或说明。

```po
#: src/main.c:12
# This is a comment for translators
msgid "Save"
msgstr "Enregistrer"
```

---

## 如何使用 GNU gettext 实现多语言支持

### 提取文本
使用 `xgettext` 工具从源代码中提取需要翻译的字符串，生成 `.pot` 文件（Portable Object Template）。

```bash
xgettext -o messages.pot myprogram.c
```

### 创建 `.po` 文件
使用 `msginit` 从 `.pot` 文件生成特定语言的 `.po` 文件。

```bash
msginit -i messages.pot -o fr.po -l fr
```

### 编辑 `.po` 文件
使用文本编辑器或专用工具（如 Poedit）翻译 `msgstr` 部分。

### 编译 `.po` 文件
使用 `msgfmt` 将 `.po` 文件编译为 `.mo` 文件（Machine Object），供程序使用。

```bash
msgfmt -o fr.mo fr.po
```

### 在程序中使用
通过 `gettext` 函数加载 `.mo` 文件，实现多语言支持。

---

## 示例：为 C++ 项目添加中文支持

## **C++ 项目代码示例**

假设我们有一个简单的 C++ 程序，使用 `gettext` 实现国际化：

```cpp
#include <iostream>
#include <libintl.h> // gettext 库
#include <locale.h>  // setlocale 函数

#define _(string) gettext(string) // 定义翻译宏

int main() {
    // 设置语言环境
    setlocale(LC_ALL, "zh_CN.UTF-8");
    bindtextdomain("myapp", "./locales"); // 指定翻译文件目录
    textdomain("myapp"); // 设置文本域

    // 翻译文本
    std::cout << _("Hello, world!") << std::endl;
    std::cout << _("Welcome to my C++ project.") << std::endl;

    return 0;
}
```

---

## **提取翻译字符串**

使用 `xgettext` 工具从 C++ 代码中提取需要翻译的字符串，生成 `.pot` 文件：

```
sudo apt install gettext libgettextpo-dev
```


```bash
mkdir -p locales
xgettext -o locales/myapp.pot -k_ main.cpp --from-code=UTF-8
```

生成的 `myapp.pot` 文件内容如下：

```pot
# SOME DESCRIPTIVE TITLE.
# Copyright (C) YEAR THE PACKAGE'S COPYRIGHT HOLDER
# This file is distributed under the same license as the PACKAGE package.
# FIRST AUTHOR <EMAIL@ADDRESS>, YEAR.
#
#, fuzzy
msgid ""
msgstr ""
"Project-Id-Version: PACKAGE VERSION\n"
"Report-Msgid-Bugs-To: \n"
"POT-Creation-Date: 2025-01-04 17:06+0800\n"
"PO-Revision-Date: YEAR-MO-DA HO:MI+ZONE\n"
"Last-Translator: FULL NAME <EMAIL@ADDRESS>\n"
"Language-Team: LANGUAGE <LL@li.org>\n"
"Language: \n"
"MIME-Version: 1.0\n"
"Content-Type: text/plain; charset=CHARSET\n"
"Content-Transfer-Encoding: 8bit\n"

#: main.cpp:14
msgid "Hello, world!"
msgstr ""

#: main.cpp:15
msgid "Welcome to my C++ project."
msgstr ""
```

---

## **创建中文 `.po` 文件**

使用 `msginit` 从 `.pot` 文件生成中文的 `.po` 文件：

```bash
mkdir -p locales/zh_CN/LC_MESSAGES
msginit -i locales/myapp.pot -o locales/zh_CN/LC_MESSAGES/myapp.po -l zh_CN --locale=zh_CN.UTF-8
```

生成的 `myapp.po` 文件内容如下：

```po
# Chinese translations for PACKAGE package.
# Copyright (C) 2025 THE PACKAGE'S COPYRIGHT HOLDER
# This file is distributed under the same license as the PACKAGE package.
#  <ken@laptop-44oc4fg2.>, 2025.
#
msgid ""
msgstr ""
"Project-Id-Version: PACKAGE VERSION\n"
"Report-Msgid-Bugs-To: \n"
"POT-Creation-Date: 2025-01-04 17:06+0800\n"
"PO-Revision-Date: 2025-01-04 17:07+0800\n"
"Last-Translator:  <ken@laptop-44oc4fg2.>\n"
"Language-Team: Chinese (simplified) <i18n-zh@googlegroups.com>\n"
"Language: zh_CN\n"
"MIME-Version: 1.0\n"
"Content-Type: text/plain; charset=UTF-8\n"
"Content-Transfer-Encoding: 8bit\n"

#: main.cpp:14
msgid "Hello, world!"
msgstr "你好，世界！"

#: main.cpp:15
msgid "Welcome to my C++ project."
msgstr "欢迎来到我的 C++ 项目。"
```

如果上面生成的charset=UTF-8不是UTF-8可以手动改一下。

---

## **编译 `.po` 文件为 `.mo` 文件**

使用 `msgfmt` 将 `.po` 文件编译为 `.mo` 文件，供程序使用：

```bash
msgfmt -o locales/zh_CN/LC_MESSAGES/myapp.mo locales/zh_CN/LC_MESSAGES/myapp.po
```

---

## **项目目录结构**

最终的目录结构如下：

```
myapp/
├── src/
│   └── main.cpp
├── locales/
│   ├── myapp.pot
│   └── zh_CN/
│       └── LC_MESSAGES/
│           ├── myapp.po
│           └── myapp.mo
└── Makefile
```

---

## **编译和运行程序**

### 编译 C++ 程序
确保安装了 `gettext` 库，并使用以下命令编译程序：

```bash
g++ -o myapp main.cpp -lintl
```

### 运行程序
设置语言环境为中文并运行程序：

```bash
export LANG=zh_CN.UTF-8
./myapp
```

输出结果为：

```
你好，世界！
欢迎来到我的 C++ 项目。
```

---

## 示例：为 Rust 项目添加中文支持

当然可以！以下是一个为 Rust 项目编写的 `.po` 文件示例。假设我们有一个简单的 Rust 项目，需要支持英语（默认）和法语（`fr`）两种语言。

---

### **Rust 项目代码示例**
假设我们有以下 Rust 代码，使用了 `gettext` 库来实现国际化：

```rust
use gettextrs::{gettext, LocaleCategory};

fn main() {
    // 设置语言环境
    gettextrs::setlocale(LocaleCategory::LcAll, "zh_CN.UTF-8");
    gettextrs::bindtextdomain("myapp", "./locales").unwrap();
    gettextrs::textdomain("myapp").unwrap();

    // 翻译文本
    println!("{}", gettext("Hello, world!"));
    println!("{}", gettext("Welcome to my Rust project."));
}
```

---

### **提取翻译字符串**
使用 `xgettext` 工具从 Rust 代码中提取需要翻译的字符串，生成 `.pot` 文件：

```bash
xgettext -o locales/myapp.pot -kgettext -kngettext:1,2 src/main.rs
```

生成的 `myapp.pot` 文件内容如下：

```pot
#: src/main.rs:9
msgid "Hello, world!"
msgstr ""

#: src/main.rs:10
msgid "Welcome to my Rust project."
msgstr ""
```

---

### **运行程序**
设置语言环境为法语并运行程序：

```bash
export LANG=zh_CN.UTF-8
cargo run
```

.mo文件中，是通过msgid去匹配可执行程序中的字符串的。所以如果rust和c++中的字符是一样的。那么是可以共用.mo文件来翻译的。同理，其它语言也是。

## 没有成功翻译？

如果你的c++和rust，运行没有翻译效果，可以尝试下面的方法解决：
```
# 通过命令查看系统存在的locale
locale -a

# 如果发现没有zh_CN.utf8
locale-gen zh_CN.UTF-8
```

然后再运行就可以了。

---

为 Rust 项目创建了一个 `.po` 文件，并实现了多语言支持。`.po` 文件是国际化的核心，结合 `gettext` 工具链，可以轻松管理多语言翻译。

---

## 常用工具推荐

+ **Poedit**：一款图形化工具，适合编辑 `.po` 文件。
+ **GNU gettext**：命令行工具集，包括 `xgettext`、`msginit`、`msgfmt` 等。
+ **Weblate**：基于 Web 的翻译平台，支持协作翻译。

通过 GNU gettext，开发者可以轻松实现软件的多语言支持机制。`.po` 文件作为翻译的核心载体，结合强大的工具链，使得国际化变得简单高效。无论是小型工具还是大型应用程序，gettext 都能帮助你快速实现多语言支持，让你的软件走向全球。

---

**参考：**

- [GNU gettext 官方文档](https://www.gnu.org/software/gettext/)
- [Poedit 官网](https://poedit.net/)
- [Weblate 官网](https://weblate.org/)
- [语言文件.po .pot和.mo简介及gettext工具简介](https://blog.csdn.net/zhaominyong/article/details/129385311)
- [从零开始编写一个 .po 文件解析器](https://github.com/huang825172/.po-File-reader-from-scratch)
- [Make your GCC compiler kawaii.](https://github.com/Bill-Haku/kawaii-gcc)