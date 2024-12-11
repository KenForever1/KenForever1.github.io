---
title: unicode编码和utf-8转换不同语言实现的差别？以及locale杂谈
date: 2024-12-11
authors: [KenForever1]
categories: 
  - 编程
labels: []
---


## 故事开始

故事的背景发生在，我有个图片压缩包，里面有中文文件名，我解压以后变了。

<!-- more -->

正确的文件名：
```
1801万里长城永不倒_123445_5_3811-3826s_000004.jpg
```
解压后的文件名：
```
1801#U4e07#U91cc#U957f#U57ce#U6c38#U4e0d#U5012_123445_5_3811-3826s_000004.jpg
```

通常使用"\u"或"\U"来表示Unicode字符。但是unzip解压文件和文件夹名称为#Uxxxx，其中"x"表示Unicode字符的十六进制编码。

unzip转换中文出现问题，是unzip解压会将其转换成默认的编码（比如ASCII）。而默认编码不支持中文。还有一些情况是终端显示时，按照了locale设置的编码，也不支持中文的正确显示。总之，这种问题就是编码不对应。

如果是转错了，那我们就要用工具、或者自己写个程序转换到正确的编码。
如果是显示错了，就需要设置locale，这个我们后面聊。

## 中文编码解压工具

先看看第一种解压转错了，我们有哪些工具可以用呢？

+ unar工具，它可以自动检测编码，转换成功率很高。

```bash
apt install unar
unar xxx.zip
```
+ 另一种p7zip和encoding conversion，比如
  
```bash
# Traditional Chinese (Taiwan, Singapore, Hong Kong, Macao...)
LANG=zh_HK 7za x filename.zip
cd gamefolder
convmv --notest -f cp950 -t utf8 -r *
```
参考文章：https://wiki.easyrpg.org/unzip-games

一切工作都建立在不顺利的基础上，让我们有了向下探究的机会。由于网络原因安装不了unar，unar手动下载依赖又很多...


## ASCII和Unicode的区别

先了解下ASCII和Unicode区别, ASCII(美国标准，英文字母、数字标点符、控制字符，就是不包含中文和其它国家文字)和Unicode（大而全）是两种字符编码方案，它们都用于在计算机中表示文本。这是它们之间的一些主要区别：

1. 字符集大小：
   - ASCII（American Standard Code for Information Interchange）是基于英语字母的编码方案，只能表示128个字符，包括英文大小写字母、数字0-9、标点符号以及一些控制字符。
   - Unicode是一个全球字符集，可以表示世界上几乎所有语言的字符。Unicode可以表示超过110,000个字符。

2. 编码方式：
   - ASCII使用一个字节（7位，但通常用8位表示）来表示一个字符。
   - Unicode有几种不同的编码方式，包括UTF-8、UTF-16和UTF-32。在UTF-8中，一个字符可以使用1到4个字节来表示。在UTF-16中，一个字符通常使用2个或4个字节来表示。在UTF-32中，所有字符都使用4个字节来表示。

3. 兼容性：
   - ASCII是最早的字符编码标准之一，被广泛应用在各种系统和协议中。
   - Unicode设计时兼容ASCII，也就是说在Unicode中，ASCII的所有字符都与ASCII编码相同。

ASCII的主要优点是简单和广泛支持，但它不能表示非英语字符。Unicode更复杂，但它可以表示世界上几乎所有的字符，所以更适合于国际化的应用。


## UTF8和unicode的关系

> Unicode 解决了编码统一的问题，但没有解决编码存储和解析的问题。UTF-8 则解决了 Unicode 没有解决的问题。
> UTF-8 是一种变长编码，会使用 1 到 4 个字节表示一个字符，类似于哈夫曼树，具体与 Unicode 的映射关系如下 :

| Unicode 范围（十六进制） | UTF-8 编码方式（二进制）            |
| ------------------------ | ----------------------------------- |
| 0000 0000 ~ 0000 007F    | 0xxxxxxx                            |
| 0000 0080 ~ 0000 07FF    | 110xxxxx 10xxxxxx                   |
| 0000 0800 ~ 0000 FFFF    | 1110xxxx 10xxxxxx 10xxxxxx          |
| 0001 0000 ~ 0010 FFFF    | 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx |

详细介绍可以参考：https://sf-zhou.github.io/programming/chinese_encoding.html

## 写个Python脚本转换

c代表一个unicode字符（可能不止一个字节，比如2个字节、3个字节、4个字节），'\u4e00' <= c <= '\u9fff'，在这个范围的utf8就是汉字。

```python


def to_unicode_string(raw_string):
    # ord将字符转换成unicode编码，hex取16进制
    return ''.join(['#U'+hex(ord(c))[2:] if '\u4e00' <= c <= '\u9fff' else c for c in raw_string])


# 采用正则表达式匹配 #Uxxxx 字符，将其转换成汉字的utf-8字符
import re
def split_with_unicode(s):
    return re.split(r'(#[Uu][0-9a-fA-F]{4})', s)

def from_unicode_string(unicode_string):
    return ''.join([chr(int(c[2:], 16)) if c.startswith("#U") else c for c in split_with_unicode(unicode_string) if c])


def test_func(s):
    res = to_unicode_string(s)
    print(res)
    print(from_unicode_string(res))

s = "1801万里长城永不倒_123445_5_3811-3826s_000004.jpg"
test_func(s)

test_func("我爱C++")
```

输出结果就是我们的预期，实现的很不错！！
```
1801#U4e07#U91cc#U957f#U57ce#U6c38#U4e0d#U5012_123445_5_3811-3826s_000004.jpg
1801万里长城永不倒_123445_5_3811-3826s_000004.jpg
```

## 写个rust程序看看

函数和实现逻辑都是和python一样的，区别是regex正则表达式写法不同。
```rust

extern crate regex;
use regex::Regex;

fn to_unicode_string(raw_string: &str) -> String {
    raw_string.chars().map(|c| {
        if '\u{4e00}' <= c && c <= '\u{9fff}' {
            format!("#U{:04x}", c as u32)
        } else {
            c.to_string()
        }
    }).collect()
}

fn split_with_unicode(input: &str) -> Vec<&str> {
    let re = Regex::new(r"#U[0-9a-fA-F]{4}").unwrap();
    let mut result = Vec::new();
    let mut last_end = 0;

    for mat in re.find_iter(input) {
        // Add the portion before the current match to the result
        if mat.start() > last_end {
            result.push(&input[last_end..mat.start()]);
        }
        // Add the matched pattern to the result
        result.push(&input[mat.start()..mat.end()]);
        last_end = mat.end();
    }

    // Add the remaining portion of the string, if any
    if last_end < input.len() {
        result.push(&input[last_end..]);
    }

    return result

}

fn from_unicode_string(unicode_string: &str) -> String {
    println!("{}", unicode_string);
    split_with_unicode(unicode_string).iter().map(|&c| {
        if c.starts_with("#U") {
            char::from_u32(u32::from_str_radix(&c[2..], 16).unwrap()).unwrap().to_string()
        } else {
            c.to_string()
        }
    }).collect()
}

fn test_func(s: &str) {
    let res = to_unicode_string(s);
    println!("{}", res);
    println!("{}", from_unicode_string(&res));
}

fn main() {
    let s = "1801万里长城永不倒_123445_5_3811-3826s_000004.jpg";
    println!("{}", s);
    test_func(s);
}

// 输出结果：
// 1801万里长城永不倒_123445_5_3811-3826s_000004.jpg
// 1801#U4e07#U91cc#U957f#U57ce#U6c38#U4e0d#U5012_123445_5_3811-3826s_000004.jpg
// 1801#U4e07#U91cc#U957f#U57ce#U6c38#U4e0d#U5012_123445_5_3811-3826s_000004.jpg
// 1801万里长城永不倒_123445_5_3811-3826s_000004.jpg
```

## 写个c++看看

### 直接翻译python到c++，错误版本

直接翻译python逻辑可以吗？这个是错误的，我们来看一看。
因为c++遍历字符串就是一个char，std::string、char只支持单字符。
错误写法：
```c++
#include <iostream>
#include <string>
#include <regex>
#include <sstream>
#include <iomanip>
#include <vector>

std::string to_unicode_string(const std::string& raw_string) {
    std::ostringstream oss;
    for (const auto& c : raw_string) {
        std::cout << "c : " << c << std::endl;
        if (c >= 0x4E00 && c <= 0x9FFF) {
            oss << "#U" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c);
        } else {
            oss << c;
        }
    }
    return oss.str();
}

std::vector<std::string> split_with_unicode(const std::string& s) {
    std::regex re(R"((#[Uu][0-9a-fA-F]{4}))");
    std::sregex_token_iterator iter(s.begin(), s.end(), re, {-1, 0});
    std::sregex_token_iterator end;
    std::vector<std::string> result(iter, end);
    return result;
}

std::string from_unicode_string(const std::string& unicode_string) {
    std::ostringstream oss;
    auto parts = split_with_unicode(unicode_string);
    for (const auto& part : parts) {
        if (part.starts_with("#U") || part.starts_with("#u")) {
            int code;
            std::istringstream(part.substr(2)) >> std::hex >> code;
            oss << static_cast<char>(code);
        } else {
            oss << part;
        }
    }
    return oss.str();
}

void test_func(const std::string& s) {
    auto res = to_unicode_string(s);
    std::cout << res << std::endl;
    std::cout << from_unicode_string(res) << std::endl;
}

int main() {
    std::string s = "1801万里长城永不倒_123445_5_3811-3826s_000004.jpg";
    test_func(s);
    return 0;
}
```

输出结果：
```
1801万里长城永不倒_123445_5_3811-3826s_000004.jpg
1801万里长城永不倒_123445_5_3811-3826s_000004.jpg
```
由于c是单字符，因此不会走下面的逻辑，
```c++
 if (c >= 0x4E00 && c <= 0x9FFF) {
            oss << "#U" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c);
        }
```


编译命令：
```bash
g++ test.cpp  -std=gnu++20
```

### 手动实现unicode和utf-8转换逻辑

根据前面的Unicode和utf-8关系介绍，手动实现unicode和utf-8转换逻辑, 一起看看utf8和unicode如何互转：

```cpp
#include <iostream>
#include <string>
#include <regex>
#include <sstream>
#include <iomanip>
#include <vector>

// Helper function to decode UTF-8 to a single Unicode code point
uint32_t utf8_to_codepoint(const char*& it, const char* end) {
    uint32_t codepoint = 0;
    unsigned char ch = *it;

    if (ch < 0x80) {
        codepoint = ch;
    } else if ((ch & 0xE0) == 0xC0) {
        codepoint = ch & 0x1F;
    } else if ((ch & 0xF0) == 0xE0) {
        codepoint = ch & 0x0F;
    } else if ((ch & 0xF8) == 0xF0) {
        codepoint = ch & 0x07;
    }

    ++it;
    while (it != end && (*it & 0xC0) == 0x80) {
        codepoint = (codepoint << 6) | (*it & 0x3F);
        ++it;
    }

    return codepoint;
}

std::string to_unicode_string(const std::string& raw_string) {
    std::ostringstream oss;
    const char* it = raw_string.c_str();
    const char* end = it + raw_string.size();

    while (it < end) {
        const char* start = it;
        uint32_t codepoint = utf8_to_codepoint(it, end);

        if (codepoint >= 0x4E00 && codepoint <= 0x9FFF) {
            oss << "#U" << std::hex << std::setw(4) << std::setfill('0') << codepoint;
        } else {
            oss.write(start, it - start);
        }
    }

    return oss.str();
}

std::vector<std::string> split_with_unicode(const std::string& s) {
    std::regex re(R"((#[Uu][0-9a-fA-F]{4}))");
    std::sregex_token_iterator iter(s.begin(), s.end(), re, {-1, 0});
    std::sregex_token_iterator end;
    std::vector<std::string> result(iter, end);
    return result;
}

std::string from_unicode_string(const std::string& unicode_string) {
    std::ostringstream oss;
    auto parts = split_with_unicode(unicode_string);
    for (const auto& part : parts) {
        if (part.starts_with("#U") || part.starts_with("#u")) {
            int code;
            std::istringstream(part.substr(2)) >> std::hex >> code;
            if (code < 0x80) {
                oss << static_cast<char>(code);
            } else {
                // Encode code into UTF-8
                if (code < 0x800) {
                    oss << static_cast<char>((code >> 6) | 0xC0);
                    oss << static_cast<char>((code & 0x3F) | 0x80);
                } else if (code < 0x10000) {
                    oss << static_cast<char>((code >> 12) | 0xE0);
                    oss << static_cast<char>(((code >> 6) & 0x3F) | 0x80);
                    oss << static_cast<char>((code & 0x3F) | 0x80);
                }
            }
        } else {
            oss << part;
        }
    }
    return oss.str();
}

void test_func(const std::string& s) {
    auto res = to_unicode_string(s);
    std::cout << res << std::endl;
    std::cout << from_unicode_string(res) << std::endl;
}

int main() {
    std::string s = "1801万里ggg长城永不倒_123445_5_3811-3826s_000004.jpg";
    test_func(s);
    return 0;
}
```

### wstring实现

wstring和wchar是支持多字符的，和平台相关，也许2个字节，也许4个字节。

> C++ 大力出奇迹，提出使用 wchar_t 解决多字节编码带来的问题。既然多字节计算长度不准确，那就直接使用一个类型代替多字节。目前 UTF-8、Unicode 等编码至多需要 4 个字节，那就直接定义 wchar_t 为 4 个字节，所有问题都可以迎刃而解了。是的，这不是开玩笑。wchar_t 具体定义为 2 个字节，还是 4 个字节并没有明确规定，视平台而定。

```cpp
#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#include <vector>
#include <regex>
#include <codecvt>

// Function to convert a wide string to a Unicode representation
std::wstring to_unicode_wstring(const std::wstring& raw_wstring) {
    std::wostringstream woss;

    for (wchar_t wc : raw_wstring) {
        if (wc >= 0x4E00 && wc <= 0x9FFF) {
            woss << L"#U" << std::hex << std::setw(4) << std::setfill(L'0') << static_cast<int>(wc);
            woss << std::dec; 
        } else {
            woss << wc;
        }
    }

    return woss.str();
}

// Helper function to split a wide string based on Unicode markers
std::vector<std::wstring> split_with_unicode(const std::wstring& s) {
    std::wregex re(L"(#[Uu][0-9a-fA-F]{4})");
    std::wsregex_token_iterator iter(s.begin(), s.end(), re, {-1, 0});
    std::wsregex_token_iterator end;
    std::vector<std::wstring> result(iter, end);
    for(auto item: result ){
    std::wcout << "split : " << item << std::endl;

    }
    return result;
}

// Function to convert a Unicode representation back to a wide string
std::wstring from_unicode_wstring(const std::wstring& unicode_wstring) {
    std::wostringstream woss;
    auto parts = split_with_unicode(unicode_wstring);
    for (const auto& part : parts) {
        if (part.starts_with(L"#U") || part.starts_with(L"#u")) {
            int code;
            std::wistringstream(part.substr(2)) >> std::hex >> code;
            woss << static_cast<wchar_t>(code);
        } else {
            woss << part;
        }
    }
    return woss.str();
}

// Function to test the conversion process
void test_func(const std::wstring& s) {
    auto res = to_unicode_wstring(s);
    // const std::locale utf8( std::locale(), new std::codecvt_utf8<wchar_t> );
    // std::wcout.imbue(utf8);
    std::wcout << L"Encoded: " << res << std::endl;
    // Encoded: 1801#U4,e07#U9,1cc#U9,57f#U5,7ce#U6,c38#U4,e0d#U5,012_123445_5_3811-3826s_000004.jpg???
    
    // Convert wstring to string for display (requires a codecvt facet)
    // std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    // std::string utf8_string = converter.to_bytes(res);

    // Output the converted UTF-8 string
    // std::cout << "UTF-8 String: " << utf8_string << std::endl;

    res = L"1801#U4e07#U91cc#U957f#U57ce#U6c38#U4e0d#U5012_123445_5_3811-3826s_000004.jpg";
    std::wcout << L"Decoded: " << from_unicode_wstring(res) << std::endl;
}

int main() {
    std::locale::global(std::locale(""));  // Set the locale to support wide characters
    std::wstring s = L"1801万里长城永不倒_123445_5_3811-3826s_000004.jpg";
    test_func(s);
    return 0;
}
```

#### wstring 和 string相互转换

```c++
#include <iostream>
#include <string>
#include <locale>
#include <codecvt>

// Convert std::string to std::wstring
std::wstring string_to_wstring(const std::string& str) {
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    return converter.from_bytes(str);
}

// Convert std::wstring to std::string
std::string wstring_to_string(const std::wstring& wstr) {
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    return converter.to_bytes(wstr);
}

int main() {

    std::string narrow = "Hello, world!";
    std::wstring wide = string_to_wstring(narrow);
    std::wcout << L"Converted to wstring: " << wide << std::endl;

    
    std::wstring wstr = L"你好，世界！";
    std::string str = wstring_to_string(wstr);
    std::cout << "Converted to string: " << str << std::endl;

    return 0;
}
```

#### 一些注意的问题

##### std::wcout和std::cout输出顺序的影响

你会发现先调用输出std::wcout后，调用std::cout不会输出结果。但是反过来的调用顺序就可以。

参考这里有个回答：
[unexpected_behavior_of_stdcout_following_stdwcout](https://www.reddit.com/r/cpp_questions/comments/190mzf0/unexpected_behavior_of_stdcout_following_stdwcout/)

意思是说，底层c实现在输出的时候，需要确定宽度。

> The underlying C streams have a concept of being either wide-oriented or narrow-oriented. You can't use wide output functions on a narrow stream and vice-versa.

但是没有说明为啥反过来可以，我这里没有深究，留待以后，只是提出遇到的这个现象。

#### locale的设置会影响to_unicode_wstring函数的结果


你可以将一个流与特定的`locale`关联，以便自动处理小数点、千位分隔符、货币符号等的格式。

比如：
如何设置localeI("en_US.utf-8"), 输出的每个hex 16进制编码有“,”符号分割。

```bash
std::locale::global(std::locale("en_US.utf-8"));
输出：1801#U4,e07#U91cc#U9,57f#U5,7ce#U6,c38#U4,e0d#U5,012_123445_5_3811-3826s_000004.jpg
```

如果设置localeI("C"), 输出的每个hex 16进制编码不会有符号分割，按照c语言规范。

```bash
std::locale::global(std::locale("C"));
输出：1801#U4e07#U91cc#U957f#U57ce#U6c38#U4e0d#U5012_123445_5_3811-3826s_000004.jpg
```

## locale是个啥？

### linux locale命令

在Linux系统中，`locale`是一个与地区和语言环境设定相关的工具和设置。它用于定义程序如何处理与语言相关的任务，如字符编码、数字格式、日期和时间格式、货币符号等。`locale`设置对于开发国际化和本地化的应用程序非常重要。

#### 主要的`locale`环境变量

- **`LANG`**: 定义了默认的地域和语言环境设定。如果没有其他特定的`locale`变量被设置，系统会使用这个。
- **`LC_ALL`**: 覆盖所有其他`LC_*`变量的优先设置，用于强制使用特定的`locale`。通常用于测试，不建议在普通环境中设置。
- **`LC_COLLATE`**: 控制字符串的排序。
- **`LC_CTYPE`**: 控制字符分类和字符处理（例如，大小写转换）。
- **`LC_MESSAGES`**: 控制系统消息的语言。
- **`LC_MONETARY`**: 控制货币格式。
- **`LC_NUMERIC`**: 控制数字格式（例如，小数点符号）。
- **`LC_TIME`**: 控制日期和时间格式。

#### 使用`locale`命令

- **查看当前locale设置**:
  ```bash
  locale
  ```

- **查看支持的locale**:
  ```bash
  locale -a
  ```

- **设置locale**:
  你可以在终端会话中临时设置locale，例如：
  ```bash
  export LANG=en_US.UTF-8
  ```

- **locale未设置或不正确**: 可能会导致字符显示错误（如乱码）或者程序无法正常运行。确保正确生成并设置所需的locale。

- **缺少特定locale**: 使用`locale-gen`命令生成并使其可用。例如，生成`en_US.UTF-8`可以通过以下命令：
```bash
sudo locale-gen en_US.UTF-8
```

### c++ std::locale

在C++中，`std::locale`是一个类，用于封装与地区和语言设定相关的信息。这包括日期和时间格式、货币符号、数字格式、字符分类和字符转换等。这些信息对于国际化（i18n）和本地化（l10n）是非常重要的，因为它们允许程序根据用户的地区或语言设定来调整输出和输入格式。

`std::locale`类提供了一种机制，可以将程序的某些部分（如输入输出流）设置为使用特定的地方化规则。例如，你可以将一个流与特定的`locale`关联，以便自动处理小数点、千位分隔符、货币符号等的格式。

以下是一些常见的用法：
```c++
#include <locale>
#include <iostream>
int main()
{
    // 创建一个locale对象
    std::locale loc("C.UTF-8");
    // 将locale应用于流
    std::cout.imbue(loc);
    // 获取当前全局locale
    std::locale currentLocale = std::locale();
    // 使用locale进行字符转换
    char c = 'a';
    if (std::isalpha(c, loc))
    {
        std::cout << c << " is alphabetic in the given locale." << std::endl;
    }
}
```