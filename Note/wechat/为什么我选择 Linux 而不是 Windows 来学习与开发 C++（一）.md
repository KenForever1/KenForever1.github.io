---
title: 为什么我选择 Linux 而不是 Windows 来学习与开发 C++（一）
date: 2025-01-04
authors: [KenForever1]
categories: 
  - cpp
labels: [cpp]
---


故事的起因是因为，我在一些群里看到同学们、朋友们，很多都是大学生朋友。问：我的这个c++代码怎么在**Microsoft Visual C++ 6.0**（VC++ 6.0）上跑不起来啊？

比如下面这段代码是在**Microsoft Visual C++ 6.0**上跑不过的，会报错：range based for 不支持，因为这个语法是c++11才出来的，也就是2011年。但是VC++ 6.0只支持c++98的。

<!-- more -->

```c++
#include <iostream>
#include <vector>
int main(){
  std::vector<int> arr{1, 2, 3};
  for(auto item:arr){
    std::cout << item << std::endl;
  }
}
```

```bash
warning: range-based for loop is a C++11 extension [-Wc++11-extensions]
for(auto item:arr)
```

所以，很多问题，可能不是你的程序有问题，但是你却花费了很多时间和心智在和环境做斗争中。

注:devc++目前还有下载官网，vc++6.0我没有看到下载官网。

![](https://raw.githubusercontent.com/KenForever1/CDN/main/vc6.png)

![](https://raw.githubusercontent.com/KenForever1/CDN/main/devc++.png)


作为对比，我插入两张图，clang和gcc的官网，可以看到更新日期都是2024年。

![](https://raw.githubusercontent.com/KenForever1/CDN/main/gcc.png)

![](https://raw.githubusercontent.com/KenForever1/CDN/main/clang.png)

所以有了下面的一系列文章，

- [为什么我选择 Linux 而不是 Windows 来学习与开发 C++（一）](./为什么我选择%20Linux%20而不是%20Windows%20来学习与开发%20C++（一）.md)
- [C++ 编译器与工具链解析：MSVC、MinGW、Cygwin、Clang 和 GCC（二）](./C++%20编译器与工具链解析：MSVC、MinGW、Cygwin、Clang%20和%20GCC（二）.md)
- [现代 C++ 开发：为什么选择 Linux或WSL和Clang GCC 而非 VC++ 6.0 和 Visual Studio（三）](./现代%20C++%20开发：为什么选择%20Linux或WSL和Clang%20GCC%20而非%20VC++%206.0%20和%20Visual%20Studio（三）.md)

在学校推荐的课程中，一般都会推荐两款经典的工具，**Microsoft Visual C++ 6.0**（VC++ 6.0）和**Dev c++**。这两款经典软件都是1998年推出的一款windows平台的C++编译器。但是如今都2025年了啊。

在 C++ 的学习与开发中，选择一个合适的操作系统和开发环境至关重要。尽管 Windows 作为全球最流行的操作系统之一，拥有广泛的用户基础和丰富的软件生态，但在 C++ 开发领域，**Linux** 却以其独特的优势成为许多开发者的首选。本文将从多个维度深入探讨为什么 Linux 更适合 C++ 的学习与开发，并与 Windows 进行对比，帮助你做出更明智的选择。
<!-- more -->
---

## 开发环境一致性：无缝衔接生产环境

在开发和学习中可以使用: WSL2 或者 虚拟机 或者 远程linux服务器。

一般企业中都是远程开发机，或者docker容器。操作系统一般都是centos或者ubuntu等比较常用。我在我的windows主机上，常用的就是WSL2。安装的ubuntu22.04环境。安装gcc 、clang只需要执行命令:

```bash
apt install gcc clang
```

当然还可以安装一个vmware等虚拟机，再装上一个linux系统用于开发。这也是我以前用过的方式，相比WSL2安装会麻烦一些，使用笔记本上的显卡也会麻烦一些。

![](https://raw.githubusercontent.com/KenForever1/CDN/main/wsl2.png)

### Linux
- **主流开发平台**：Linux 是 C++ 开发的主流平台，尤其是在服务器、嵌入式系统和高性能计算领域。许多开源工具链和库在 Linux 上更容易配置和使用。
- **一致性**：开发环境与生产环境高度一致。大多数企业的服务器运行 Linux，因此在 Linux 上开发的代码可以直接部署到生产环境，减少兼容性问题。

### Windows
- **依赖 Windows API**：Windows 上的开发环境通常针对 Windows 平台。
- **环境差异**：开发环境与生产环境（通常是 Linux）不一致，可能导致代码在部署时出现兼容性问题，增加调试成本。

---

## 编译器与工具链：现代与高效的结合

### Linux
- **默认编译器**：Linux 默认使用 **GCC** 和 **Clang**，它们支持最新的 C++ 标准（如 C++17/20以及23、26等），编译速度快，生成的代码性能高。
  - **GCC**：GNU 编译器集合，支持多种编程语言，是 Linux 系统的默认编译器。
  - **Clang**：基于 LLVM 的编译器，以模块化设计和友好的错误提示著称。
- **调试工具**：GDB 和 LLDB 是强大的调试工具，适合复杂项目的调试。
- **构建工具**：支持 CMake、Makefile 等现代构建工具，方便管理大型项目。

### Windows
- **默认编译器**：Windows 上通常使用 **MSVC**（Microsoft Visual C++），它对 Windows API 支持较好，但对 C++ 标准的支持可能滞后。
- **其他编译器**：可以通过 MinGW 或 Cygwin 使用 GCC，但配置较复杂。
- **调试工具**：Visual Studio 的调试器非常强大。

对c++标准支持滞后可以从cppreference的看出：

![](https://raw.githubusercontent.com/KenForever1/CDN/main/cpp_compare.png)

---

## 开源生态系统：丰富的资源与社区支持

### Linux
- **丰富的开源库**：Linux 上有大量高质量的开源库（如 Boost、Eigen、OpenCV），适合算法开发、科学计算和系统编程。
- **社区支持**：Linux 社区庞大且活跃，遇到问题时可以快速找到解决方案。开源生态系统的繁荣为开发者提供了无限的可能性。

### Windows
- **商业软件为主**：Windows 的生态系统以商业软件为主，开源工具相对较少。
- **依赖第三方工具**：需要手动下载和配置库，或使用第三方工具（如 vcpkg、Conan），增加了开发复杂度。

---

## 性能与效率：释放硬件的全部潜力

### Linux
- **高性能计算**：Linux 内核优化更好，适合高性能计算和资源密集型任务。许多高性能计算集群和超级计算机都运行 Linux。
- **轻量级**：Linux 系统资源占用少，可以充分利用硬件性能，尤其是在低配置设备上表现优异。

### Windows
- **资源占用高**：Windows 系统本身资源占用较高，Visual Studio 和 MSVC 的编译速度较慢，尤其是在大型项目中。
- **性能较低**：在相同硬件条件下，Linux 上的编译和运行速度通常更快。

---

## 跨平台开发性更好

### Linux
- **代码可移植性**：在 Linux 上开发的代码更容易移植到其他平台（如 macOS、嵌入式系统）。
- **容器化支持**：Docker 等容器技术在 Linux 上运行效率最高，适合微服务和分布式系统开发。

### Windows
- **跨平台支持较弱**：Windows 上的开发环境通常针对 Windows 平台。

---

## 学习价值：深入理解系统与工具链

### Linux
- **深入理解系统**：Linux 提供了更多接触操作系统底层的机会，有助于深入理解系统原理，如内存管理、进程调度和文件系统。
- **命令行技能**：Linux 命令行工具的使用是开发者的一项重要技能，掌握这些工具可以显著提升开发效率。

### Windows
- **图形化界面**：Windows 的图形化界面适合初学者快速上手，但可能限制对底层工具链的理解。

---

## 成本：开源与免费的优势

### Linux
- **免费开源**：Linux 是免费的，可以节省操作系统和工具链的授权费用。
- **硬件成本低**：Linux 可以在低配置硬件上运行，降低硬件成本。

### Windows
- **商业软件**：Windows 和 Visual Studio 是商业软件，需要购买授权，增加了开发成本。

---

## 企业中的应用：Linux 的主导地位

在企业中，尤其是涉及 **算法** 和 **C++ 开发** 的领域，**Linux** 是首选的开发平台。以下是企业选择 Linux 进行 C++ 开发的主要原因：

1. **性能与效率**：Linux 内核优化更好，适合高性能计算和资源密集型任务。
2. **开发工具链完善**：GCC 和 Clang 支持最新的 C++ 标准，编译速度快，生成的代码性能高。
3. **开源生态系统**：Linux 上有大量高质量的开源库，适合算法开发和科学计算。
4. **跨平台开发**：在 Linux 上开发的代码更容易移植到其他平台。
5. **服务器环境一致性**：大多数企业的服务器运行 Linux，开发环境与生产环境一致，减少部署问题。
6. **成本**：Linux 是免费的，企业可以节省操作系统和工具链的授权费用。

---

## Linux 是 C++ 开发的理想选择

- **Linux 的优势**：
  - 开发环境一致性高，工具链完善，开源生态系统强大，性能优越，适合跨平台开发，学习价值高，成本低。

对于学习和开发 C++ 的初学者和开发者来说，**Linux** 是更优的选择。它不仅提供了强大的工具链和开源生态系统，还能帮助你深入理解系统原理，掌握跨平台开发技能。如果你主要开发 Windows 应用，Windows + Visual Studio 是不错的选择，但如果你希望更高效地学习 C++ 和进行跨平台开发，Linux 无疑是更好的平台。

---

### 对比表格：Linux 与 Windows 开发 C++ 的对比

| 特性                | Linux                              | Windows                            |
|---------------------|------------------------------------|------------------------------------|
| **开发环境一致性**  | 开发与生产环境一致                 | 开发环境与生产环境不一致           |
| **编译器**          | GCC、Clang（支持最新 C++ 标准）    | MSVC（Windows 专用）、MinGW        |
| **调试工具**        | GDB、LLDB（功能强大）              | Visual Studio 调试器（Windows 专用）|
| **构建工具**        | CMake、Makefile（现代构建工具）    | 配置过程更加繁琐一些           |
| **学习价值**        | 深入理解系统，掌握命令行技能       | 图形化界面易上手，但限制底层理解   |

---

当然，如果你本身就是windows应用开发者，或者你已经熟悉了windows上进行开发那么windows会更加适合你。如果你是新手，希望更快投入语言学习本身，避免和环境作斗争，可以试试linux。
感谢您的阅读！！！