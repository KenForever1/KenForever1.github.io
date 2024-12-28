
# python调用图

对于包含多个 `.py` 文件的 Python 项目，你可以使用以下工具生成整个项目的调用关系图。这些工具能够处理多个文件，并生成更全面的调用关系图。

### 0.1 使用 pyan

`pyan` 是一个可以从多个 Python 源文件生成调用图的工具。它可以处理整个项目的所有 `.py` 文件。

1. **安装 pyan**:
   ```bash
   pip install pyan3
   ```

2. **生成调用关系图**:
   假设你的项目目录结构如下：
   ```
   my_project/
   ├── module1.py
   ├── module2.py
   └── module3.py
   ```

   你可以运行以下命令来生成调用图：
   ```bash
   pyan my_project/*.py --dot --colored --grouped > output.dot
   ```

3. **可视化调用关系图**:
   使用 Graphviz 将 `.dot` 文件转换为图像：
   ```bash
   dot -Tpng output.dot -o output.png
   ```

### 0.2 使用 Doxygen

`Doxygen` 可以为整个 Python 项目生成文档和调用图。

1. **安装 Doxygen**:
   - 可以通过系统的包管理器安装，或者从官方网站下载。
   - 例如，在 Ubuntu 上：
     ```bash
     sudo apt-get install doxygen
     ```

2. **生成配置文件**:
   在项目根目录运行以下命令生成 `Doxyfile` 配置文件：
   ```bash
   doxygen -g
   ```

3. **编辑 Doxyfile**:
   打开 `Doxyfile`，设置以下选项以启用 Python 支持和调用图生成：
   ```plaintext
   FILE_PATTERNS = *.py
   EXTRACT_ALL = YES
   HAVE_DOT = YES
   CALL_GRAPH = YES
   CALLER_GRAPH = YES
   ```

4. **生成文档和图表**:
   在项目根目录运行：
   ```bash
   doxygen Doxyfile
   ```

   这将在 `html` 目录中生成文档，其中包含调用图。

### 0.3 使用 Graphviz 结合其他工具

对于更复杂的项目，可以编写自定义脚本，结合 `ast` 模块解析 Python 源码，然后生成 `.dot` 文件并使用 Graphviz 可视化。


https://github.com/chanhx/crabviz

# python import

  

## 1 1

在python的同一个程序中，使用多个版本的package，如何避免冲突？

比如导入的包叫timm，但是存在两个版本，文件夹命名都是timm，如果仅仅依靠修改sys.path，导入包，然后再pop，不能保证搜索到正确的版本。因为python包导入会使用第一次导入缓存的版本。

  

```python

import sys

  

sys.path.insert(0, 'py_crate_example/v2/')

print(__file__,sys.path)

from timm_v2 import infer

sys.path.pop(0)

  

import t1

  

import t2

  

infer.infer()

  

t1.hello()

  

t2.hello()

```

  

正确的做法：采用不同的名字，比如timm_v1和timm_v2。

```

├── main.py

├── readme.md

├── t1.py

├── t2.py

├── v1

│ └── timm

│ ├── __init__.py

│ └── infer.py

└── v2

└── timm_v2

├── __init__.py

└── infer.py

```

  

这只是简单的情况，如果infer.py中用了import timm.xxx，那么在v2中就不能使用import timm，因为改成了timm_v2。

  

这个时候就只能用不同的python子进程，单独跑了。

  

动态加载也不行的。

```

动态加载库：如果你非要在同一进程中使用不同版本的库，可以尝试动态加载库。虽然这不保证所有库都能正常工作，但对于某些库可能有效。

  

将不同版本的库安装到不同的路径，然后使用importlib动态加载。

import sys

import importlib.util

  

# 加载第一个版本

spec = importlib.util.spec_from_file_location(

"timm_v1", "/path/to/timm_v1/timm/__init__.py")

timm_v1 = importlib.util.module_from_spec(spec)

sys.modules["timm_v1"] = timm_v1

spec.loader.exec_module(timm_v1)

  

# 加载第二个版本

spec = importlib.util.spec_from_file_location(

"timm_v2", "/path/to/timm_v2/timm/__init__.py")

timm_v2 = importlib.util.module_from_spec(spec)

sys.modules["timm_v2"] = timm_v2

spec.loader.exec_module(timm_v2)

  

# 使用不同版本的库

timm_v1.some_function()

timm_v2.some_function()

```

  

## 2 2

  

python中，脚本执行sys.path.append(‘./’)和sys.path.insert(0, ‘./’)后，不会改变系统的环境变量path。

可以在执行脚本后，通过python -c "import sys;print(sys.path)"查看。

  

`sys.path` 是 Python 在运行时用于确定模块搜索路径的列表。这些路径在 Python 启动时初始化，通常包括脚本所在的目录、Python 标准库目录以及其他依赖项的目录。需要注意的是，`sys.path` 的修改在 Python 进程结束后不会持久化。因此，如果你想要永久性地更改 `sys.path`，需要从以下几个方法中选择合适的方法：

  

### 2.1 方法 1：修改环境变量 `PYTHONPATH`

  

`PYTHONPATH` 是一个环境变量，你可以将其设置为包含多个路径的列表（路径用操作系统特定的分隔符分隔，如 Unix/Linux 上的冒号 `:` 或 Windows 上的分号 `;`）。这些路径在 Python 启动时会被添加到 `sys.path`。

  

编辑你的 shell 配置文件（如 `~/.bashrc`、`~/.bash_profile`、`~/.zshrc` 等），并添加：

  

```bash

export PYTHONPATH="/your/custom/path1:/your/custom/path2:$PYTHONPATH"

```

  

然后运行 `source ~/.bashrc` 或重启终端以应用更改。

  

全局更改 `PYTHONPATH` 会影响所有使用相同 Python 解释器的项目，小心可能导致的版本冲突或导入问题。

  

### 2.2 方法 2：使用 `.pth` 文件

  

将路径添加到 Python 的 `site-packages` 目录内的 `.pth` 文件中。Python 会在启动时读取这些文件并将列出的路径添加到 `sys.path`。

  

1. 找到你的 Python 环境中的 `site-packages` 目录。你可以通过以下命令找到它：

  

```python

import site

print(site.getsitepackages())

```

  

2. 在该目录中创建一个新的文本文件，文件扩展名为 `.pth`（例如，`my_paths.pth`）。

  

3. 在文件中列出你希望添加到 `sys.path` 的每个路径，每行一个路径。

  

你可以实现对 `sys.path` 的持久化更改。不过，最好还是在项目级别管理依赖和路径，使用虚拟环境(virtualenv, virtualenv wrapper、 pipenv等工具管理)可以更好地控制和隔离项目的依赖。

# ubuntu23安装python3.8

在 Ubuntu 23.04 上，如果你已经有了 Python 3.10，并且希望安装 Python 3.8，可以通过以下步骤进行安装和管理多个 Python 版本：

1. **更新包管理器**：
   首先，确保你的包管理器是最新的。

   ```bash
   sudo apt update
   sudo apt upgrade
   ```

2. **安装 Python 3.8**：
   Ubuntu 的官方存储库可能不包含较旧版本的 Python，因此你可以使用 `deadsnakes` PPA 来安装 Python 3.8。

   ```bash
   sudo add-apt-repository ppa:deadsnakes/ppa
   sudo apt update
   sudo apt install python3.8
   ```

3. **验证安装**：
   安装完成后，验证 Python 3.8 是否正确安装。

   ```bash
   python3.8 --version
   ```

4. **管理多个 Python 版本**：
   使用 `update-alternatives` 命令来管理多个 Python 版本。

   首先，为新的 Python 版本创建一个替代项。

   ```bash
   sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
   sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 2
   ```

   然后，你可以使用以下命令来选择默认的 Python 版本：

   ```bash
   sudo update-alternatives --config python3
   ```

   这将显示一个选择列表，你可以在其中选择要使用的默认 Python 版本。

5. **安装 pip**：
   如果需要为 Python 3.8 安装 `pip`，你可以使用以下命令：

   ```bash
   sudo apt install python3.8-venv python3.8-distutils
   curl https://bootstrap.pypa.io/get-pip.py | sudo python3.8
   ```

这样，你就成功地在 Ubuntu 23.04 上安装了 Python 3.8，并能够在 Python 3.10 和 Python 3.8 之间进行切换。使用虚拟环境（`venv`）也是一个很好的方式来隔离不同项目的 Python 依赖。