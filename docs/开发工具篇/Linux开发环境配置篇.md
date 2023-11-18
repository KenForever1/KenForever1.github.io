## 开发环境：

- shell fish
- vim + tmux
- vscode

vscode 使用了remote explorer插件，可以自动识别docker container，直接连接进入开发环境。相比clion，clion docker插件需要创建一个新的container，而不是连接一个存在的container。
## fish
不用像zsh一样大量配置，开箱即用，和fzf，riggrep等工具都很好兼容。
## tmux
tmux使用了默认快捷键，Ctrl-b触发
## fzf
fish 设置环境变量过滤目录，设置fd
```shell
set -x FZF_DEFAULT_COMMAND "fd --exclude={.git,.idea,.sass-cache,node_modules,build} --type f"
```
## vim配置
安装 vim9
```cpp
sudo add-apt-repository ppa:jonathonf/vim
sudo apt install vim
```
配置.vimrc
```shell
vim9script

# basic setting
set mouse=a
set number
set tabstop=4
set encoding=utf-8 fileencodings=ucs-bom,utf-8,cp936
# Set shift width to 4 spaces.
set shiftwidth=4
# Set tab width to 4 columns.
set tabstop=4
set showcmd

# set xclip, Ctrl-C copy visual content to clipboard
vnoremap <C-c> :w !xclip -selection clipboard<CR>

# ale
g:ale_linters_explicit = 1
g:ale_linters = { 'cpp': ['gcc', 'clangd'] }
# let g:ale_linters = {
	  # \   'csh': ['shell'],
	    # \   'zsh': ['shell'],
	      # \   'go': ['gofmt', 'golint'],
	        # \   'python': ['flake8', 'mypy', 'pylint'],
		  # \   'c': ['gcc', 'cppcheck'],
		    # \   'cpp': ['gcc', 'cppcheck'],
		      # \   'text': [],
		        # \}
g:ale_completion_delay = 500
g:ale_echo_delay = 20
g:ale_lint_delay = 500
g:ale_echo_msg_format = '[%linter%] %code: %%s'
g:ale_lint_on_text_changed = 'normal'
g:ale_lint_on_insert_leave = 1
g:airline#extensions#ale#enabled = 1
g:ale_c_gcc_options = '-Wall -O2 -std=c99'
g:ale_cpp_gcc_options = '-Wall -O2 -std=c++11'
g:ale_c_cppcheck_options = ''
g:ale_cpp_cppcheck_options = ''

# fzf, add to bashrc
# export FZF_DEFAULT_COMMAND="fd --exclude={.git,.idea,.sass-cache,node_modules,build} --type f"

# Start NERDTree. If a file is specified, move the cursor to its window.
autocmd StdinReadPre * let s:std_in=1
autocmd VimEnter * NERDTree | if argc() > 0 || exists("s:std_in") | wincmd p | endif

nnoremap <leader>n :NERDTreeFocus<CR>
nnoremap <C-n> :NERDTree<CR>
nnoremap <C-t> :NERDTreeToggle<CR>
nnoremap <C-f> :NERDTreeFind<CR>

# lsp
inoremap <expr> <Tab> pumvisible() ? "\<C-n>" : "\<Tab>"
inoremap <expr> <S-Tab> pumvisible() ? "\<C-p>" : "\<S-Tab>"
inoremap <expr> <cr> pumvisible() ? "\<C-y>\<cr>" : "\<cr>"


call plug#begin()

Plug 'preservim/nerdtree', { 'on': 'NERDTreeToggle' }

# fzf
Plug 'junegunn/fzf', { 'do': { -> fzf#install() } }
Plug 'junegunn/fzf.vim'

# c++
Plug 'dense-analysis/ale'

Plug 'terryma/vim-multiple-cursors'
# Plug 'mg979/vim-visual-multi', {'branch': 'master'}

Plug 'prabirshrestha/async.vim'
Plug 'prabirshrestha/asyncomplete.vim'
Plug 'prabirshrestha/vim-lsp'
Plug 'prabirshrestha/asyncomplete-lsp.vim'
Plug 'mattn/vim-lsp-settings'
 
# theme
Plug 'joshdick/onedark.vim'
Plug 'itchyny/lightline.vim'

# edit
Plug 'tpope/vim-surround'
Plug 'jiangmiao/auto-pairs'

# async run
Plug 'skywind3000/asyncrun.vim'

call plug#end()

# set colorschema
syntax on
colorscheme onedark
```
## clang-format
```shell
sudo update-alternatives --install /usr/bin/clang-format clang-format /usr/bin/clang-format-9 100
```

-i : 直接更改文件，不加就终端输出
clang-format -style=file -i main.cpp
## neovim的使用
neovim开源配置，github高Star的。

- nvchad 界面支持更加根据丰富
- lunarVim lua配置的neovimIDE，支持

也可以自己配置，但是配置需要做好不断更新和维护，不然会插件不兼容报错等。使用这样的开源配置有个好处，由仓库管理者维护版本，可以更加聚焦学习编辑的内容，许多vim入门者在vim和neovim配置上花费了大量时间。
## vim无插件编辑
[vim 无插件编辑]([https://zhuanlan.zhihu.com/p/43510931](https://zhuanlan.zhihu.com/p/43510931))，无插件也是一个很好的选择，比如：当年到同事的机器上编辑代码，终端修改服务器配置，出差编辑代码，无网络环境编辑等，这些场景都只是一个干干净净的vim，因此掌握了无插件编辑可以提高效率， 同时无插件编辑也是vim使用的基础。
我在使用终端时，会尽量使用不配置的tmux，以及安装fish开箱即用（而不是需要大量配置的 zsh），掌握默认的按键规则。

- ~：选中字符大小写转换
- =：格式对齐，gg=G 对全文格式对齐
- u：撤销，Ctrl-r：u的逆操作
- 注释添加"//"，Ctrl-v，选中添加注释的段落，然后按'I'或者'A'，输入'//'，再按ESC退出，实现注释。删除注释，Ctrl-v，选中注释符号，然后按d，ESC。
- bve：选中一个字符串
- 窗口切换Ctrl-w-hjkl，tab切换gt和gT
- :Te 以tab窗口形式显示当前目录 然后可进行切换目录、打开某个文件
- 掌握vim与系统剪贴板的交互，[https://zhuanlan.zhihu.com/p/73984381](https://zhuanlan.zhihu.com/p/73984381)
- 自带的文件浏览器Netrw，打开: E，[https://zhuanlan.zhihu.com/p/61588081](https://zhuanlan.zhihu.com/p/61588081)
- 在ex模式，显示可以使用的Commands，Ctrl-d

## GNOME Terminal (Ubuntu 默认终端)
掌握快捷键可以更加高效：

1. 字体调整
- Ctrl-Shift-+ ：放大终端
- Ctrl--：缩小终端（Ctrl键和减号键一起使用）
2. 打开终端 Ctrl-Alt-t
3. 在打开的终端上新增加Tab：Ctrl-Shift-t，关闭Tab：Ctrl-Shift-w（类似vscode默认Ctrl-w关闭Tab）
4. 打开多个Tab，使用键盘切换Tab，Ctrl-Page Down，Ctrl-PaOn，Alt+数字也可以切换到对于的Tab。
5. 全屏 F11
6. 复制粘贴：Ctrl-Shift-v，Ctrl-Shift-c
7. 跳转到命令最前面：Ctrl-a，命令末尾：Ctrl-e，和Emacs一样的按键。

这些命令不用死记硬背，多使用就能形成肌肉记忆，忘了查看Terminal Help中的快捷键，掌握这些命令可以提高效率。
## vim系统剪贴板
[Vim 剪贴板里面的东西 粘贴到系统粘贴板？ - 知乎](https://www.zhihu.com/question/19863631/answer/1289724886?utm_id=0)

```
sudo apt install vim-gui-common # 解决-clipboard 问题，不用重新装vim
set clipboard=unnamedplus # 现在你的y，d，x，p已经能和 ctrl-c和ctrl-v 一个效果，并且能互相混用。
```