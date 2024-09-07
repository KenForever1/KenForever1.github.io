## 技巧
### 跳转

- :jumps

查看跳跃列表

- <C-o>

跳转到上一个位置，o可以理解为old

- <C-i>

和<C-o>相对应配合使用，跳转方向相反
### 插入模式
（1）在插入模式下，为了快速删除，删除word，删除到行首等，可以使用：

- <C-u> 删除到行首
- <C-w> 向前删除一个word
- <C-h> 回退一个字符

（2）快速插入寄存器内容：
<C-r>加上寄存器的名字，比如：<C-r>a
（3）快速计算
比如编写图片处理程序，需要计算1080 * 1920 * 3的值，是不是要打开一个计算器，然后计算复制粘贴，这样太麻烦了。就可以使用
<C-r>=，然后提示你输入一个表达式比如：1920 * 1080 * 3，就会自动插入计算结果，也可以使用pow函数等计算次方。
### 在Ex模式快速插入
在Ex模式下，如果需要将光标位置的长字符串插入Ex输入中，
执行：<C-r><C-w>
### 快速打开终端

- <C-z>配合fg命令使用
- vim执行:ter，打开一个终端，然后exit退出
- vim执行:sh，打开一个终端，然后exit退出
### quickfix窗口

- :ccl[ose] 命令关闭窗口
- :copen 命令打开窗口
### Ex模式提示输入
在Ex模式输入一部分字符后，忘记了命令，或者需要提示，按<C-d>会出现提示。
### 文件跳转

- 在ex模式，执行 :r !pwd 可以在光标位置输入当前路径。
- 在normal模式，敲击 fg 可以跳转打开文件，如果文件路径有空格，先<S-v>选中文件路径，再使用fg。
### 打开文件的不同方式

- vim -o * 使用n个横向窗口打开文件
- vim -O * 使用n个纵向窗口（vertically）打开文件
- vim -p * 将多个文件按照多个tab打开
### 自带的自动补全

- <C-n> 正序遍历提示项
- <C-p> 倒序遍历提示项
### 搜索

- / 
- ？
- n next，p previous
- q/ 会打开搜索历史记录，然后就可以快速选择搜索项，再次搜索。
### 加密文件
vim -x file.txt 键入两次密码确认，然后加密文件
或者在ex模式，:X （注意是大写X）加密文件，或者两次回车解密文件，然后:wq保存。

-x命令打开文件，例如: vim -x t1.py；
也可以在打开文件后，执行 :X，注意这里是大写的X。
会提示你输入两次密码，

```
Enter encryption key: ***
Enter same key again: ***
```
然后执行:wq 保存退出。通过file命令查看文件，使用gedit等工具打开文件，会发现文件是加密后的乱码。

```
$ file t1.py
t1.py: Vim encrypted file data
```
file命令查看未加密文件:

```
$ file t1.py
t1.py: Python script, ASCII text executable
```
解密：
输入密码打开t1.py文件，然后执行 :X 命令后，提示输入密码，连续按两次回车，:wq 保存退出。
使用file查看解密后的文件:

```
Enter encryption key: 
Enter same key again:
```
参考：[https://github.com/adrianlarion/wizardly-tips-vim#encrypt](https://github.com/adrianlarion/wizardly-tips-vim#encrypt)

## vim9script配置

```
vim9script

# basic setting
set mouse=a
set number
set encoding=utf-8 fileencodings=ucs-bom,utf-8,cp936
# Set shift width to 4 spaces.
set shiftwidth=4
# Set tab width to 4 columns.
set tabstop=4
set showcmd
# autocomplete in ex mode, press tab 
set wildmenu

set clipboard=unnamedplus

# close quickfix list, such as after you excute :LspWorkspaceSymbols
# :cclose
# open quickfix list
# :copen

# goto previous place in jumps list, <C-o>, o mean : old
# goto next place in jumps list, <C-i>

# set xclip, Ctrl-C copy visual content to clipboard
# vnoremap <C-c> :w !xclip -selection clipboard<CR>

# ale
g:ale_linters_explicit = 1
g:ale_linters = {
	'csh': ['shell'],
	'python': ['flake8', 'mypy', 'pylint'],
	'c': ['gcc', 'cppcheck'],
	'cpp': ['gcc', 'cppcheck'],
	'text': []
}

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
# autocmd VimEnter * NERDTree | if argc() > 0 || exists("s:std_in") | wincmd p | endif

nnoremap <leader>n :NERDTreeFocus<CR>
# nnoremap <C-n> :NERDTree<CR>
nnoremap <C-t> :NERDTreeToggle<CR>
nnoremap <C-f> :NERDTreeFind<CR>

nnoremap <leader>` :ter<CR>

# lsp
nnoremap <leader>gd :LspDefinition<CR>
nnoremap <leader>ld :LspDeclaration<CR>
nnoremap <leader>lh :LspHover<CR>
nnoremap <leader>li :LspImplementation<CR>
nnoremap <leader>ls :LspDocumentSymbol<CR>
nnoremap <leader>lr :LspRename<CR>
nnoremap <leader>lf :LspDocumentFormat<CR>

# window resize
nnoremap <leader>w+ :vertical resize +30<CR>
nnoremap <leader>w- :vertical resize -30<CR>
nnoremap <leader>r+ :resize +30<CR>
nnoremap <leader>r- :resize -30<CR>

# async run
nnoremap <F5> :AsyncRun -mode=term -pos=gnome  ls -la<CR>
# :AsyncRun -mode=term -pos=tmux  ls -la

# lsp
inoremap <expr> <Tab> pumvisible() ? "\<C-n>" : "\<Tab>"
inoremap <expr> <S-Tab> pumvisible() ? "\<C-p>" : "\<S-Tab>"
inoremap <expr> <cr> pumvisible() ? "\<C-y>\<cr>" : "\<cr>"

# snipt
# " If you want :UltiSnipsEdit to split your window.
g:UltiSnipsEditSplit = "vertical"

# mru
# set max lenght for the mru file list
g:mru_file_list_size = 7 
# set path pattens that should be ignored
g:mru_ignore_patterns = 'fugitive\|\.git/\|\_^/tmp/' 

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

Plug 'rhysd/vim-lsp-ale'
 
# theme
Plug 'joshdick/onedark.vim'
Plug 'itchyny/lightline.vim'

# edit
Plug 'tpope/vim-surround'
Plug 'jiangmiao/auto-pairs'

# async run
Plug 'skywind3000/asyncrun.vim'
Plug 'benmills/vimux'

# recent files
Plug 'lvht/mru'

# ultisnips
Plug 'SirVer/ultisnips'
Plug 'thomasfaingnaert/vim-lsp-snippets'
Plug 'thomasfaingnaert/vim-lsp-ultisnips'

call plug#end()

# set colorschema
syntax on
colorscheme onedark

```
