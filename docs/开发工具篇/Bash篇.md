## Bash scripting_cheatsheet_
[https://devhints.io/bash](https://devhints.io/bash)
还有其他语言，工具的使用说明，cheatsheet
## [pure-bash-bible](https://github.com/dylanaraps/pure-bash-bible#foreword)
[https://github.com/dylanaraps/pure-bash-bible](https://github.com/dylanaraps/pure-bash-bible#foreword)
在vscode中，通过安装 Pure Bash Bible插件，输入函数名，就可以自动补全该仓库中的函数，提高效率。
vscode中还可以安装其他bash相关插件。
## bash语法
### 3.1 Internal Field Separator ([IFS](https://man7.org/linux/man-pages/man1/bash.1.html))
[https://www.baeldung.com/linux/ifs-shell-variable](https://www.baeldung.com/linux/ifs-shell-variable)
查看命令解释，在bash终端：

```
help read
help declare
```
注意，man read是查看linux api。
### 3.2 Bash 特殊变量
[https://www.baeldung.com/linux/bash-special-variables](https://www.baeldung.com/linux/bash-special-variables)
#### 输入参数相关的变量

- ${#}，输入参数的个数
- ${@}，输入参数拼接字符串
- ${*}，输入参数拼接字符串，以IFS的第一个字符相连接
- ${N}，N是具体的数字，代表了输入参数

例如：my_func是一个自定义的bash函数。执行下面的命令

```
./my_func --user admin --pass hello
```
对应输入参数相关变量的值就是：

```
${#} ：4
${@} : --useradmin--passhello
${*} : --useradmin--passhello

${0} : ./xxx.sh # xxx.sh是定义my_func的文件
${1} : --user
${2} : admin
${3} : --pass
```
如果将IFS改为':'符号，${@}不变，${*}会发生变化，

```
${*} : --user:admin:--pass:hello
```
## 进程相关

- ${?}，上一个执行的进程返回的状态码，常用于判断调用一个函数后的执行结果判断
- _**${!}，**_**get our background Job process id**
- ${$}，current shell process ID

获取后台进程ID，等待执行结束。

```
login & 
wait ${!}
```
watchdog，作为在一些deamon进程中很常见功能，通过查看/proc/$pid是否存在，判断进程是否在执行。

```
!/bin/bash
function watchdog(){
    pid=`cat ./shell.pid`
    if [[ -e /proc/$pid ]];then
        echo "Login still running"
    else 
        echo "Login is finished"
    fi
}
watchdog
```
### 3.3 内置函数
#### Shift
[解释]([https://www.gnu.org/savannah-checkouts/gnu/bash/manual/bash.html#index-shift](https://www.gnu.org/savannah-checkouts/gnu/bash/manual/bash.html#index-shift))

```
function login(){
     previous checks omitted 
    while [[ ${#} > 0 ]]; do
        case $1 in
            -user) 
            user=${2};
            shift;;
            -pass)
            pass=${2};
            shift;;
            *)
            echo "Bad Syntax. Usage: -user [username] -pass [password] required"; 
            return;;
        esac
        shift
    done
    echo "User=${user}"
    echo "Pass=${pass}"
}
```
#### declare

```
function download_linux(){
    declare -a linux_versions=("5.7" "5.6.16" "5.4.44")
    declare -a commands
    mkdir linux
    for version in ${linux_versions[@]}
        do 
            curl -so ./linux/kernel-${version}.tar.xz -L \
            "https://cdn.kernel.org/pub/linux/kernel/v5.x/linux-${version}.tar.xz" &
            echo "Running with pid ${!}"
            commands+=(${!})
        done   
    for pid in ${commands[@]}
        do
            echo "Waiting for pid ${pid}"
            wait $pid
        done
}
```