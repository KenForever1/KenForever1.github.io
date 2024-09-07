## 学习网址
[https://learngitbranching.js.org/?locale=zh_CN](https://learngitbranching.js.org/?locale=zh_CN)
通过完成网址的实验，已经可以掌握大多数概念和操作。
## 多人同一个分支开发
git stash
git pull --rebase
git stash pop
git add -u
git commit -m ""
git push origin HEAD:refs/for/xxx
如果其他人在remote进行了提交，本地做了修改准备提交
git add
git commit -m "" # 首先提交本地修改, 很重要，否则会丢失代码
git pull --rebase # git 会进行对比合并
git add # 再次add本地修改
git commit
git push origin HEAD:refs/for/xxx
提交的代码审核出现问题，需要修改
修改代码
git add
git commit --amend --no-edit # 复用同一个commit
git push
git 提交错误(为了安全起见，在撤销有其他开发者工作的 repo 中的修改时，请使用git revert。)
git reset HEAD~1
git add 
git commit
git push --force
## 多分支开发流程
### main和dev分支
```rust
git checkout -b dev
git commit -m "add some thing"
git rebase main dev 或者 git rebase main # 拉取main分支的修改，可能在开发的过程中main提交了新功能
git push origin dev
git checkout main # 合并dev分支
git merge dev
git push origin main
```
## git rebase
git rebase 和git merge都是用于合并分支的操作。
区别：
git merge 产生一条新的commit，它的parent同时指向了两个合并的分支。
git rebase 在HEAD位置重新应用要合并的提交请求，不会产生新的commit。

git 多分支rebase，把多个分支的修改rebase到main分支。
[https://zhuanlan.zhihu.com/p/271677627](https://zhuanlan.zhihu.com/p/271677627)
git rebase main bugFix ，main指定了目标分支，bugFix指定了执行分支。
等效于：
```rust
git checkout bugFix
git rebase main
```
```rust
          A---B---C topic*
         /
    D---E---F---G master
```
```rust
                  A'--B'--C' topic*
                 /
    D---E---F---G master
```
在主分支main上应用：git checkout main; git merge bugFix。

如果在rebase的过程中，出错了，想要回退到rebase之前的状态，那么应该结合git reflog和git reset命令，
[https://zhuanlan.zhihu.com/p/462531895](https://zhuanlan.zhihu.com/p/462531895)。
```rust
git reflog
git reset --hard HEAD@{2} # HEAD@{2} 是reflog命令执行后显示的rebase前的状态。
```
## git rebase -i
--interactive的缩写，提供了一种交互式的方式修改git提交，比如修改、重排包括HEAD以及HEAD的前三条commit，执行： git rebase -i HEAD~4
该操作执行后，会打开vim，对不同的提交可以重新排序，以及pick，omit删除等操作。
## git HEAD
^: 表示当前位置的parent
^^: 表示当前位置的parent的parent
~：表示HEAD的上一个位置
~num：表示HEAD的上num个位置
## git cherry-pick
该操作可以将任意一个hash提交记录应用到HEAD的下一个位置，比如c1提交应用到HEAD的下一个位置，执行：git cherry-pick c1。
如果有多个c1，c2，c3，那么执行：git cherry-pick c1 c2 c3，分支变为：HEAD->c1'->c2'->c3'。

## git revert 和git reset
git revert 会通过增加一个commit提交，比如c1->c2，要撤销c2提交，则执行git revert会变成c1->c2->c3，c3分支的状态等效于c1。
git reset 直接删除c2 commit，但是要注意soft，hard ,以及mix的区别，git默认--mix。
##  git 分离HEAD
使用git checkout hash，可以将HEAD指向hash指向的位置。

## git bisect 查错命令
[https://ruanyifeng.com/blog/2018/12/git-bisect.html](https://ruanyifeng.com/blog/2018/12/git-bisect.html)
采用二分法，直到找出提交产生bug的commit。

## git tag 打标签
git tag v1 c3 在c3的位置打上v1的标签。
## git describe
 查找最近的tag，以及距离，描述信息等

## git 如何合并和压缩？
[https://zhuanlan.zhihu.com/p/462532410](https://zhuanlan.zhihu.com/p/462532410)
将feature分支中的多个commit合并提交到main分支
```rust
git checkout main
git merge --squash feature
git commit -m ""
```
## git fetch
git fetch只是同步远程分支到本地的origin/main，如果要把它同步到main分支，还需要git merge 或者git rebase。
git提供了git pull，等于git fetch 加上 git merge origin/main。
git pull --rebase，等于git fetch 加上 git rebase main origin/main。
## 设置远程跟踪分支
默认是main跟踪origin/main
通过如下命令，foo就会跟踪origin/main分支，foo分支的提交也会提交到远程main分支。
```rust
git checkout -b foo origin/main # 或者
git branch -u origin/main foo
```
## git push参数

- git push

git push orgin main
等效于git push origin main:main

- git push 如果目的地和提交的分支不一样：

git push origin source:destination
比如：git push origin man:NewDes，将main分支代码提交到远程分支NewDes上。

- git fetch 的参数和push类似，只是方向相反，一个是上传，一个是拉取。
- git push origin :foo，如果source留空，会删除远程的foo分支。
- git fetch origin :foo1，如果source留空，会在本地创建foo1分支。
