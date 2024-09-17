---
title: git如何使用？多个修改分多个commit修改
date: 2024-09-17
authors: [KenForever1]
categories: 
  - git
  - 开发工具
  - 开发技能
labels: []
---
## 一个文件多处修改分多个commit提交

在项目开发中很常用的工作流，将修改拆分为多个清晰的提交，这样在回退和排查问题时也更加容易，代码review也更加好理解。

采用git add -p, 根据提示分成多个小块add修改。
<!-- more -->

git add -p的提示命令有：
s: split，将当前修改拆分为多个小块；
e: edit，进入编辑模式手动搞，如果split不成功，就可以采用这个手动解决；

比如下面的情况是split划分不了的，需要edit模式解决。
e模式下，
```
-11
-22
-33
+1
+2
+3

+4
```

如果你只想add:
```
+4
```
那么在e按下后，编辑模式下需要做的编辑更改是，删除不需要add的'+'号行，将减号'-'行换成' '空格。注释是换成空格不是删除减号，否则会提示你修改不通过，继续重新修改。
```
 11
 22
 33

+4
```
这样就只add了“+ 4”的修改。剩下的修改如下，可以下次再add到另一个提交。
```
-11
-22
-33
+1
+2
+3
```

下面这篇文章讲的很详细，可以参考：[同一个文件修改了多处如何分作多个提交](https://ttys3.dev/blog/git-how-to-commit-only-parts-of-a-file)。

## 修改git默认编辑器

git默认的是nano编辑器，我用的不大习惯，配置成vim。
```bash
git config --global core.editor vim
```

## 撤销git add，重新提交
当我们git add出错时，想要回到原来的修改状态，然后重新git add。
```bash
git reset <file>
```