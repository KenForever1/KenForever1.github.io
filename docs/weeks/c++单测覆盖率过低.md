---
comments: true
---
在使用gcov + lcov测试分支覆盖率时，发现声明一个对象，以及调用vector push_back函数，都会出现50%的分支覆盖率，比如：

```cpp
+ -    Json::Value root;

+ -    Json::Reader reader;

+ -    xxx.push_back(yyy);
```

出现这种问题，是因为exception的缘故，那么我不希望通过exception情况的分支覆盖率有以下几种方式：

- 1. 编译选项中加入--no-exceptions，但是这种方法大概率代码编译不通过，比如c库里面有很多throw exception。
    

```bash
set(CMAKE_CXX_FLAGS "-fprofile-arcs -ftest-coverage --coverage --no-exceptions --no-inline")
```

- 想办法在生成代码覆盖率的工具层过滤，lcov没有提供直接的命令选项，而gcovr提供了--exclude-throw-branches。通过查看gcovr生成的html报告可以明显看出以上行不会出现分支未覆盖的情况了。
    

```bash
gcovr -r ${current_dir}  --html --html-details -o gc_coverage.html -v --filter . --exclude "/3rdparty/*" --exclude "/usr/*" --exclude-throw-branches
```

> `--exclude-throw-branches`ignores branches that GCC marks as exception only. This is mostly relevant in C++. Use of this feature is a double-edged sword. Often you're not interested in covering the branch from a function that throws, except when you have an explicit`catch`block.

还有值得注意的是，单条件分支，实际上有两个分支，如果只编写覆盖if的测试代码，只会覆盖50%的分支覆盖率。

```cpp
if(xxx == 0){

}
```

参考：

[https://github.com/gcovr/gcovr/issues/324](https://github.com/gcovr/gcovr/issues/324)

[https://stackoverflow.com/questions/42003783/lcov-gcov-branch-coverage-with-c-producing-branches-all-over-the-place](https://stackoverflow.com/questions/42003783/lcov-gcov-branch-coverage-with-c-producing-branches-all-over-the-place)