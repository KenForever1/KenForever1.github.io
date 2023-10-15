
## install mkdocks

首先，介绍blog搭建用到的相关技术：
+ mkdocks
+ github pages
+ github actions

## mkdocks框架
上一版本使用的是github pages和hexo框架搭建的blog，hexo需要安装npm js等开发环境，由于我是后端开发，使用python脚本更多，而mkdocks使用python开发，因此感觉环境更加整洁。

mkdocks使用pip工具进行管理和安装，安装mkdocs：
```
python3 -m pip install mkdocks
```

### 创建新项目
执行命令，生成一个项目模板：
```
mkdocs new mysite
```

本地部署，打开浏览器查看
```
mkdocks serve
```

### 已有项目
针对已有的blog项目，比如之前hexo创建的blog项目
首先，拉取原来的仓库，比如在xxx.github.io目录中,将上面项目模板中的mkdocs.yml拷贝到xxx.github.io的跟目录中, 保证docs目录中有index.md或者README.md文件,作为blog首页。

```
proxychains4 git push origin main
```
## 自动化ci
在项目的根目录下，创建.github/workflows/ci.yml文件，这个文件会驱动github actions对blog项目中markdown（main分支中）进行编译和部署，生成静态前端展示页面文件，在gh-pages分支上。
```
name: ci 
on:
  push:
    branches:
      - master 
      - main
permissions:
  contents: write
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: 3.x
      - run: echo "cache_id=$(date --utc '+%V')" >> $GITHUB_ENV 
      - uses: actions/cache@v3
        with:
          key: mkdocs-material-${{ env.cache_id }}
          path: .cache
          restore-keys: |
            mkdocs-material-
      - run: pip install mkdocs-material 
      - run: mkdocs gh-deploy --force
```

## github pages设置
进入xxx.github.io项目github页面的setting中,选择pages,主要设置Build and deployment.

Source选择: deploy from branch;
Branch选择: gh-pages, /root .

经过以上设置,只要更新了项目文件,进行commit push后,就会触发ci,自动部署blog.

## 参考
[MkDocs 快速上手指南](https://sspai.com/prime/story/mkdocs-primer)
[publishing-your-site](https://squidfunk.github.io/mkdocs-material/publishing-your-site/#with-github-actions-material-for-mkdocs)