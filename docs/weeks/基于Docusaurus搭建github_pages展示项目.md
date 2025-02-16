---
comments: true
---
基于Docusaurus搭建github_pages效果如图所示：

![](https://raw.githubusercontent.com/KenForever1/CDN/main/llm_cool_docs.png)

比如你有一些项目介绍文档，或者你不想把文档放在博客里面，或者你想写本电子书。你有很多技术选择，包括静态的动态的网页，经过调研，看了几个网页介绍，都说Docusaurus好，那直接开干。

因为我的博客[kenforever1's Blog](https://kenforever1.github.io/)是基于MkDocs搭建的，因此换个技术试一下。

你或许有个疑问，我已经有个github部署的markdown静态博客，比如xxx.github.io访问。我还可以创建一个good_proj的项目，然后为这个项目搭建一个静态项目文档网页吗？

当然，这并不冲突，搭建好了以后通过**xxx.github.io/good_proj**访问就可以了。

## 初始化模板

直接上最简单的步骤：
```bash
npx create-docusaurus@latest my-website classic
```

安装的过程中，会让你选择语言，选择个javascript就可以了。

在本地启动例子程序。
```bash
npm start
```

然后你可以去docs目录下把默认的文章博客内容都删除了，换成你的。然后去docusaurus.config.js中把一些名字改改，把超链接注释一下。一个干净的网站就有了。

注释的内容比如：
+ src/components/HomepageFeatures/index.js 中
+ src/pages/index.js中

## 修改你的内容和配置

编辑docusaurus.config.js文件,
+ 项目名称projectName改为你创建的github项目
+ url改成你的
+ baseUrl改成你的项目名称
  
```js
const config = {
  title: 'LLM cool Docs',
  tagline: 'Dinosaurs are cool',
  favicon: 'img/favicon.ico',

  // Set the production url of your site here
  url: 'https://KenForever1.github.io/',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/llm_cool_docs/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'KenForever1', // Usually your GitHub org/user name.
  projectName: 'llm_cool_docs', // Usually your repo name.

}
```

### 语法高亮

默认python是语法高亮的，但是cpp这些没有，修改配置文件docusaurus.config.js，添加：
```js
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
        additionalLanguages: ['cpp','rust'],
      },
```
配置中的名字,比如cpp不能是c++，去网站上https://prismjs.com/#supported-languages找，因为这个实现是基于prismjs高亮的。

### 如何部署到github pages

在github上创建一个my-website的public项目。

在你的开发机器上，安装yarn
```
$ sudo npm install yarn
```

在你的开发机上应该都配置了ssh验证连接github吧，然后推送部署：
```bash
$ USE_SSH=true yarn deploy
```

我部署好了以后的网站，[kenforever1.github.io/llm_cool_docs](https://kenforever1.github.io/llm_cool_docs/docs/intro)。

## 改为github workflow自动部署

上面的手动部署方式，是没有把markdown文件提交到git上的。我们希望把markdown内容提交到主分支，然后自动触发workflow部署编译好的html到gh-pages分支。

分为如下步骤：
+ 进入github setting，创建Developer Setting->Personal access token -> Fine grained tokens -> Generate new token。并且勾选相关的权限。
+ 将KenForever1/llm_cool_docs项目的Setting中，Pages->Build and deployment,改为Github Actions。
+ 本地创建main分支，创建.github/workflows/deploy.yml文件

```
git checkout -b main
```

deploy.yml文件内容：
```yaml
name: Deploy to GitHub Pages

on:
  push:
    branches:
      - main
    # Review gh actions docs if you want to further define triggers, paths, etc
    # https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions#on

jobs:
  build:
    name: Build Docusaurus
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - uses: actions/setup-node@v4
        with:
          node-version: 18
          cache: yarn

      - name: Install dependencies
        run: yarn install --frozen-lockfile
      - name: Build website
        run: yarn build

      - name: Upload Build Artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: build

  deploy:
    name: Deploy to GitHub Pages
    needs: build

    # Grant GITHUB_TOKEN the permissions required to make a Pages deployment
    permissions:
      pages: write # to deploy to Pages
      id-token: write # to verify the deployment originates from an appropriate source

    # Deploy to the github-pages environment
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}

    runs-on: ubuntu-latest
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

保存提交github，就可以触发workflow了。

配置文件可以在[triggering-deployment-with-github-actions](https://docusaurus.io/docs/deployment#triggering-deployment-with-github-actions)这里找到。

参考：
+ [基于 Docusaurus 搭建自建博客](https://magicpenta.github.io/blog/2022/02/15/docusaurus/#35-%E8%AF%AD%E6%B3%95%E9%AB%98%E4%BA%AE)

+ [docusaurus deployment](https://docusaurus.io/zh-CN/docs/deployment)
