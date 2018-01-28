---
title: blog+github备份和恢复
date: 2018-01-24 20:00:00
tags: hexo
categories: hexo
toc: True

---
换了新电脑以后如何恢复BLOG

<!--more-->

在用hexo搭建博客的时候需要备份一下，这样电脑出问题的时候或者换新的电脑以后才能快速的恢复blog环境。

首先讲一讲搭建的流程

1. 创建仓库，名字必须是 用户名.github.io
2. 创建两个分支，master和hexo
3. 设置hexo为默认分支
4. 使用git clone git@github.com:(仓库名)克隆到本地
5. 在文件夹下打开终端，分支显示为__hexo__，依次执行npm install hexo  \  hexo init  \  npm install  \  npm install hexo-deploy-git
6. 修改 _config.yml中的deploy参数，分支应为master
7. 依次执行git add . , git commit -m "..." , git push origin hexo　提交网站相关资料
8. 执行hexo g -d 部署blog

对于日常修改博客的时候，应该首先用git add . , git commit -m "..." , git push origin hexo　将内容保存至github中，然后才用hexo g -d部署博客

更换电脑以后如何恢复呢？

1. 使用 git clone -b hexo git@github.com: ....将仓库克隆到本地。
2. 然后在文件夹下执行: npm install hexo  /  npm install  /  npm install hexo-deployer-git (不需要hexo init)

可能会遇到一些问题，命令行里面也会提示你到相应的地方去找解决方案，按照提示做就行，再不济google一下。

完美！