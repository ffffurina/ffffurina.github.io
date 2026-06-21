# 主页维护备忘录

## 1. 这套站点怎么改

这个仓库是 Academic Pages 的 Jekyll 站点。最常改的地方有三类：

- 站点全局配置：`_config.yml`
- 顶部导航：`_data/navigation.yml`
- 首页正文：`_pages/about.md`

如果你要改个人信息、侧边栏、站点标题、颜色，优先看 `_config.yml`。
如果你要改首页显示内容，优先看 `_pages/about.md`。
如果你要改顶栏菜单，优先看 `_data/navigation.yml`。

## 2. 上线代码的执行方式

这个站点的上线方式是：改代码 -> 提交到 GitHub -> GitHub Pages / GitHub Actions 自动构建 -> 部署。

你本地通常不需要“手动执行上线代码”，而是通过 Git 提交触发部署。

常用流程：

```powershell
git status
git add -A
git commit -m "your message"
git push origin master
```

如果远程有更新导致 push 被拒绝，先同步远程：

```powershell
git pull origin master
git push origin master
```

如果拉取时提示需要选择合并方式，可以先设置一次：

```powershell
git config pull.rebase false
```

## 3. 本地预览方式

如果你想在本地看效果，通常可以运行 Jekyll 预览：

```powershell
bundle install
bundle exec jekyll serve
```

然后打开本地地址查看页面。

注意：如果你改的是 `_config.yml`，通常需要重启本地预览服务。

## 4. 常见修改例子

### 4.1 修改首页个人信息

改 `_config.yml` 里的 `author` 段：

- `name`：侧边栏姓名
- `bio`：简介
- `location`：位置
- `employer`：单位
- `email`：邮箱
- `github`：GitHub 用户名
- `googlescholar`：Google Scholar 链接
- `orcid`：ORCID 链接

首页正文则改 `_pages/about.md`。

### 4.2 修改顶部菜单

改 `_data/navigation.yml`。

例如把 Blog 改成外部博客：

```yaml
- title: "Blog"
  url: https://ffffurina.github.io/academic-blog/
```

### 4.3 新增一篇 publication

在 `_publications/` 下新增一个 `.md` 文件，写前置元数据：

```markdown
---
title: "Paper Title"
collection: publications
category: conferences
permalink: /publication/paper-title
excerpt: 'Short summary.'
date: 2026-06-01
venue: 'Conference Name'
paperurl: '/files/paper-title.pdf'
---
```

PDF 放在 `files/` 目录，比如 `files/paper-title.pdf`。

### 4.4 删除某个模块

如果不想要某个栏目，通常要删两层：

- 顶部菜单入口：`_data/navigation.yml`
- 对应页面文件：例如 `year-archive.html`、`talks.html`、`teaching.html`、`portfolio.html`

如果还想连内容也去掉，再删对应集合目录里的示例文件：

- `_posts/`
- `_talks/`
- `_teaching/`
- `_portfolio/`

## 5. 文件放哪里

- 图片：`images/`
- PDF：`files/`
- 论文页面：`_publications/`
- 首页正文：`_pages/about.md`

## 6. 推送前检查清单

- `git status` 看有没有漏提交的文件
- `git pull origin master` 先同步远程
- `git push origin master` 推送
- GitHub 仓库里看 Actions 是否成功
- 如果页面没更新，等几分钟再强刷浏览器

## 7. 这套主页目前的关键约定

- 首页是 `_pages/about.md`
- 顶栏是 `_data/navigation.yml`
- 侧边栏信息是 `_config.yml`
- publication 下载链接是 `paperurl`
- PDF 放在 `files/`