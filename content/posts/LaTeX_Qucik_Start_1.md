---
author: "xingyu"
author_link: ""
title: "LaTeX Learning"
date: 2020-12-30T10:20:16+08:00
lastmod: 2020-12-30T10:20:16+08:00
draft: false
description: ""
show_in_homepage: true
description_as_summary: false
license: ""

tags: ["paper"]
categories: ["paper"]

featured_image: ""
featured_image_preview: ""

comment: true
toc: true
auto_collapse_toc: true
math: false
---

>一直都想要学LaTeX，最近恰逢模式识别大作业，而且为数学建模做准备，把这个顺手学一学。

<!--more-->

## 基本格式

```latex
\documentclass[UTF8]{ctexart}  % 为了使用中文   \documentclass{article}
% 导言区   %为注释符
\begin{document}
	% 内容
\end{document}
```

第一行`\documentclass{article}`中包含了一个**控制序列**（或称命令 / 标记）。

所谓**控制序列**，是以反斜杠`\`开头，以第一个*空格或非字母* 的字符结束的一串文字，他们并不被输出，但是他们会影响输出文档的效果。

这里的控制序列是`documentclass`，它后面紧跟着的`{article}`代表这个控制序列有一个必要的参数，该参数的值为`article`. 这个控制序列的作用，是调用名为 “article” 的**文档类**。

其后出现了控制序列`begin`，这个控制序列总是与`end`成对出现。这两个控制序列以及他们中间的内容被称为**“环境”**；他们之后的第一个必要参数总是一致的，被称为环境名。

只有在 “document” 环境中的内容，才会被正常输出到文档中去或是作为控制序列对文档产生影响。因此，在`\end{document}`之后插入任何内容都是无效的。

`\begin{document}`与`\documentclass{article}`之间的部分被称为**导言区**。导言区中的控制序列，通常会影响到整个输出文档。

## 导言区设置

### 调整布局

```latex
\usepackage[a4paper]{geometry} % 调整纸张大小和页边距的包，中括号中规定了纸张大小
```

```latex
\geometry{left=2.0cm,right=2.0cm,top=2.0cm,bottom=2.0cm} % 页边距设置
```

### 代码格式设置

```latex
\lstset{
    basicstyle          =   \sffamily,          % 基本代码风格
    keywordstyle        =   \bfseries,          % 关键字风格
    commentstyle        =   \rmfamily\itshape,  % 注释的风格，斜体
    stringstyle         =   \ttfamily,  % 字符串风格
    flexiblecolumns,                % 别问为什么，加上这个
    numbers             =   left,   % 行号的位置在左边
    showspaces          =   false,  % 是否显示空格，显示了有点乱，所以不现实了
    numberstyle         =   \zihao{-5}\ttfamily,    % 行号的样式，小五号，tt等宽字体
    showstringspaces    =   false,
    captionpos          =   t,      % 这段代码的名字所呈现的位置，t指的是top上面
    frame               =   lrtb,   % 显示边框
}

\lstdefinestyle{Python}{
    language        =   Python, % 语言选Python
    basicstyle      =   \zihao{-5}\ttfamily,
    numberstyle     =   \zihao{-5}\ttfamily,
    keywordstyle    =   \color{blue},
    keywordstyle    =   [2] \color{teal},
    stringstyle     =   \color{magenta},
    commentstyle    =   \color{red}\ttfamily,
    breaklines      =   true,   % 自动换行，建议不要写太长的行
    columns         =   fixed,  % 如果不加这一句，字间距就不固定，很丑，必须加
    basewidth       =   0.5em,
}
```

## 行文格式

```latex
% 生成目录
\tableofcontents
```

`article`/`ctexart` 中，定义了五个控制序列来调整行文组织结构。他们分别是

 ```latex
 \section{·}
 \subsection{·}
 \subsubsection{·}
 \paragraph{·}
 \subparagraph{·}
 ```

## 相关包

```latex
\usepackage{graphicx}
\usepackage{subfigure}
\usepackage{amsmath}
\usepackage{listings}
```

### graphicx

### subfigure

插入图片

```latex
\begin{figure}[h]
\centering
\includegraphics[width=12cm,height=6cm]{file_name} % 文件名称 当前路径
\caption{Max Pool} %标题
\end{figure}
```

### amsmath

### listings

分点， 前面序号

```latex
\begin{itemize}
\item 1
\item 2
\end{itemize}
```

* 1
* 2

```latex
\begin{enumerate}
\item 1
\item 2
\item 3
\end{enumerate}
```

1. 1
2. 2
3. 3

### lstlisting

插入代码

```latex
\begin{lstlisting}
	print("Hello, World!")
\end{lstlisting}
```

### align

多行公式

```latex
\begin{align}
Y(m, n) &= X(m, n) * H(m, n) \nonumber \\ 
		   &= \sum_{i=-\inf}^{+\inf}\sum_{j=-\inf}^{+\inf}X{i, j}H{m-i, n-j} \nonumber  \\  
		   &= \sum_{i=-\inf}^{+\inf}\sum_{j=-\inf}^{+\inf}X(m-i, n-j)H(i,j)
		   \nonumber
\end{align}
```

## 附加

```latex
\vspace*{64pt}   % 垂直间距64pt
\hspace{2em}  % 水平间距 两个字符
```

```latex
% 设置字体大小
\LARGE{}
\
\fontsize{48pt}{0}{模\quad 式\quad 识 \quad 别 \quad 大 \quad 作 \quad 业}
```

```latex
\clearpage  % 新建一页
\par %另起一段
```

