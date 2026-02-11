# arXiv风格LaTeX论文模板

基于《GEBench: Benchmarking Image Generation Models as GUI Environments》论文格式，适用于计算机科学、人工智能等领域的学术论文。

## 文件列表

- `arxiv_paper_template.tex` - 主LaTeX模板文件
- `references.bib` - 参考文献示例文件
- `compile.bat` - Windows编译脚本
- `clean.bat` - 清理中间文件脚本
- `COMPILE_INSTRUCTIONS.md` - 详细编译说明
- `README.md` - 本文件

## 快速开始

### Windows用户
1. 双击 `compile.bat` 编译论文
2. 生成的PDF文件为 `arxiv_paper_template.pdf`
3. 双击 `clean.bat` 清理中间文件

### 所有用户
```bash
# 编译
pdflatex arxiv_paper_template.tex
bibtex arxiv_paper_template
pdflatex arxiv_paper_template.tex
pdflatex arxiv_paper_template.tex

# 或使用latexmk
latexmk -pdf arxiv_paper_template.tex
```

## 模板特性

### 格式设置
- **字体**: Times Roman (学术论文标准字体)
- **段落**: 块状段落，段间空行 (arXiv风格)
- **页面边距**: 2.5cm (标准学术页面设置)
- **图表标题**: "Figure X | 描述" 格式
- **参考文献**: 作者-年份引用格式

### 主要组件
1. **arXiv标识**: 文档开头的arXiv编号和日期
2. **多作者支持**: 使用authblk包处理多个单位和作者
3. **专业表格**: booktabs包实现三线表
4. **代码展示**: listings包支持语法高亮
5. **智能引用**: cleveref包自动处理引用格式

## 定制指南

### 修改论文信息
编辑 `arxiv_paper_template.tex` 中的以下部分：
```latex
\title{Your Paper Title: A Comprehensive Study on [Your Topic]}
\author[1]{First Author}
\affil[1]{First Institution}
% arXiv标识
arXiv:XXXX.YYYYYv1 [cs.AI] \today
```

### 添加内容
- **章节**: `\section{Title}`, `\subsection{Title}`, `\subsubsection{Title}`
- **图表**: 使用 `figure` 和 `table` 环境
- **公式**: 使用 `equation` 或 `align` 环境
- **引用**: `\citep{key}` 或 `\citet{key}`

### 更新参考文献
1. 编辑 `references.bib` 文件
2. 添加新的文献条目
3. 在文中使用 `\citep{引用键}`

## 示例内容
模板包含完整的示例内容，包括：
- 标题页
- 摘要和关键词
- 6个主要章节
- 图表示例
- 数学公式
- 参考文献
- 附录

## 扩展建议

### 添加中文支持
```latex
\usepackage{xeCJK}
\setCJKmainfont{SimSun}
```

### 添加算法描述
```latex
\usepackage{algorithm}
\usepackage{algorithmic}
```

### 添加定理环境
```latex
\newtheorem{theorem}{Theorem}[section]
\newtheorem{lemma}[theorem]{Lemma}
```

## 问题排查

### 编译错误
1. 检查是否安装了完整的LaTeX发行版 (TeX Live或MiKTeX)
2. 检查缺失的包：`tlmgr install <package-name>`
3. 查看 `arxiv_paper_template.log` 文件获取详细错误信息

### 参考文献问题
1. 确保执行了 `bibtex` 步骤
2. 检查 `.bib` 文件中的引用键是否与文中一致
3. 重新运行完整编译流程

### 图片问题
1. 将图片放在同一目录或指定正确路径
2. 使用支持的格式：PDF, PNG, JPG
3. 检查图片文件名是否正确

## 版本控制建议
建议使用Git管理论文版本。将以下文件添加到 `.gitignore`：
```
*.aux
*.bbl
*.blg
*.log
*.out
*.toc
*.lof
*.lot
*.pdf (可选，可重新生成)
```

## 相关资源
- [Overleaf](https://www.overleaf.com/) - 在线LaTeX编辑器
- [TeX Live](https://tug.org/texlive/) - LaTeX发行版
- [BibTeX格式指南](http://www.bibtex.org/Format/)
- [LaTeX Wikibook](https://en.wikibooks.org/wiki/LaTeX)

## 许可证
本模板基于MIT许可证，可自由使用、修改和分发。