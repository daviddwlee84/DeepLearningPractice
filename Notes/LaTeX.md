# LaTeX

## Basics

### Creating LaTeX document

#### The preamble of a document

The part of your .tex file before the `\begin{document}` is called the preamble. In the preamble, you define the type of document you are writing and the language, load extra packages you will need, and set several parameters.

### Compile

![LATEX compilation file flow](https://cdn.sharelatex.com/learn-scripts/images/e/ea/Latex-file-flow.png)

```sh
latex mydocument.tex
# This will create "mydocument.dvi", a DVI document
```

```sh
pdflatex mydocument.tex
# This will generate "mydocument.pdf", a PDF document
```

#### BibTeX

```sh
# need paper.tex (xxx.sty optional)
latex paper # get paper.aux
# need paper.aux, xxx.bst
bibtex paper # get paper.bbl

# final output (with paper.tex paper.bbl)
latex paper # get paper.dvi
pdflatex paper # get paper.pdf
```

![BibTeX proess 1](https://www.arakhne.org/autolatex/process.png)

[![BibTeX proess 2](http://www.zapata.org/stuart/latex/bibtex_process.jpg)](http://www.zapata.org/stuart/latex/bibtex.shtml)

### Extension

#### Main Document

* **tex**: LaTeX

#### Bibliography

> The word *BibTeX* stands for a tool and a file format which are used to describe and process lists of references, mostly in conjunction with LaTeX documents.

* **bst**: BibTeX style file
  * [Guidelines for customizing biblatex styles](https://tex.stackexchange.com/questions/12806/guidelines-for-customizing-biblatex-styles/13076#13076)
  * [Where the .bst file comes in](https://tex.stackexchange.com/questions/85432/where-the-bst-file-comes-in)
* **bib**: BibTex-File
* **bbl**: (Generated file)

```sh
# Make a bst file
latex makebst
```

Example of using `acl_natbib.bst` with custom bib BibTex file called mybib

```tex
\bibliographystyle{acl_natbib}
\bibliography{mybib}
```

### Style

* **sty**: Style

Example of NAACL HLT 2019 (`naaclhlt2019.sty`)

```tex
% in main.tex, after documentclass
\usepackage[hyperref]{naaclhlt2019}
```

> [NAACL-HLT 2019 Call For Papers](https://naacl2019.org/calls/papers/)
>
> **Follow style and format guidelines**
>
> Submissions should follow the NAACL-HLT 2019 style guidelines. Long paper submissions must follow the two-column format of ACL proceedings without exceeding eight (8) pages of content. Short paper submissions must also follow the two-column format of ACL proceedings, and must not exceed four (4) pages. References do not count against these limits. We strongly recommend the use of the official NAACL-HLT 2019 style templates:
>
> * [LaTeX](https://naacl2019.org/downloads/naaclhlt2019-latex.zip)
>   * (this will include `acl_natbib.bst`, `naaclhlt2019.bib`, `naaclhlt2019.sty`, `naaclhlt2019.tex`)
> * Microsoft Word
> * Overleaf

All submissions must be in PDF format.

Submissions that do not adhere to the above author guidelines or ACL policies will be rejected without review.

#### Output

* **dvi**: Device independent file format consists of binary data describing the visual layout of a document in a manner not reliant on any specific image format, display hardware or printer.
* **ps**: PostScript file format describes text and graphics on page and it is based on vector graphics. PostScript is, until now, a standard in desktop publishing areas.
* **pdf**: Portable Document Format is a file format, based on PostScript, used to represent documents in a manner independent of application software, hardware, and operating systems. It is now widely used as a file format for printing and for distribution on the Web.

## List

### Unordered List

```tex
\begin{itemize}
  \item The individual entries are indicated with a black dot, a so-called bullet.
  \item The text in the entries may be of any length.
\end{itemize}
```

### Ordered List

```tex
\begin{enumerate}
  \item The labels consists of sequential numbers.
  \item The numbers starts at 1 with every call to the enumerate environment.
\end{enumerate}
```

Available styles for numbered lists:

| Code      | Description                               |
| --------- | ----------------------------------------- |
| `\alph`   | Lowercase letter (a, b, c, ...)           |
| `\Alph`   | Uppercase letter (A, B, C, ...)           |
| `\arabic` | Arabic number (1, 2, 3, ...)              |
| `\roman`  | Lowercase Roman numeral (i, ii, iii, ...) |
| `\Roman`  | Uppercase Roman numeral (I, II, III, ...) |

### Nested List

```tex
\begin{enumerate}
   \item The labels consists of sequential numbers.
   \begin{itemize}
     \item The individual entries are indicated with a black dot, a so-called bullet.
     \item The text in the entries may be of any length.
   \end{itemize}
   \item The numbers starts at 1 with every call to the enumerate environment.
\end{enumerate}
```

### List Style

### Spacing

> [enumitem package](#enumitem)

* [Vertical space in lists](https://tex.stackexchange.com/questions/10684/vertical-space-in-lists)
* [Latex Remove Spaces Between Items in List](https://stackoverflow.com/questions/3275622/latex-remove-spaces-between-items-in-list)
* [How to adjust list spacing](https://texfaq.org/FAQ-complist)

## Figures and Tables

### Positioning

| Parameter | Position                                                                                                                         |
| --------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `h`       | Place the float **here**, i.e., approximately at the same point it occurs in the source text (however, not exactly at the spot)  |
| `t`       | Position at the **top** of the page.                                                                                             |
| `b`       | Position at the **bottom** of the page.                                                                                          |
| `p`       | Put on a special **page** for floats only.                                                                                       |
| `!`       | Override internal parameters LaTeX uses for determining "good" float positions.                                                  |
| `H`       | Places the float at precisely the location in the LATEX code. Requires the float package. This is somewhat *equivalent to* `h!`. |

### Table Rules

The typeset of tables should be based on the following rules:

1. never use vertical lines
2. avoid double lines
3. place the units in the heading of the table (instead of the body)
4. do not use quotation marks to repeat the content of cells

> [**booktabs** package](#booktabs)

## References and Citations - Bibliography

## Cross Referencing Sections, Figures and Equations

## References and Citations

### How to find BibTeX of a paper

1. Open [Google Scholar](https://scholar.google.com.tw/)
2. Search the paper using its name
   * e.g. BERT => BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
3. There will be a `"` symbol under the searching result
4. Click it and select BibTex
    * e.g. [BERT BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:HmKlRThp8ysJ:scholar.google.com/&output=citation&scisig=AAGBfm0AAAAAXKycFNmRKivBlTvhmo4Dt0b6Pxci9O9e&scisf=4&ct=citation&cd=-1&hl=en)

        ```txt
        @article{devlin2018bert,
        title={Bert: Pre-training of deep bidirectional transformers for language understanding},
        author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
        journal={arXiv preprint arXiv:1810.04805},
        year={2018}
        }
        ```

## Formatting

### Page Numbering

`\pagestyle{empty}` stops the pages being numbered
`\pagestyle{plain}` this is the default; it puts the numbers at the bottom of the page
`\pagestyle{headings}` puts the numbers at the top of the page; the precise style and content depends on the document class
`\pagenumbering{roman}` numbers pages using Roman numerals; use arabic to switch it back

#### Start numbering after the table of contents

```tex
% preamble %

\begin{document}

\maketitle
\thispagestyle{empty}

\newpage

\tableofcontents
\thispagestyle{empty}

\clearpage
\pagenumbering{arabic}

% contents %
```

### Multiple Columns

> [multicol package](#multicol)

* [How to produce a double column document in Latex?](https://tex.stackexchange.com/questions/285443/how-to-produce-a-double-column-document-in-latex)

### Footnotes

```tex
\footnote{...}
```

## Fonts

### Background Knowledge of Fonts

* TrueType (TTF)
* OpenType (OTF)

### Default Fonts

```tex
\renewcommand{\familydefault}{<family>}
```

* `\rmdefault`: serif (roman)
* `\sfdefault`: sans serif
* `\ttdefault`: typewriter (monospace)

### Times New Roman

> [mathptmx package](#mathptmx)

```tex
\usepackage{mathptmx}
```

> [newtx bundle](#newtx)

```tex
\usepackage{newtxtext,newtxmath}
```

> [Times package](#Times) => This will get error. (This package is outdated.)

```tex
\usepackage{Times}
```

* [How to set document font to times new roman by command](https://tex.stackexchange.com/questions/153168/how-to-set-document-font-to-times-new-roman-by-command)

### Customized Fonts

> [fontspec package](#fontspec)

## Package

### booktabs

> For prettier tables

* replace `\hline` with `\toprule`, `\midrule` and `\bottomrule`

### enumitem

[enumitem](https://ctan.org/pkg/enumitem)

```tex
\usepackage{enumitem}
\setlist{nosep} % or \setlist{noitemsep} to leave space around whole list
```

### multicol

[multicol manual](http://ftp.yzu.edu.tw/CTAN/macros/latex/required/tools/multicol.pdf)

### geometry

### titling

> Prettify the `\maketitle`

```tex
% This will center the title
\usepackage{titling}
\renewcommand\maketitlehooka{\null\mbox{}\vfill}
\renewcommand\maketitlehookd{\vfill\null
```

> You can also write your own page like this
>
> ```tex
> \begin{titlepage}
>     \begin{center}
>         ...
>     \end{center}
> \end{titlepage}
> ```

### graphicx

```tex
\usepackage{graphicx}
\graphicspath{ {./images/} }
```

### Drawing Related

#### TikZ

> [Tutorial](#Drawing)

#### Circuitikz

### Font Related

#### fontspec

Example: (Pretty good but current with citation problem)

```tex
\usepackage{fontspec}
\setmainfont[Ligatures=TeX]{Georgia}
\setsansfont[Ligatures=TeX]{Arial}
```

#### mathptmx

> For Times New Roman

[mathptmx](https://ctan.org/pkg/mathptmx)

#### newtx

> For Times New Roman

[newtx bundle](https://ctan.org/pkg/newtx)

#### Times

> For Times New Roman (deprecated)

### Chinese Related

> Compile with `xelatex`

#### CTeX

```tex
% Use it as a documentclass
\documentclass[UTF8]{ctexart}
\begin{document}
你好，這是一個測試文件。
\end{document}
```

```tex
% Use it as a macro package
\documentclass{article}
\usepackage[UTF8]{ctex}
\begin{document}
你好，這是一個測試文件。
\end{document}
```

#### xeCJK

## Appendix

### LaTeX units and lengths

| Abbreviation   | Definition                                          |
| -------------- | --------------------------------------------------- |
| `pt`           | A point, is the default length unit. About 0.3515mm |
| `mm`           | a millimetre                                        |
| `cm`           | a centimetre                                        |
| `in`           | an inch                                             |
| `ex`           | the height of an x in the current font              |
| `em`           | the width of an m in the current font               |
| `\columnsep`   | distance between columns                            |
| `\columnwidth` | width of the column                                 |
| `\linewidth`   | width of the line in the current environment        |
| `\paperwidth`  | width of the page                                   |
| `\paperheight` | height of the page                                  |
| `\textwidth`   | width of the text                                   |
| `\textheight`  | height of the text                                  |
| `\unitlength`  | units of length in the picture environment.         |

### Setting the VSCode LaTeX Workshop

#### Add another compile method

> [LaTeX-Workshop Wiki - LaTeX Recipes](https://github.com/James-Yu/LaTeX-Workshop/wiki/Compile#latex-recipes)

1. Open up the VSCode setting.
2. Search for `latex-workshop.latex.recipe`.
3. Click __Edit in settings.json__
4. Typing `latex-workshop.latex.recipe` and you will see the auto completion.
5. You can add something like this for compile with `xelatex`

    ```json
    {
        "name": "XeLaTeX",
        "tools": [
            "xelatex"
        ]
    }
    ```

6. Also don't forget to add the additional tool in `latex-workshop.latex.tools`

    ```json
    {
        "name": "xelatex",
        "command": "xelatex",
        "args": [
            "%DOC%"
        ],
        "env": {}
    }
    ```

7. Then you're done!

### Set TeXcount on local

#### Installation

1. Download TeXcount script from the [download page](https://app.uio.no/ifi/texcount/download.html)
2. Add a alias in your `.bashrc` or `.zshrc`...

    ```sh
    # TeXcount
    alias texcount="perl /path/to/TeXcount_3_1_1/texcount.pl"
    ```

3. And you're able to use `texcount` as a command anywhere

#### Chinese Word Count

```sh
texcount -v0 -ch main.tex
```

## Links

### Article

#### Overleaf

* [**Learn LaTeX in 30 minutes**](https://www.overleaf.com/learn/latex/Learn_LaTeX_in_30_minutes)
* Basics
  * [Creating a document in LaTeX](https://www.overleaf.com/learn/latex/Creating_a_document_in_LaTeX)
  * [**Choosing a LaTeX Compiler**](https://www.overleaf.com/learn/latex/Choosing_a_LaTeX_Compiler)
* Document Structure
  * [**Multi-file LaTeX projects**](https://www.overleaf.com/learn/latex/Multi-file_LaTeX_projects)
  * [Hyperlinks](https://www.overleaf.com/learn/latex/Hyperlinks)
  * [Management in a large project](https://www.overleaf.com/learn/latex/Management_in_a_large_project)
  * [Cross referencing sections and equations](https://www.overleaf.com/learn/latex/Cross_referencing_sections_and_equations)
* Figures and Tables
  * [Positioning images and tables](https://www.overleaf.com/learn/latex/Positioning_images_and_tables)
* References and Citations
  * [Bibliography management in LaTeX](https://www.overleaf.com/learn/latex/Bibliography_management_in_LaTeX)
* Formatting
  * [Multiple columns](https://www.overleaf.com/learn/latex/Multiple_columns)
  * [Footnotes](https://www.overleaf.com/learn/latex/Footnotes)

How to Write a Thesis in LaTeX

* [(Part 1): Basic Structure](https://www.overleaf.com/learn/latex/How_to_Write_a_Thesis_in_LaTeX_(Part_1):_Basic_Structure)
* [(Part 2): Page Layout](https://www.overleaf.com/learn/latex/How_to_Write_a_Thesis_in_LaTeX_(Part_2):_Page_Layout)
* [(Part 3): Figures, Subfigures and Tables](https://www.overleaf.com/learn/latex/How_to_Write_a_Thesis_in_LaTeX_(Part_3):_Figures,_Subfigures_and_Tables)
* [(Part 4): Bibliographies with BibLaTeX](https://www.overleaf.com/learn/latex/How_to_Write_a_Thesis_in_LaTeX_(Part_4):_Bibliographies_with_BibLaTeX)
* [(Part 5): Customising Your Title Page and Abstract](https://www.overleaf.com/learn/latex/How_to_Write_a_Thesis_in_LaTeX_(Part_5):_Customising_Your_Title_Page_and_Abstract)

#### WikiBooks - [LaTeX](https://en.wikibooks.org/wiki/LaTeX)

* [LaTeX/Tables](https://en.wikibooks.org/wiki/LaTeX/Tables)
* [LaTeX/Fonts](https://en.wikibooks.org/wiki/LaTeX/Fonts)

#### Using Chinese in LaTeX

* [**LaTeX 中如何使用中文**](https://jdhao.github.io/2018/03/29/latex-chinese.zh/)
* [How does one type Chinese in LaTeX?](https://tex.stackexchange.com/questions/17611/how-does-one-type-chinese-in-latex)

#### Others

* [**Keeping tables/figures close to where they are mentioned**](https://tex.stackexchange.com/questions/2275/keeping-tables-figures-close-to-where-they-are-mentioned)
* [Tabular: title above and caption below?](https://tex.stackexchange.com/questions/15282/tabular-title-above-and-caption-below)
* [Trying to replicate a table from academic paper](https://tex.stackexchange.com/questions/63204/trying-to-replicate-a-table-from-academic-paper)
* [What is the standard/recommended font to use in papers?](https://academia.stackexchange.com/questions/26889/what-is-the-standard-recommended-font-to-use-in-papers)
* [How to draw images in Latex?](https://tex.stackexchange.com/questions/397528/how-to-draw-images-in-latex)
* [Is there any way to do a correct word count of a LaTeX document?](https://tex.stackexchange.com/questions/534/is-there-any-way-to-do-a-correct-word-count-of-a-latex-document)
* [知乎 - 你寫論文時發現了哪些神網站？](https://www.zhihu.com/question/35931336/answer/641198933)

Title thing

* [Vertically centering a title page](https://stackoverflow.com/questions/3141702/vertically-centering-a-title-page)
* [Vertically Center title page : Article](https://tex.stackexchange.com/questions/290432/vertically-center-title-page-article)

Page Numbering

* [Start page numbering from second page in Latex](https://swetava.wordpress.com/2015/05/14/start-page-numbering-from-second-page-in-latex/)
* [7 Titles and Contents](http://www-users.york.ac.uk/~pjh503/LaTeX/frontstuff.html)

### LaTeX Tools

* [The LaTeX Project](https://www.latex-project.org/)

(support Mac)

* [MacTeX](https://www.tug.org/mactex/) - a distribution including a complete TeX system and editors (basically contains everything you need)
  * GUI Applications
    * [TeXShop](https://pages.uoregon.edu/koch/texshop/)
    * BibDesk
    * [LaTeXiT](https://www.chachatelier.fr/latexit/)
    * TeX Live Utility
    * Excalibur (the spell checker)
  * Ghostscript 9.23
  * TeX Live 2018 - the actual TeX distribution

IDE

* [TeXstudio](https://www.texstudio.org/)
* [Texmaker](http://www.xm1math.net/texmaker/)

BibTeX

* [How to use BibTeX](http://www.bibtex.org/Using/)
* [BibTeX Format Description](http://www.bibtex.org/Format/) - `.bib`

XeLaTeX

* [Wiki - XeTeX](https://en.wikipedia.org/wiki/XeTeX)

VS Code Extension

* [Visual Studio Code LaTeX Workshop Extension](https://github.com/James-Yu/LaTeX-Workshop)

#### Formula/Equation

* [**LaTeXiT**](https://www.chachatelier.fr/latexit/) - equation
* [**Mathpix**](https://mathpix.com/) - convet images to LaTeX
  * [examples](https://mathpix.com/examples.pdf)

#### iOS App

* [MyScript](https://www.myscript.com/)
  * MathPad ([End-of-sale](https://www.myscript.com/retired-apps/))
  * [Nebo](https://www.myscript.com/nebo)

#### Word Count

* [TeXcount](http://app.uio.no/ifi/texcount/index.html)
  * [**TeXcount web service**](https://app.uio.no/ifi/texcount/online.php) - Online word count service (support Chinese!)
  * [download script](http://app.uio.no/ifi/texcount/download.html)
* [detex](https://www.ctan.org/pkg/detex) + wc (the word count command)
* [latexcount](https://www.ctan.org/pkg/latexcount) - Perl script to count words of LATEX documents
* [wordcount](https://ctan.org/pkg/wordcount) - Estimate the number of words in a LATEX document

#### Markdown Editor which support LaTeX

* [MacDown](https://macdown.uranusjr.com/)
* [Typora](https://typora.io/)

Online Collaboration

* [HackMD](https://hackmd.io/)

VS Code Extension

* Markdown All in One
* Markdown PDF
* markdownlint

### Tutorial

* [**LaTeX-Tutorial.com**](https://www.latex-tutorial.com/)
  * [**quick start**](https://www.latex-tutorial.com/quick-start/)
  * [LaTeX tables - Tutorial with code examples](https://www.latex-tutorial.com/tutorials/tables/)
    * [Tables from .csv in LaTeX with pgfplotstable](https://www.latex-tutorial.com/tutorials/pgfplotstable/)

#### Video

* [Derek Banas - LaTeX Tutorial](https://youtu.be/VhmkLrOjLsw)

#### Drawing

TikZ

* Overleaf - LaTeX Graphics using TikZ: A Tutorial for Beginners
  * [(Part 1)—Basic Drawing](https://www.overleaf.com/learn/latex/LaTeX_Graphics_using_TikZ:_A_Tutorial_for_Beginners_(Part_1)%E2%80%94Basic_Drawing)
  * [(Part 2)—Generating TikZ Code from GeoGebra](https://www.overleaf.com/learn/latex/LaTeX_Graphics_using_TikZ:_A_Tutorial_for_Beginners_(Part_2)%E2%80%94Generating_TikZ_Code_from_GeoGebra)
  * [**(Part 3)—Creating Flowcharts**](https://www.overleaf.com/learn/latex/LaTeX_Graphics_using_TikZ:_A_Tutorial_for_Beginners_(Part_3)%E2%80%94Creating_Flowcharts)
  * [(Part 4)—Circuit Diagrams Using Circuitikz](https://www.overleaf.com/learn/latex/LaTeX_Graphics_using_TikZ:_A_Tutorial_for_Beginners_(Part_4)%E2%80%94Circuit_Diagrams_Using_Circuitikz)
  * [(Part 5)—Creating Mind Maps](https://www.overleaf.com/learn/latex/LaTeX_Graphics_using_TikZ:_A_Tutorial_for_Beginners_(Part_5)%E2%80%94Creating_Mind_Maps)

### Online Editor

* [Overleaf](https://www.overleaf.com/) - The easy to use, online, collaborative LaTeX editor
  * [ShareLaTeX](https://www.sharelatex.com/) (ShareLaTeX is now part of Overleaf)
* [LaTeX Base](https://latexbase.com/)
* [MyScript Web Demo](https://webdemo.myscript.com/)

#### Table Generator

* [LaTeX Table Generator](https://www.tablesgenerator.com/latex_tables)

#### Drawing Graph

* GeoGebra
* [LaTeXDraw](http://latexdraw.sourceforge.net/)

### Template

* [Overleaf - Templates](https://www.overleaf.com/latex/templates)
  * [Gallery — arXiv](https://www.overleaf.com/gallery/tagged/arxiv)
    * [Style and Template for Preprints (arXiv, bio-arXiv)](https://www.overleaf.com/latex/templates/style-and-template-for-preprints-arxiv-bio-arxiv/fxsnsrzpnvwc)
  * [Instructions for NAACL-HLT 2019 Proceedings](https://www.overleaf.com/latex/templates/instructions-for-naacl-hlt-2019-proceedings/xyyfwfkswhth)
* [Google Drive - article and journal templates.zip](https://drive.google.com/file/d/0B5wBAHw-J-ViRG9CVEZVWF96OVk/view)
* [arXiv BERT Template](https://arxiv.org/format/1810.04805)
* [Springer LaTeX templates](https://www.springer.com/gp/livingreviews/latex-templates)

ElegantLaTeX

* [ElegantLaTeX Templates](https://elegantlatex.org/en/)
* [Github - ElegantLaTeX/ElegantNote](https://github.com/ElegantLaTeX/ElegantNote)

### Q&A

* [**TEX FAQ**](https://texfaq.org/)
* [StackExchange TEX](https://tex.stackexchange.com/)

### E-book

* [**LaTeX Tutorials A Primer**](https://www.tug.org/twg/mactex/tutorials/ltxprimer-1.0.pdf)
* [LaTeX for Beginners](http://www.docs.is.ed.ac.uk/skills/documents/3722/3722-2014.pdf)
* [**Tables in LATEX2ε: Packages and Methods**](https://www.tug.org/pracjourn/2007-1/mori/mori.pdf) - Table Rules
* [MIT - Figures and Tables in a LATEX Document](http://web.mit.edu/rsi/www/pdfs/fig-intro.pdf)
* [**Everything you always wanted to know about BiBTEX**](https://ntg.nl/bijeen/pdf-s.20031113/BibTeX-tutorial.pdf)
* [Graphics in LaTeX using TikZ](https://www.tug.org/TUGboat/tb29-1/tb91walczak.pdf)
