# Data

This directory contains the corpora and other data sets used for this project.

It has to contain the following data sets, each in its own folder.

| Folder name      | Source                           | Included |
| ---------------- | -------------------------------- | -------- |
| `SentiCoref_1.0` | http://hdl.handle.net/11356/1285 | Yes      |
| `JOB_1.0`        | http://hdl.handle.net/11356/1112 | Yes      |
| `SentiNews_1.0`  | http://hdl.handle.net/11356/1110 | **No**   |

## Directory structure

```
data
├── JOB_1.0
│   └── Slovene_sentiment_lexicon_JOB.txt
├── SentiCoref_1.0
│   ├── 1.tsv
│   ├── 20.tsv
│   └── ...
└── SentiNews_1.0
    ├── SentiNews_document-level.txt
    ├── SentiNews_paragraph-level.txt
    └── SentiNews_sentence-level.txt
```
