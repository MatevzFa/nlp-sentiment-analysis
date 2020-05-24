# Models

This directory contains [Stanza](https://stanfordnlp.github.io/stanza/index.html) models. Since these model files can be large, they are not kept in this repository. The models can be downloaded from web pages linked in the table below.

It has to contain the following models:

| File name                 | Source                                        |
| ------------------------- | --------------------------------------------- |
| `ssj500k_lemmatizer.pt`   | http://hdl.handle.net/11356/1286              |
| `slo-hr-en-bert-pytorch/` | Model fine-tuned by Ulčar and Robnik-Šikonja. |

Optionally, if you want to run `pretrained_bert_embeddings_balanced.py`, it has to contain

| File name                             | Source                                                                 |
| ------------------------------------- | ---------------------------------------------------------------------- |
| `pretrained_bert_embeddings_balanced` | https://drive.google.com/file/d/1lUXpav0wHxH7m7J_Xae-87kgINxypx0C/view |

## Directory structure

```
models
├── README.md
├── pretrained_bert_embeddings_balanced
│   ├── BertEmbeddingsSentiCoref.data-00000-of-00002
│   ├── BertEmbeddingsSentiCoref.data-00001-of-00002
│   ├── BertEmbeddingsSentiCoref.index
│   └── checkpoint
├── slo-hr-en-bert-pytorch
│   ├── config.json
│   ├── pytorch_model.bin
│   └── vocab.txt
└── ssj500k_lemmatizer.pt
```
