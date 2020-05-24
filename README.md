# Aspect-based Sentiment Analysis

## Requirements

The sentiment analysis software is written in Python, and supports Python 3.7 and above.
In addition, folders _data_ and _models_ must contain some files. Check README files in respective folders for further information.

To install requried dependencies, you can run `pip install -r requirements.txt`.

### Running with Docker

This project is also available as a docker container. **It is recommended to use _Docker_ and _docker-compose_ for running this project**. Commands to run phases within Docker containers are given in chapter _Usage_ along with the regular commands for running the project locally. We use the same docker-compose volumes for all steps. Therefore, as long as you do not delete these volumes (by e.g. runnnig docker-compose down), the intermediate data should be persistent.

The docker container contains all data, models and code required for running this project successfully. Its size is around 3.8GB.

## Usage

### 1. Preprocessing & Feature Extraction

Before any sentiment prediction can be done on the SentiCoref corpus we have to preprocess the data. This can be done by running

```
python -m nlp_code.preprocessing

# Docker
docker-compose run preprocessing
```

from the root folder of the repository. This will perform lemmatisation and POS tagging on the corpus using the Stanza library. Results at this step will be saved to _data/cache_.

Next, feature extraction is performed, using `FeaturePipeline` with defined feature extracotrs. The latter are defined in module `nlp_code.features`. Features extracted from preprocessed text are saved in _data/features_. These TSV files contain one row of extracted features for each word in a coreference chain.

It is recommended that feature extractors are only added to the `FeaturePipeline`, since removing them might break previously implemented models that depend on some removed features.

### 2. Model Evaluation

#### Classical models

_After preprocessing is completed_, model evaluation can be started by running

```
python -m nlp_code.models

# Docker
docker-compose run models
```

from the root folder of the repository. Models can use any of the features stored in _data/features_.

#### Neural models

To run the CustomSentiCorefModel, you can run

```
python -m nlp_code.models_neural

# Docker
docker-compose run models_neural
```

from the root folder of the repository. This will perform the required preprocessing, training and evaluation of this model. Make sure you downloaded the required BERT model, described [here](/models/README.md).

To run the BertEmbeddingsSentiCoref (takes quite some time), you can run

```
python -m nlp_code.bert_embeddings

# Docker
docker-compose run bert_embeddings
```

For running the pretrained model with balanced training set run the Docker command (see instructions from CustomSentiCorefModel) or download the trained model from [here](https://drive.google.com/file/d/1lUXpav0wHxH7m7J_Xae-87kgINxypx0C/view?usp=sharing), unzip it into _models/_ and run

```
python -m nlp_code.pretrained_bert_embeddings_balanced

# Docker
docker-compose run pretrained_bert_embeddings_balanced
```
