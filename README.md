# Aspect-based Sentiment Analysis

## Requirements

The sentiment analysis software is written in Python, and supports Python 3.7 and above.
In addition, folders _data_ and _models_ must contain some files. Check README files in respective folders for further information.

## Usage

### 1. Preprocessing & Feature Extraction

Before any sentiment prediction can be done on the SentiCoref corpus we have to preprocess the data. This can be done by running

```
python -m nlp_code.preprocessing
```

from the root folder of the repository. This will perform lemmatisation and POS tagging on the corpus using the Stanza library. Results at this step will be saved to _data/cache_.

Next, feature extraction is performed, using `FeaturePipeline` with defined feature extracotrs. The latter are defined in module `nlp_code.features`. Features extracted from preprocessed text are saved in _data/features_. These TSV files contain one row of extracted features for each word in a coreference chain.

It is recommended that feature extractors are only added to the `FeaturePipeline`, since removing them might break previously implemented models that depend on some removed features.

### 2. Model Evaluation

After preprocessing is completed, model evaluation can be started by running

```
python -m nlp_code.models
```

from the root folder of the repository. Models can use any of the features stored in _data/features_.
