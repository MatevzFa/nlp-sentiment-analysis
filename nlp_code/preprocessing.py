import logging
import os
import pickle
import re
from collections import defaultdict
from typing import Dict, List, Optional

import pandas as pd
import stanza

from nlp_code import features
from nlp_code.articles import *
from nlp_code.sentiment_lexicon import JOBLexicon

log = logging.getLogger('senti_an')
log.setLevel(logging.INFO)

"""
Passing environment variable DEBUG (with any non-empty value) enables debug mode
"""
IS_DEBUG = bool(os.getenv("DEBUG"))
if IS_DEBUG:
    log.setLevel(logging.DEBUG)


"""
POS tagging and lemmatisation of the dataset.
TODO: Store augmented data
"""
if __name__ == "__main__":

    article_preprocessor = ArticlePreprocessor()

    article_loader = ArticleLoader("data/SentiCoref_1.0")

    articles = ["42.tsv"] if IS_DEBUG else article_loader.list_articles()

    feature_pipeline = features.FeaturePipeline(
        features.word_n_pos(-3), features.word_n_word_sentiment(-3),  # 3 left
        features.word_n_pos(-2), features.word_n_word_sentiment(-2),  # 2 left
        features.word_n_pos(-1), features.word_n_word_sentiment(-1),  # 1 left

        features.word_n_pos(1), features.word_n_word_sentiment(1),  # 1 right
        features.word_n_pos(2), features.word_n_word_sentiment(2),  # 2 right
        features.word_n_pos(3), features.word_n_word_sentiment(3),  # 3 right

        features.entity_type,

        features.sentence_neg_count, features.sentence_pos_count,
        features.sentence_pos_neg,

        features.sentence_sentiment,

        features.sentence_pos_entities, features.sentence_neg_entities, features.entity_references_sentiment
    )

    for art_name in articles:
        art = article_loader.load_article(art_name)
        art = article_preprocessor(art_name, art)

        art.filter_words(lambda w: w.pos_tag in ["NOUN", "PROPN", "VERB", "ADJ"])

        dfs = []
        for chain_id, words in art.coreference_chains.items():
            data = feature_pipeline(art, words)
            data["article_id"] = art.article_id
            data["chain_id"] = chain_id
            data["sentiment"] = art.chain_sentiments[chain_id]
            dfs.append(data)

        pd.concat(dfs).to_csv(f"data/features/{art_name}", sep="\t")
