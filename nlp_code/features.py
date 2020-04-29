from functools import wraps
from typing import List

import pandas as pd

from nlp_code.preprocessing import Article, Word


def feature(feature_name=None):
    """
    This decorator makes the wrapped function return a dictionary {feature_name: return_value}
    """

    def wrapper(func):
        func.__name__ = feature_name or func.__name__

        @wraps(func)
        def aux(*args, **kwargs):
            return {func.__name__: func(*args, **kwargs)}

        return aux

    return wrapper


class FeaturePipeline:

    def __init__(self, *extractors):
        """
        Each function must take arguments (Article, Word, features), and return a dictironary of extracted features
        of form {feature_name: feature_value}.
        Features are currently calculated features for this word.

        These functions are called on each of the words passed into the __call__ function and
        collected into a Pandas DataFrame.

        For example:
        ```
        ...
        pipeline = FeaturePipeline(
            word_n_lemma(1), word_n_pos(1), word_n_word_sentiment(1),
            word_n_lemma(2), word_n_pos(2), word_n_word_sentiment(2),
            word_n_lemma(3), word_n_pos(3), word_n_word_sentiment(3),
        )

        df = pipeline(article, article.coreference_chains[chain_id])
        ...
        ```
        """
        self.extractors = extractors

    def __call__(self, article: Article, words: List[Word]) -> pd.DataFrame:
        """
        Returns a Pandas DataFrame with lines representing extracted features on each of the words
        provided in list 'words'.
        Features are extracted using self.extractors
        """
        records = []
        for word in words:
            rec = {}
            for f in self.extractors:
                rec.update(f(article, word, rec))
            records.append(rec)

        return pd.DataFrame.from_records(records)


"""
FEATURE EXTRACTORS
"""


def word_n_lemma(n: int):
    @feature(f"word_{n}_lemma")
    def aux(article: Article, word: Word, features: dict):
        if 0 <= word.word_index + n < article.num_words:
            return article.word_at(word.word_index + n).lemma
        else:
            return None

    return aux


def word_n_pos(n: int):
    @feature(f"word_{n}_pos")
    def aux(article: Article, word: Word, features: dict):
        if 0 <= word.word_index + n < article.num_words:
            return article.word_at(word.word_index + n).pos_tag
        else:
            return None

    return aux


def word_n_word_sentiment(n: int):
    @feature(f"word_{n}_word_sentiment")
    def aux(article: Article, word: Word, features: dict):
        if 0 <= word.word_index + n < article.num_words:
            return article.word_at(word.word_index + n).word_sentiment
        else:
            return None

    return aux


@feature()
def entity_type(article: Article, word: Word, features: dict):
    l = list(word.entity_types)
    return l[0] if len(l) > 0 else None


@feature()
def sentence_pos_count(article: Article, word: Word, features: dict):
    num = 0
    for w in article.iter_sentence(word.sentence_index):
        if w.word_sentiment is not None and len(w.chain_ids) == 0 and w.word_sentiment > 0:
            num += 1
    return num


@feature()
def sentence_neg_count(article: Article, word: Word, features: dict):
    num = 0
    for w in article.iter_sentence(word.sentence_index):
        if w.word_sentiment is not None and len(w.chain_ids) == 0 and w.word_sentiment < 0:
            num += 1
    return num


@feature()
def sentence_pos_neg(article, word, features):
    return (features["sentence_pos_count"] + 1) / (features["sentence_neg_count"] + 1)


@feature()
def sentence_neg_entities(article, word, features):
    """
    Number of surrounding entities with negative sentiment(< 3) in a sentence
    """
    num = set()
    for w in article.iter_sentence(word.sentence_index):
        if len(w.chain_ids) > 0:
            for id in w.chain_ids:
                if (id not in word.chain_ids) and article.chain_sentiments[id] < 3:
                    num.add(id)
    return len(num)

@feature()
def sentence_pos_entities(article, word, features):
    """
    Number of surrounding entities with positive sentiment(> 3) in a sentence
    """
    num = set()
    for w in article.iter_sentence(word.sentence_index):
        if len(w.chain_ids) > 0:
            for id in w.chain_ids:
                if (id not in word.chain_ids) and article.chain_sentiments[id] > 3:
                    num.add(id)
    return len(num)    



@feature() #Doesn't work - yet 
def entity_references_sentiment(article, word, features):
    """
    returns pos_count/neg_count of all reference words for an entity
    """
    pos_count = 0
    neg_count = 0
    for id in word.chain_ids:
        for w in article.coreference_chains[id]:
            if w.word_sentiment is not None and w.word_sentiment > 0:
                pos_count += 1
            elif w.word_sentiment is not None and w.word_sentiment < 0:
                neg_count += 1
    if pos_count and neg_count:
        print("sentiment exists", word)
    return (pos_count + 1)/(neg_count + 1)
    

@feature()
def sentence_sentiment(article, word, features):
    return word.sentence_sentiment
