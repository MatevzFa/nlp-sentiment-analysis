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
        Each function must take arguments (Article, Word), and return a dictironary of extracted features
        of form {feature_name: feature_value}.

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
                rec.update(f(article, word))
            records.append(rec)

        return pd.DataFrame.from_records(records)


"""
FEATURE EXTRACTORS
"""


def word_n_lemma(n: int):
    @feature(f"word_{n}_lemma")
    def aux(article: Article, word: Word):
        if 0 <= word.word_index + n < article.num_words:
            return article.word_at(word.word_index + n).lemma
        else:
            return None

    return aux


def word_n_pos(n: int):
    @feature(f"word_{n}_pos")
    def aux(article: Article, word: Word):
        if 0 <= word.word_index + n < article.num_words:
            return article.word_at(word.word_index + n).pos_tag
        else:
            return None

    return aux


def word_n_word_sentiment(n: int):
    @feature(f"word_{n}_word_sentiment")
    def aux(article: Article, word: Word):
        if 0 <= word.word_index + n < article.num_words:
            return article.word_at(word.word_index + n).word_sentiment
        else:
            return None

    return aux


@feature
def entity_type(article: Article, word: Word):
    return list(word.entity_types)[0]
