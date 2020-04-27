import logging
import os
import pickle
import re
from collections import defaultdict
from typing import Dict, List, Optional

import pandas as pd
import stanza

from nlp_code import features
from nlp_code.sentiment_lexicon import JOBLexicon

log = logging.getLogger('senti_an')
log.setLevel(logging.INFO)

"""
Passing environment variable DEBUG (with any non-empty value) enables debug mode
"""
IS_DEBUG = bool(os.getenv("DEBUG"))
if IS_DEBUG:
    log.setLevel(logging.DEBUG)


class Word:
    """
    This class represents data rows of articles in SentiCoref 1.0
    """

    RE_ENTITY_TAG = re.compile(r"([A-Z]+)(?:\[(\d+)])?")
    RE_SENTIMENT = re.compile(r"(\d+).*?")
    RE_CHAIN_INDEX = re.compile(r"\*->(\d+)-(\d+)")
    RE_ENTITY_ID = re.compile(r"\*\[(\d+)\]")

    def __init__(self, data_row: List[str]):
        """
        Constructs Word object from a TSV row of SentiCoref 1.0 dataset.

        IMPORTANT: All indices are 0-based, ass oposed to 1-based indexing in the dataset.
        """
        # 1-14	 80-89	   Slovenije	ORG[2]|LOC[3]	_	       *->13-1|*->14-1	*[13]|*[14]
        word_id, char_pos, word, entity_tags, sentiment, __chain_indices, chain_ids = data_row

        wid_split = word_id.split("-")
        self.article_id, self.original_word_index = wid_split[0], int(wid_split[1]) - 1
        self.char_start, self.char_end = [int(p) for p in char_pos.split("-")]
        self.word_raw = word

        # Parse entity type (e.g. 'ORG[2]' or ORG[2]|LOC[3])
        # The [2] here indicates that this is the 2nd occurence of a named entity among all the named entitites.
        entity_types_matches = Word.RE_ENTITY_TAG.findall(entity_tags)
        self.entity_types = set()
        for named_entity_tag, _ in entity_types_matches:
            self.entity_types.add(named_entity_tag)

        # Strip sentiment name
        self.chain_sentiment = int(Word.RE_SENTIMENT.match(sentiment).group(1)) if sentiment != "_" else None

        # Dictionary of {chain_id: index}, where index is the index of this occurence
        # in coreference chain with id chain_id.
        # e.g. '*->15-1|*->19-1' to dict {15: 0, 19: 0}
        self.chain_indices = {int(chain_id): int(chain_index) - 1
                              for (chain_id, chain_index)
                              in Word.RE_CHAIN_INDEX.findall(__chain_indices)}

        # Set of chain IDs this word belongs to
        # e.g. '*[15]|*[19]' to set {15, 19}
        self.chain_ids = {int(chain_id) for chain_id in Word.RE_ENTITY_ID.findall(chain_ids)}

        self.lemma = None
        self.pos_tag = None
        self.word_sentiment = None

        # Set by Article
        self.word_index = None
        self.sentence_index = None

    def __str__(self):
        return " ".join([f"{name}[{type(value).__name__}]={value}" for name, value in self.__dict__.items()])

    def __repr__(self):
        return str(self)


class ArticleLoader:

    def __init__(self, senticoref_path):
        self.senticoref_path = senticoref_path

    def list_articles(self):
        return [a for a in natural_sort(os.listdir(self.senticoref_path))
                if a.endswith(".tsv")]

    def load_article(self, file_name):
        """
        Read a TSV file of a SentiCoref 1.0 article into an Article object.
        """
        article_id = file_name.replace(".tsv", "")

        with open(os.path.join(self.senticoref_path, file_name), encoding="UTF-8") as hnd:
            lines = [l.rstrip("\r\n") for l in hnd.readlines()]
            lines = [l for l in lines if len(l) > 0]

        header_lines = [l.split("=", 1) for l in lines if l.startswith("#")]
        headers = {name[1:]: value for name, value in header_lines}

        words = [Word(l.strip().split("\t")) for l in lines if not l.startswith("#")]

        return Article(article_id, headers, headers["Text"], words)


class Article:

    def __init__(self, article_id, headers, text, words: List[Word]):

        self.article_id = article_id

        self.text = text
        self.words = words
        for i, w in enumerate(self.words):
            w.word_index = i

        self.coreference_chains = self._generate_coreference_chains()
        self.chain_sentiments = self._generate_chain_sentiments()

        for chain_id, sentiment in self.chain_sentiments.items():
            if sentiment is None:
                log.warning(f"In article {self.article_id}, coreference chain {chain_id} has no sentiment")

    @property
    def num_words(self):
        return len(self.words)

    def word_at(self, index: int) -> Word:
        """
        Returns the word in this article at the specified index.
        """
        return self.words[index]

    def add_sentence_ids(self):
        sentence_id = 0
        for word in self.words:
            word.sentence_index = sentence_id
            if word.pos_tag == 'PUNCT' and word.word_raw == '.':
                sentence_id += 1

    def filter_words(self, predicate):
        self.words = [w for w in self.words if predicate(w)]
        for i, w in enumerate(self.words):
            w.word_index = i
        self.coreference_chains = self._generate_coreference_chains()
        self.chain_sentiments = self._generate_chain_sentiments()


    def _generate_coreference_chains(self) -> Dict[int, List[Word]]:
        """
        Returns dict of form {entity_id: wordlist} mapping entities to the coreference chain.
        Coreference chain is a list of Word objects found in self.words
        """
        coreference_dict = defaultdict(list)
        for word in self.words:
            if word.chain_ids:
                for el in word.chain_ids:
                    coreference_dict[el].append(word)
        return coreference_dict

    def _generate_chain_sentiments(self) -> Dict[int, Optional[int]]:
        """
        Returns a dict of form {chain_id: sentiment} containing sentiments of all coreference chains in this article.
        Note that some coreference chains do not have a sentiment assigned. In that case, sentiment is None.
        """
        return {entity_id: words[-1].chain_sentiment
                for entity_id, words in self.coreference_chains.items()}


def word_pos(word):
    properties = word.misc.split("|")
    values = {k: v for k, v in [p.split("=", 1) for p in properties]}
    return int(values["start_char"]), int(values["end_char"])


def natural_sort(xs):
    import re
    decomposed = [tuple(int(d) if d else s for d, s in re.findall(r"(\d+)|(\D+)", x))
                  for x in xs]
    return ["".join(str(s) for s in d)
            for d in sorted(decomposed)]


def cache_path(name):
    return os.path.join("data/cache", name)


def pipe_doc_cached(name, pipe, text):
    cpath = cache_path(name + ".stanzadoc.pickle")

    if os.path.exists(cpath):
        with open(cpath, 'rb') as f:
            doc = pickle.load(f)
        return doc

    doc = pipe(text)
    with open(cpath, 'wb') as f:
        pickle.dump(doc, f)

    return doc


"""
POS tagging and lemmatisation of the dataset.
TODO: Store augmented data
"""
if __name__ == "__main__":

    # stanza.download('sl')
    #
    # stanza_config = dict(
    #     lang="sl",
    #     processors="tokenize,pos,lemma",
    #     lemma_model_path="models/ssj500k_lemmatizer.pt",
    #     tokenize_pretokenized=True,
    # )
    #
    # pipe = stanza.Pipeline(**stanza_config)

    pipe = None

    article_loader = ArticleLoader("data/SentiCoref_1.0")
    senti_lexicon = JOBLexicon.load("data/JOB_1.0/job.tsv")

    articles = ["42.tsv"] if IS_DEBUG else article_loader.list_articles()

    feature_pipeline = features.FeaturePipeline(
        features.word_n_lemma(-3), features.word_n_pos(-3), features.word_n_word_sentiment(-3),  # 3 left
        features.word_n_lemma(-2), features.word_n_pos(-2), features.word_n_word_sentiment(-2),  # 2 left
        features.word_n_lemma(-1), features.word_n_pos(-1), features.word_n_word_sentiment(-1),  # 1 left

        features.word_n_lemma(1), features.word_n_pos(1), features.word_n_word_sentiment(1),  # 1 right
        features.word_n_lemma(2), features.word_n_pos(2), features.word_n_word_sentiment(2),  # 2 right
        features.word_n_lemma(3), features.word_n_pos(3), features.word_n_word_sentiment(3),  # 3 right

        features.entity_type
    )

    for art_name in articles:
        log.info(f"Processing {art_name}")

        art = article_loader.load_article(art_name)

        doc = pipe_doc_cached(art_name, pipe, art.text)

        # Construct position -> stanza.Word dictionary
        pos_dict = {}
        for sentence in doc.sentences:
            for tok in sentence.tokens:
                start, end = tok.start_char, tok.end_char
                pos_dict[(start, end)] = tok

        # Add lemma and POS tag to each word
        for word in art.words:
            loc = (word.char_start, word.char_end)
            if loc not in pos_dict:
                log.error(f"{art_name} missing key {loc}")
                continue

            stanza_token = pos_dict[loc]

            if len(stanza_token.words) > 1:
                log.warning(f"stanza.token.words for word {word.article_id}-{word.word_index} has len > 1")

            word.lemma = stanza_token.words[0].lemma
            word.pos_tag = stanza_token.words[0].upos

            if len(word.chain_ids) == 0 and word.pos_tag in ["NOUN", "VERB", "PROPN", "ADJ"]:
                word.word_sentiment = senti_lexicon.get_sentiment(word.lemma)

            # TODO: somehow add syntactic dependencies

        art.add_sentence_ids()
        art.filter_words(lambda w: w.pos_tag in ["NOUN", "VERB", "PROPN", "ADJ"])

        dfs = []
        for chain_id, words in art.coreference_chains.items():
            data = feature_pipeline(art, words)
            data["article_id"] = art.article_id
            data["chain_id"] = chain_id
            data["sentiment"] = art.chain_sentiments[chain_id]
            dfs.append(data)

        # TODO: use the data in some machine learning algorithm
