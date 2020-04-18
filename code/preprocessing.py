import logging
import os
import pickle
import re
from typing import Dict, List

import stanza

from code.sentiment_lexicon import JOBLexicon

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
        word_id, char_pos, word, entity_tag, sentiment, chain_indices, entity_ids = data_row

        wid_split = word_id.split("-")
        self.document, self.word_index = wid_split[0], int(wid_split[1]) - 1
        self.char_start, self.char_end = [int(p) for p in char_pos.split("-")]
        self.word = word

        # Strip entity_id that is sometimes present (e.g. 'ORG[2]')
        if entity_tag != "_":
            et_match = Word.RE_ENTITY_TAG.match(entity_tag).groups()
            self.entity_tag = et_match[0]
            self.entity_tag_UNKNOWN = int(et_match[1]) if et_match[1] is not None else None
        else:
            self.entity_tag = None
            self.entity_tag_UNKNOWN = None

        # Strip sentiment name
        self.sentiment = int(Word.RE_SENTIMENT.match(sentiment).group(1)) if sentiment != "_" else None

        # '*->15-1|*->19-1' to dict {15: 0, 19: 0}
        if chain_indices != "_":
            self.chain_indices = {int(entity_id): int(chain_index) - 1
                                  for (entity_id, chain_index)
                                  in Word.RE_CHAIN_INDEX.findall(chain_indices)}
        else:
            self.chain_indices = None

        # '*[15]|*[19]' to set {15, 19}
        self.entity_ids = {int(s) for s in Word.RE_ENTITY_ID.findall(entity_ids)} if entity_ids != "_" else None

        self.lemma = None
        self.pos_tag = None
        self.sentiment = None

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
        with open(os.path.join(self.senticoref_path, file_name)) as hnd:
            lines = [l.rstrip("\r\n") for l in hnd.readlines()]
            lines = [l for l in lines if len(l) > 0]

        header_lines = [l.split("=", 1) for l in lines if l.startswith("#")]
        headers = {name[1:]: value for name, value in header_lines}

        words = [Word(l.strip().split("\t")) for l in lines if not l.startswith("#")]

        return Article(headers, headers["Text"], words)


class Article:

    def __init__(self, headers, text, words: List[Word]):
        self.text = text
        self.words = words
        self._word_dict = {word.word_index: word for word in words}

        self.coreference_chains = self._generate_coreference_chains()

    @property
    def num_words(self):
        return len(self.words)

    def word_at(self, index: int) -> Word:
        """
        Returns the word in this article at the specified index.
        """
        return self._word_dict[index]

    def _generate_coreference_chains(self) -> Dict[int, List[Word]]:
        """
        TODO @Ela
        Returns dict of form {entity_id: wordlist} mapping entities to the coreference chain.
        Coreference chain is a list of Word objects found in self.words
        """
        pass


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

    stanza.download('sl')

    stanza_config = dict(
        lang="sl",
        processors="tokenize,pos,lemma",
        lemma_model_path="models/ssj500k_lemmatizer.pt",
        tokenize_pretokenized=True,
    )

    pipe = stanza.Pipeline(**stanza_config)

    article_loader = ArticleLoader("data/SentiCoref_1.0")
    senti_lexicon = JOBLexicon.load("data/JOB_1.0/job.tsv")

    articles = ["42.tsv"] if IS_DEBUG else article_loader.list_articles()

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
                log.warning(f"stanza.token.words for word {word.document}-{word.word_index} has len > 1")

            word.lemma = stanza_token.words[0].lemma
            word.pos_tag = stanza_token.words[0].upos

            if word.entity_ids is None and word.pos_tag in ["NOUN", "VERB", "PROPN", "ADJ"]:
                word.sentiment = senti_lexicon.get_sentiment(word.lemma)
                log.debug(f"{word.word} {word.lemma} {word.sentiment}")

            # TODO: somehow add syntactic dependencies
