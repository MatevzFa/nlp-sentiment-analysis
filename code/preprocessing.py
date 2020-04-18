import logging
import os
import re
from typing import List

import stanza

log = logging.getLogger('senti_an')
log.setLevel(logging.INFO)


class Word:
    """
    This class represents data rows of articles in SentiCoref 1.0
    """

    RE_ENTITY_TAG = re.compile(r"([A-Z]+)(?:\[(\d+)])?")
    RE_SENTIMENT = re.compile(r"(\d+).*?")
    RE_CHAIN_INDEX = re.compile(r"\*->(\d+)-(\d+)")
    RE_ENTITY_ID = re.compile(r"\*\[(\d+)\]")

    def __init__(self, data_row):
        word_id, char_pos, word, entity_tag, sentiment, chain_indices, entity_ids = data_row

        self.document_id, self.word_index = [int(p) for p in word_id.split("-")]
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

        # '*->15-1|*->19-1' to dict {15: 1, 19:1}
        if chain_indices != "_":
            self.chain_indices = {entity_ids: chain_indices
                                  for (entity_ids, chain_indices)
                                  in Word.RE_CHAIN_INDEX.findall(chain_indices)}
        else:
            self.chain_indices = None

        # '*[15]|*[19]' to set {15, 19}
        self.entities = {int(s) for s in Word.RE_ENTITY_ID.findall(entity_ids)} if entity_ids != "_" else None

    def __str__(self):
        return " ".join([f"{name}={value}" for name, value in self.__dict__.items()])

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

    for art_name in ["42.tsv"]:
        # for art_name in article_loader.list_articles():
        log.info(f"Processing {art_name}")

        art = article_loader.load_article(art_name)

        for word in art.words:
            print(word)

        # doc = pipe(art.text)
        #
        # # Construct position -> stanza.Word dictionary
        # pos_dict = {}
        # for sentence in doc.sentences:
        #     for tok in sentence.tokens:
        #         start, end = tok.start_char, tok.end_char
        #         pos_dict[f"{start}-{end}"] = tok
        #
        # for word in art.words:
        #     loc = word[1]
        #     if loc not in pos_dict:
        #         log.error(f"{art_name} missing key {loc}")
        #         continue
        #
        #     stanza_token = pos_dict[loc]
        #
        #     info = [(w.lemma, w.upos) for w in stanza_token.words]
        #     print(word[2], word[3], info)
