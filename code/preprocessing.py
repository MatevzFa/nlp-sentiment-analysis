import sys
import stanza
import os
from pprint import pprint
import logging

log = logging.getLogger('senti_an')
log.setLevel(logging.INFO)


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

        words = [l.split("\t") for l in lines if not l.startswith("#")]

        return Article(headers, headers["Text"], words)


class Article:

    def __init__(self, headers, text, words):
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

        doc = pipe(art.text)

        # Construct position -> stanza.Word dictionary
        pos_dict = {}
        for sentence in doc.sentences:
            for tok in sentence.tokens:
                start, end = tok.start_char, tok.end_char
                pos_dict[f"{start}-{end}"] = tok

        for word in art.words:
            loc = word[1]
            if loc not in pos_dict:
                log.error(f"{art_name} missing key {loc}")
                continue

            stanza_token = pos_dict[loc]

            info = [(w.lemma, w.upos) for w in stanza_token.words]
            print(word[2], word[3], info)
