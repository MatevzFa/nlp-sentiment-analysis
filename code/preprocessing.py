import sys
import stanza


class Article:

    def __init__(self, headers, text, words):
        self.text = text
        self.words = words

    @staticmethod
    def from_file(file_path):
        with open(file_path) as hnd:
            lines = [l.rstrip("\r\n") for l in hnd.readlines()]
            lines = [l for l in lines if len(l) > 0]

        header_lines = [l.split("=", 1) for l in lines if l.startswith("#")]
        headers = {name[1:]: value for name, value in header_lines}

        words = [l.split("\t") for l in lines if not l.startswith("#")]

        return Article(headers, headers["Text"], words)


def word_pos(word):
    properties = word.misc.split("|")
    values = {k: v for k, v in [p.split("=", 1) for p in properties]}
    return int(values["start_char"]), int(values["end_char"])


if __name__ == "__main__":

    stanza.download('sl')

    stanza_config = dict(
        lang="sl",
        processors="tokenize,pos,lemma",
        lemma_model_path="models/ssj500k_lemmatizer.pt",
    )

    pipe = stanza.Pipeline(**stanza_config)
    art = Article.from_file(sys.argv[1])

    doc = pipe(art.text)

    for sentence in doc.sentences:
        print([(w.lemma, w.upos, *word_pos(w)) for w in sentence.words])
