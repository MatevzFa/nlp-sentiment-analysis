import logging
from collections import defaultdict

log = logging.getLogger('senti_an').getChild("sentiment_lexicon")


class SentimentLexicon:
    PREFIX_SEARCH_LIMIT = 10

    def __init__(self, word_dict, min_sentiment, max_sentiment, neutral_sentiment):
        log.debug("creating word")
        self.min_sentiment = min_sentiment
        self.max_sentiment = max_sentiment
        self.neutral_sentiment = neutral_sentiment

        self._word_dict = word_dict

        # Enrich with prefixes shorter for at most 5 characters or until collision
        for word, sentiment in list(self._word_dict.items()):
            self.enrich(word, sentiment)

        self.word_to_prefix = dict()

    def get_sentiment(self, word: str) -> float:

        word = word.lower()

        if word in self._word_dict:
            return self._word_dict[word]
        elif word in self.word_to_prefix:
            return self._word_dict[self.word_to_prefix[word]]
        else:
            for prefix in self.prefix_iter(word, self.PREFIX_SEARCH_LIMIT):
                if prefix in self._word_dict:
                    self.word_to_prefix[word] = prefix
                    return self._word_dict[prefix]

            log.warning(f"Word {word} (or prefix) not present in the lexicon, using neutral_sentiment")
            self._word_dict[word] = self.neutral_sentiment
            self.enrich(word, self.neutral_sentiment)

            return self.neutral_sentiment

    def enrich(self, word, sentiment):
        for prefix in self.prefix_iter(word, self.PREFIX_SEARCH_LIMIT):
            if prefix in self._word_dict:
                break
            self._word_dict[prefix] = sentiment

    @staticmethod
    def prefix_iter(word, limit):
        prefix = word[:-1]
        while len(prefix) > 0 and len(word) - len(prefix) <= limit:
            yield prefix
            prefix = prefix[:-1]


class JOBLexicon(SentimentLexicon):

    def __init__(self, word_dict):
        super(JOBLexicon, self).__init__(word_dict, min_sentiment=-5, max_sentiment=5, neutral_sentiment=0)

    @staticmethod
    def load(dat_path):
        word_dict = {}
        with open(dat_path, encoding="UTF-8") as f:
            lines = [l.strip().split("\t") for l in f.readlines()[1:]]

        for line in lines:
            word, sentiment = line[0], float(line[1])
            word_dict[word] = sentiment

        return JOBLexicon(word_dict)
