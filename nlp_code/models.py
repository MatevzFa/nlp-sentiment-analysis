import logging
import os

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error, precision_recall_fscore_support, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.naive_bayes import GaussianNB, MultinomialNB


def binarize_column(df: pd.DataFrame, column: str):
    """
    Performs in-place column binarisation by adding a 1/0 column for each unique value of 'column'.
    Returns list of new columns.
    """
    new_columns = []
    for value in df[column].unique():
        new_col_name = f"{column}_is_{value}"
        df[new_col_name] = df[column].apply(lambda x: 1 if x == value else 0)
        new_columns.append(new_col_name)

    return new_columns


def describe(X_train, Y_train, X_test, Y_test):
    print(f"TRAIN SIZE {len(Y_train):10d}")
    for v in sorted(Y_train.unique()):
        print(f"-- {v} {len(Y_train[Y_train == v]):10d}")

    print(f"TEST SIZE {len(Y_test):10d}")
    for v in sorted(Y_test.unique()):
        print(f"-- {v} {len(Y_test[Y_test == v]):10d}")


def report_result(Y_test, Y_predicted):
    score = r2_score(Y_test, Y_predicted)

    full = np.vstack([Y_test, Y_predicted]).T
    for v in sorted(Y_test.unique()):
        masked = full[full[:, 0] == v]
        v_rmse = mean_squared_error(masked[:, 0], masked[:, 1], squared=False)
        print(f"RMSE for {v} = {v_rmse:.2f}")

    rmse = mean_squared_error(Y_test, Y_predicted, squared=False)
    print(f"RMSE for all = {rmse:.2f}")
    print(f"    r2_score = {score:.2f}")


def balance_234(data: pd.DataFrame):
    counts = data.groupby('sentiment').size()

    n = min(counts[2.0], counts[3.0], counts[4.0])

    return pd.concat([
        data[data.sentiment == 1.0],
        data[data.sentiment == 2.0].sample(n),
        data[data.sentiment == 3.0].sample(n),
        data[data.sentiment == 4.0].sample(n),
        data[data.sentiment == 5.0],
    ])


class BaseModel:

    def __init__(self, data_path):
        dfs = []

        for file in os.listdir(data_path):
            if file.endswith(".tsv"):
                dfs.append(pd.read_csv(os.path.join(data_path, file), sep="\t"))

        self.data = pd.concat(dfs)
        self.data = self.data[self.data.sentiment.notnull()]

    def split(self, train_size):
        data = self.data

        X = data
        Y = data["sentiment"]
        groups = data.groupby(["article_id", "chain_id"]).ngroup()

        splitter = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=1234321)

        train_idx, test_idx = next(splitter.split(X, Y, groups))

        X_train, Y_train = X.iloc[train_idx], Y.iloc[train_idx]
        X_test, Y_test = X.iloc[test_idx], Y.iloc[test_idx]

        return X_train, Y_train, X_test, Y_test


class Dummy(BaseModel):

    def __init__(self, data_path):
        super().__init__(data_path)

    def evaluate(self):
        print(f"=== {self.__class__.__name__} ===")

        self.data = self.data.groupby(["article_id", "chain_id"]).agg({
            'sentiment': 'first',
        }).reset_index()

        X_train, Y_train, X_test, Y_test = self.split(train_size=0.8)

        describe(X_train, Y_train, X_test, Y_test)

        dummy = DummyClassifier(strategy="most_frequent")

        dummy.fit(X_train, Y_train)

        Y_predicted = dummy.predict(X_test)

        report_result(Y_test, Y_predicted)


class PrimitiveLinearRegression(BaseModel):

    def __init__(self, data_path):
        super().__init__(data_path)

    def evaluate(self):
        print(f"=== {self.__class__.__name__} ===")

        self.data = self.data.groupby(["article_id", "chain_id"]).agg({
            'entity_type': 'first',
            'sentence_neg_count': 'sum',
            'sentence_pos_count': 'sum',
            'sentiment': 'first',
        }).reset_index()

        self.data["sentence_pos_neg"] = (self.data["sentence_pos_count"] + 1) / (self.data["sentence_neg_count"] + 1)

        self.data["entity_type_is_ORG"] = self.data.entity_type.apply(lambda x: 1 if x == "ORG" else 0)
        self.data["entity_type_is_PER"] = self.data.entity_type.apply(lambda x: 1 if x == "PER" else 0)
        self.data["entity_type_is_LOC"] = self.data.entity_type.apply(lambda x: 1 if x == "LOC" else 0)

        self.data["ORG_x_sentence_pos_neg"] = self.data["entity_type_is_ORG"] * self.data["sentence_pos_neg"]
        self.data["PER_x_sentence_pos_neg"] = self.data["entity_type_is_PER"] * self.data["sentence_pos_neg"]
        self.data["LOC_x_sentence_pos_neg"] = self.data["entity_type_is_LOC"] * self.data["sentence_pos_neg"]

        X_train, Y_train, X_test, Y_test = self.split(train_size=0.8)

        describe(X_train, Y_train, X_test, Y_test)

        lr = LinearRegression(normalize=True)

        features = ["entity_type_is_ORG", "entity_type_is_PER", "entity_type_is_LOC",
                    "sentence_pos_neg",
                    "ORG_x_sentence_pos_neg", "PER_x_sentence_pos_neg", "LOC_x_sentence_pos_neg",
                    ]

        X_train = X_train[features]
        X_test = X_test[features]

        lr.fit(X_train, Y_train)

        Y_predicted = lr.predict(X_test)

        report_result(Y_test, Y_predicted)


class PrimitiveRandomForest(BaseModel):

    def __init__(self, data_path):
        super().__init__(data_path)

    def evaluate(self):
        print(f"=== {self.__class__.__name__} ===")

        self.data = self.data.groupby(["article_id", "chain_id"]).agg({
            'entity_type': 'first',
            'sentence_neg_count': 'sum',
            'sentence_pos_count': 'sum',
            'sentiment': 'first',
        }).reset_index()

        self.data["sentence_pos_neg"] = (self.data["sentence_pos_count"] + 1) / (self.data["sentence_neg_count"] + 1)

        self.data["entity_type_is_ORG"] = self.data.entity_type.apply(lambda x: 1 if x == "ORG" else 0)
        self.data["entity_type_is_PER"] = self.data.entity_type.apply(lambda x: 1 if x == "PER" else 0)
        self.data["entity_type_is_LOC"] = self.data.entity_type.apply(lambda x: 1 if x == "LOC" else 0)

        self.data["ORG_x_sentence_pos_neg"] = self.data["entity_type_is_ORG"] * self.data["sentence_pos_neg"]
        self.data["PER_x_sentence_pos_neg"] = self.data["entity_type_is_PER"] * self.data["sentence_pos_neg"]
        self.data["LOC_x_sentence_pos_neg"] = self.data["entity_type_is_LOC"] * self.data["sentence_pos_neg"]

        X_train, Y_train, X_test, Y_test = self.split(train_size=0.8)

        describe(X_train, Y_train, X_test, Y_test)

        rf = RandomForestRegressor(n_estimators=100, random_state=12321)

        features = ["entity_type_is_ORG", "entity_type_is_PER", "entity_type_is_LOC",
                    "sentence_pos_neg",
                    "ORG_x_sentence_pos_neg", "PER_x_sentence_pos_neg", "LOC_x_sentence_pos_neg",
                    ]

        X_train = X_train[features]
        X_test = X_test[features]

        rf.fit(X_train, Y_train)

        Y_predicted = rf.predict(X_test)

        report_result(Y_test, Y_predicted)


class PerChainWordModel(BaseModel):

    def evaluate(self):
        print(f"=== {self.__class__.__name__} ===")

        self.data = balance_234(self.data)

        features = [
            "sentence_pos_neg",
            "sentence_sentiment"
        ]

        self.data[["word_1_word_sentiment",
                   "word_2_word_sentiment",
                   "word_3_word_sentiment",
                   "word_-1_word_sentiment",
                   "word_-2_word_sentiment",
                   "word_-3_word_sentiment"]] = self.data[["word_1_word_sentiment",
                                                           "word_2_word_sentiment",
                                                           "word_3_word_sentiment",
                                                           "word_-1_word_sentiment",
                                                           "word_-2_word_sentiment",
                                                           "word_-3_word_sentiment"]].fillna(value=0)

        self.data["word_n_word_sentiment"] = self.data[["word_1_word_sentiment",
                                                        "word_2_word_sentiment",
                                                        "word_3_word_sentiment",
                                                        "word_-1_word_sentiment",
                                                        "word_-2_word_sentiment",
                                                        "word_-3_word_sentiment"]].sum(axis=1)
        features += ["word_n_word_sentiment"]

        self.data["entity_type_is_ORG"] = self.data.entity_type.apply(lambda x: 1 if x == "ORG" else 0)
        self.data["entity_type_is_PER"] = self.data.entity_type.apply(lambda x: 1 if x == "PER" else 0)
        self.data["entity_type_is_LOC"] = self.data.entity_type.apply(lambda x: 1 if x == "LOC" else 0)
        features += ["entity_type_is_ORG", "entity_type_is_PER", "entity_type_is_LOC"]

        self.data["ORG_x_sentence_pos_neg"] = self.data["entity_type_is_ORG"] * self.data["sentence_pos_neg"]
        self.data["PER_x_sentence_pos_neg"] = self.data["entity_type_is_PER"] * self.data["sentence_pos_neg"]
        self.data["LOC_x_sentence_pos_neg"] = self.data["entity_type_is_LOC"] * self.data["sentence_pos_neg"]
        features += ["ORG_x_sentence_pos_neg", "PER_x_sentence_pos_neg", "LOC_x_sentence_pos_neg"]

        X_train, Y_train, X_test, Y_test = self.split(train_size=0.8)

        describe(X_train, Y_train, X_test, Y_test)

        lr = LinearRegression(normalize=True)

        X_train = X_train[features]
        X_test = X_test[features]

        lr.fit(X_train, Y_train)

        Y_predicted = lr.predict(X_test)

        report_result(Y_test, Y_predicted)


def binary_sentiment(sentiment):
    if sentiment < 3:
        return 0
    else:
        return 1


class DummyBinaryClassifciation(BaseModel):

    def evaluate(self):
        print(f"=== {self.__class__.__name__} ===")

        self.data.sentiment = self.data.sentiment.apply(binary_sentiment)

        X_train, Y_train, X_test, Y_test = self.split(train_size=0.8)

        describe(X_train, Y_train, X_test, Y_test)

        lr = DummyClassifier(strategy="most_frequent")

        lr.fit(X_train, Y_train)

        Y_predicted = lr.predict(X_test)

        p, r, f1, _ = precision_recall_fscore_support(Y_test, Y_predicted, labels=[0, 1], warn_for=[])

        print(p)
        print(r)
        print(f1)


class BinaryClassifciation(BaseModel):

    def evaluate(self):
        print(f"=== {self.__class__.__name__} ===")

        self.data.sentiment = self.data.sentiment.apply(binary_sentiment)

        features = [
            "sentence_pos_neg",
            "sentence_sentiment"
        ]

        self.data[["word_1_word_sentiment",
                   "word_2_word_sentiment",
                   "word_3_word_sentiment",
                   "word_-1_word_sentiment",
                   "word_-2_word_sentiment",
                   "word_-3_word_sentiment"]] = self.data[["word_1_word_sentiment",
                                                           "word_2_word_sentiment",
                                                           "word_3_word_sentiment",
                                                           "word_-1_word_sentiment",
                                                           "word_-2_word_sentiment",
                                                           "word_-3_word_sentiment"]].fillna(value=0)

        self.data["word_n_word_sentiment"] = self.data[["word_1_word_sentiment",
                                                        "word_2_word_sentiment",
                                                        "word_3_word_sentiment",
                                                        "word_-1_word_sentiment",
                                                        "word_-2_word_sentiment",
                                                        "word_-3_word_sentiment"]].sum(axis=1)
        features += ["word_n_word_sentiment"]

        self.data["entity_type_is_ORG"] = self.data.entity_type.apply(lambda x: 1 if x == "ORG" else 0)
        self.data["entity_type_is_PER"] = self.data.entity_type.apply(lambda x: 1 if x == "PER" else 0)
        self.data["entity_type_is_LOC"] = self.data.entity_type.apply(lambda x: 1 if x == "LOC" else 0)
        features += ["entity_type_is_ORG", "entity_type_is_PER", "entity_type_is_LOC"]

        self.data["ORG_x_sentence_pos_neg"] = self.data["entity_type_is_ORG"] * self.data["sentence_pos_neg"]
        self.data["PER_x_sentence_pos_neg"] = self.data["entity_type_is_PER"] * self.data["sentence_pos_neg"]
        self.data["LOC_x_sentence_pos_neg"] = self.data["entity_type_is_LOC"] * self.data["sentence_pos_neg"]
        features += ["ORG_x_sentence_pos_neg", "PER_x_sentence_pos_neg", "LOC_x_sentence_pos_neg"]

        X_train, Y_train, X_test, Y_test = self.split(train_size=0.8)

        describe(X_train, Y_train, X_test, Y_test)

        lr = RandomForestClassifier()

        X_train = X_train[features]
        X_test = X_test[features]

        lr.fit(X_train, Y_train)

        Y_predicted = lr.predict(X_test)

        p, r, f1, _ = precision_recall_fscore_support(Y_test, Y_predicted, labels=[0, 1])

        print(p)
        print(r)
        print(f1)

        # report_result(Y_test, Y_predicted)


if __name__ == '__main__':
    dm = Dummy("data/features")
    dm.evaluate()

    # m = PrimitiveLinearRegression("data/features")
    # m.evaluate()
    #
    # rf = PrimitiveRandomForest("data/features")
    # rf.evaluate()

    pc = PerChainWordModel("data/features")
    pc.evaluate()

    dbc = DummyBinaryClassifciation("data/features")
    dbc.evaluate()

    bc = BinaryClassifciation("data/features")
    bc.evaluate()
