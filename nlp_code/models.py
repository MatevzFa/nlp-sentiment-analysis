import logging
import os
from collections import Counter
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (classification_report, confusion_matrix,
                             mean_squared_error, plot_confusion_matrix,
                             precision_recall_fscore_support, r2_score,
                             roc_auc_score)
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


def balance_234(X: pd.DataFrame, y: pd.Series):
    counts = y.value_counts()

    n = min(counts[2.0], counts[3.0], counts[4.0])

    df = X.assign(sentiment=y)

    df_sample = pd.concat([
        df[df.sentiment == 1.0],
        df[df.sentiment == 2.0].sample(n, random_state=123),
        df[df.sentiment == 3.0].sample(n, random_state=456),
        df[df.sentiment == 4.0].sample(n, random_state=789),
        df[df.sentiment == 5.0],
    ])

    X = df_sample.drop('sentiment', axis=1)
    y = df_sample['sentiment']

    return X, y


def balance(X: pd.DataFrame, y: pd.Series):
    labels = sorted(list(y.unique()))
    counts = y.value_counts()

    n = min([counts[l] for l in labels])

    df = X.assign(sentiment=y)

    df_sample = pd.concat([df[df.sentiment == l].sample(n, random_state=123)
                           for l in labels])

    X = df_sample.drop('sentiment', axis=1)
    y = df_sample['sentiment']

    return X, y


def display_confmat(confmat):
    for row in confmat:
        print(" ".join([f"{v:.2f}" for v in row]))


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

    @property
    def name(self):
        return self.__class__.__name__


class PrimitiveDummy(BaseModel):

    def __init__(self, data_path):
        super().__init__(data_path)

    def evaluate(self):
        print(f"=== {self.name} ===")

        self.data = self.data.groupby(["article_id", "chain_id"]).agg({
            'sentiment': 'first',
        }).reset_index()

        X_train, Y_train, X_test, Y_test = self.split(train_size=0.8)

        describe(X_train, Y_train, X_test, Y_test)

        dummy = DummyClassifier(strategy="most_frequent")

        dummy.fit(X_train, Y_train)

        Y_predicted = dummy.predict(X_test)

        report_result(Y_test, Y_predicted)


class PrimitiveLinear(BaseModel):

    def __init__(self, data_path):
        super().__init__(data_path)

    def evaluate(self):
        print(f"=== {self.name} ===")

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


class PrimitiveRF(BaseModel):

    def __init__(self, data_path):
        super().__init__(data_path)

    def evaluate(self):
        print(f"=== {self.name} ===")

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


class PerWordLinear(BaseModel):

    def evaluate(self):
        print(f"=== {self.name} ===")

        # self.data = balance_234(self.data)

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


class PerWordDummy(BaseModel):

    def evaluate(self):
        print(f"=== {self.name} ===")

        X_train, Y_train, X_test, Y_test = self.split(train_size=0.8)

        describe(X_train, Y_train, X_test, Y_test)

        lr = DummyClassifier(strategy="most_frequent")

        lr.fit(X_train, Y_train)

        Y_predicted = lr.predict(X_test)

        p, r, f1, _ = precision_recall_fscore_support(Y_test, Y_predicted, labels=[1, 2, 3, 4, 5], warn_for=[])
        confmat = confusion_matrix(Y_test, Y_predicted, labels=[1, 2, 3, 4, 5], normalize='true')

        print(p)
        print(r)
        print(f1)

        display_confmat(confmat)
        print(classification_report(Y_test, Y_predicted, digits=3))


class PerWordRF(BaseModel):

    def evaluate(self):
        print(f"=== {self.name} ===")

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

        X_train, Y_train = balance(X_train, Y_train)

        describe(X_train, Y_train, X_test, Y_test)

        lr = RandomForestClassifier(random_state=444)

        X_train = X_train[features]
        X_test = X_test[features]

        lr.fit(X_train, Y_train)

        Y_predicted = lr.predict(X_test)

        p, r, f1, _ = precision_recall_fscore_support(Y_test, Y_predicted, labels=[1, 2, 3, 4, 5])
        confmat = confusion_matrix(Y_test, Y_predicted, labels=[1, 2, 3, 4, 5], normalize='true')

        print(p)
        print(r)
        print(f1)

        display_confmat(confmat)
        print(classification_report(Y_test, Y_predicted, digits=3))

        plt.figure(figsize=(4, 4))
        plot_confusion_matrix(lr, X_test, Y_test, labels=[1, 2, 3, 4, 5],
                              normalize='true',  cmap=plt.cm.Blues, ax=plt.gca())
        plt.savefig(f'report/figures/confmat_{self.name}.pdf', bbox_inches='tight')

        # report_result(Y_test, Y_predicted)


class PerChainDummy(BaseModel):

    def evaluate(self, label_joining):
        print(f"=== {self.name} ===")

        self.data = self.data.groupby(["article_id", "chain_id"]).agg({
            'sentiment': 'first',
        }).reset_index()

        sentiment_mapping = {
            1: 2 if label_joining else 1,
            2: 2,
            3: 3,
            4: 4,
            5: 4 if label_joining else 5,
        }
        self.data.sentiment = self.data.sentiment.apply(lambda x: sentiment_mapping[x])
        labels = sorted(self.data.sentiment.unique())

        X_train, Y_train, X_test, Y_test = self.split(train_size=0.8)

        describe(X_train, Y_train, X_test, Y_test)

        lr = DummyClassifier(strategy="most_frequent")

        lr.fit(X_train, Y_train)

        Y_predicted = lr.predict(X_test)

        p, r, f1, _ = precision_recall_fscore_support(Y_test, Y_predicted, labels=labels, warn_for=[])
        confmat = confusion_matrix(Y_test, Y_predicted, labels=labels, normalize='true')

        print(p)
        print(r)
        print(f1)

        display_confmat(confmat)
        print(classification_report(Y_test, Y_predicted, digits=3))


class PerChainRF(BaseModel):

    def evaluate(self, label_joining):
        print(f"=== {self.name} ===")

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

        self.data = self.data.groupby(["article_id", "chain_id"]).agg({
            'sentence_pos_neg': 'mean',
            'sentence_sentiment': 'mean',
            'word_n_word_sentiment':  'sum',
            'entity_type_is_ORG': 'first', 'entity_type_is_PER': 'first', 'entity_type_is_LOC': 'first',
            'ORG_x_sentence_pos_neg': 'sum', 'PER_x_sentence_pos_neg': 'sum', 'LOC_x_sentence_pos_neg': 'sum',

            'sentiment': 'first',
        }).reset_index()

        sentiment_mapping = {
            1: 2 if label_joining else 1,
            2: 2,
            3: 3,
            4: 4,
            5: 4 if label_joining else 5,
        }
        self.data.sentiment = self.data.sentiment.apply(lambda x: sentiment_mapping[x])
        labels = sorted(self.data.sentiment.unique())

        X_train, Y_train, X_test, Y_test = self.split(train_size=0.8)

        if label_joining:
            X_train, Y_train = balance(X_train, Y_train)
        else:
            X_train, Y_train = balance_234(X_train, Y_train)

        describe(X_train, Y_train, X_test, Y_test)

        lr = RandomForestClassifier(n_estimators=100, random_state=123123)

        X_train = X_train[features]
        X_test = X_test[features]

        lr.fit(X_train, Y_train)

        Y_predicted = lr.predict(X_test)

        p, r, f1, _ = precision_recall_fscore_support(Y_test, Y_predicted, labels=labels)
        confmat = confusion_matrix(Y_test, Y_predicted, labels=labels, normalize='true')

        print(p)
        print(r)
        print(f1)

        display_confmat(confmat)
        print(classification_report(Y_test, Y_predicted, digits=3))

        display_labels = ['1, 2', '3', '4, 5'] if label_joining else labels

        plt.figure(figsize=(4, 4))
        plot_confusion_matrix(lr, X_test, Y_test, display_labels=display_labels, normalize='true', cmap=plt.cm.Blues, ax=plt.gca())
        plt.savefig(f"report/figures/confmat_{self.name}{'_joined' if label_joining else ''}.pdf",
                    bbox_inches='tight')

        # report_result(Y_test, Y_predicted)


if __name__ == '__main__':
    dm = PrimitiveDummy("data/features")
    dm.evaluate()

    pc = PerWordLinear("data/features")
    pc.evaluate()

    dbc = PerWordDummy("data/features")
    dbc.evaluate()

    bc = PerWordRF("data/features")
    bc.evaluate()

    dcc1 = PerChainDummy("data/features")
    dcc1.evaluate(label_joining=False)

    cc1 = PerChainRF("data/features")
    cc1.evaluate(label_joining=False)

    dcc2 = PerChainDummy("data/features")
    dcc2.evaluate(label_joining=True)

    cc2 = PerChainRF("data/features")
    cc2.evaluate(label_joining=True)
