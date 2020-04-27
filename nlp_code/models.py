import os

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
import numpy as np


class PrimitiveLinearRegression:

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

        print(train_idx)
        print(test_idx)

        X_train, Y_train = X.iloc[train_idx], Y.iloc[train_idx]
        X_test, Y_test = X.iloc[test_idx], Y.iloc[test_idx]

        return X_train, Y_train, X_test, Y_test

    def evaluate(self):

        self.data["entity_type_is_ORG"] = self.data.entity_type.apply(lambda x: 1 if x == "ORG" else 0)
        self.data["entity_type_is_PER"] = self.data.entity_type.apply(lambda x: 1 if x == "PER" else 0)
        self.data["entity_type_is_LOC"] = self.data.entity_type.apply(lambda x: 1 if x == "LOC" else 0)

        self.data["ORG_x_sentence_pos_neg"] = self.data["entity_type_is_ORG"] * self.data["sentence_pos_neg"]
        self.data["PER_x_sentence_pos_neg"] = self.data["entity_type_is_PER"] * self.data["sentence_pos_neg"]
        self.data["LOC_x_sentence_pos_neg"] = self.data["entity_type_is_LOC"] * self.data["sentence_pos_neg"]

        X_train, Y_train, X_test, Y_test = self.split(train_size=0.9)

        lr = LinearRegression(normalize=True)

        features = ["entity_type_is_ORG", "entity_type_is_PER", "entity_type_is_LOC",
                    "sentence_pos_neg",
                    "ORG_x_sentence_pos_neg", "PER_x_sentence_pos_neg", "LOC_x_sentence_pos_neg",
                    ]

        X_train = X_train[features]
        X_test = X_test[features]

        lr.fit(X_train, Y_train)

        Y_predicted = lr.predict(X_test)

        full = np.vstack([Y_test, Y_predicted]).T
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            masked = full[full[:, 0] == v]
            v_rmse = mean_squared_error(masked[:, 0], masked[:, 1], squared=False)
            print(f"RMSE for {v} = {v_rmse:.2f}")

        rmse = mean_squared_error(Y_test, Y_predicted, squared=False)
        print(f"RMSE for all = {rmse:.2f}")


if __name__ == '__main__':
    m = PrimitiveLinearRegression("data/features")
    m.evaluate()