
import random
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix, f1_score, plot_confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tqdm import tqdm
from transformers import (TF2_WEIGHTS_NAME, BertConfig, BertTokenizer,
                          TFBertForTokenClassification, create_optimizer)
from transformers.modeling_tf_bert import TFBertForSequenceClassification

from _collections import defaultdict
from nlp_code.articles import ArticleLoader
from nlp_code.models import display_confmat
import textwrap

BERT_MODEL = "models/slo-hr-en-bert-pytorch"

EMBEDDING_DIM = 512
CNN_FILTERS = 64
DNN_UNITS = 128
OUTPUT_CLASSES = 3
DROPOUT_RATE = 0.0000
NB_EPOCHS = 3
BATCH_SIZE = 256


def convert(word_sequences):
    return tokenizer.batch_encode_plus(
        word_sequences,
        add_special_tokens=True,
        max_length=EMBEDDING_DIM,
        pad_to_max_length=True)["input_ids"]


class CustomSentiCorefModel(tf.keras.Model):
    """
    Provided by Slavko Žitnik in

    'Transformers architecture and BERT'.

    Jupyter Notebook, materials for Natural Language Processing.
    University of Ljubljana, Faculty of Computer and Information Science
    """

    def __init__(self,
                 vocabulary_size,
                 embedding_dimensions=EMBEDDING_DIM,
                 cnn_filters=50,
                 dnn_units=512,
                 model_output_classes=3,
                 dropout_rate=0.1,
                 training=False,
                 name="custom_imdb_model"):
        super(CustomSentiCorefModel, self).__init__(name=name)

        self.embedding = layers.Embedding(vocabulary_size,
                                          embedding_dimensions)
        self.cnn_layer1 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=2,
                                        padding="valid",
                                        activation="relu")
        self.cnn_layer2 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=3,
                                        padding="valid",
                                        activation="relu")
        self.cnn_layer3 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=4,
                                        padding="valid",
                                        activation="relu")
        self.pool = layers.GlobalMaxPool1D()

        self.dense_1 = layers.Dense(units=dnn_units, activation="relu")
        self.dropout = layers.Dropout(rate=dropout_rate)
        if model_output_classes == 2:
            self.last_dense = layers.Dense(units=1,
                                           activation="sigmoid")
        else:
            self.last_dense = layers.Dense(units=model_output_classes,
                                           activation="softmax")

    def call(self, inputs, training):
        l = self.embedding(inputs)
        l_1 = self.cnn_layer1(l)
        l_1 = self.pool(l_1)
        l_2 = self.cnn_layer2(l)
        l_2 = self.pool(l_2)
        # l_3 = self.cnn_layer3(l)
        # l_3 = self.pool(l_3)

        # concatenated = tf.concat([l_1, l_2, l_3], axis=-1)
        concatenated = tf.concat([l_1, l_2], axis=-1)
        concatenated = self.dense_1(concatenated)
        concatenated = self.dropout(concatenated, training)
        model_output = self.last_dense(concatenated)

        return model_output


if __name__ == "__main__":

    article_loader = ArticleLoader("data/SentiCoref_1.0")

    model = TFBertForSequenceClassification.from_pretrained(BERT_MODEL, from_pt=True)
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True)

    # dict {(art_name, entity): [Words]}
    #   where [Words] are sentences with this entity.
    data = dict()
    data_sentiments = dict()

    for art_name in article_loader.list_articles():
        art = article_loader.load_article(art_name)
        art.refresh()

        for entity_id, words in art.coreference_chains.items():
            if art.chain_sentiments[entity_id] is None:
                continue

            processed_sentences = set()

            word_sequence = []

            # Construct sequence of words for this entity.
            for w in words:
                if w.sentence_index in processed_sentences:
                    continue
                processed_sentences.add(w.sentence_index)

                for sw in art.iter_sentence(w.sentence_index):
                    word_sequence.append(sw.word_raw)

            data[(art_name, entity_id)] = word_sequence
            data_sentiments[(art_name, entity_id)] = art.chain_sentiments[entity_id]

    mapper = {
        1: 0,
        2: 0,
        3: 1,
        4: 2,
        5: 2,
    }

    # Data balancing
    data_sentiments = {k: mapper[v] for k, v in data_sentiments.items()}

    ks_2 = [k for k, v in data_sentiments.items() if v == 0]
    ks_3 = [k for k, v in data_sentiments.items() if v == 1]
    ks_4 = [k for k, v in data_sentiments.items() if v == 2]

    sample_size = min(len(ks_2), len(ks_3), len(ks_4))

    datapoints = []
    random.seed(123123)
    datapoints.extend(random.sample(ks_2, sample_size))
    datapoints.extend(random.sample(ks_3, sample_size))
    datapoints.extend(random.sample(ks_4, sample_size))

    # Split data for train/val/test
    train, val_test = train_test_split(datapoints, train_size=.8, random_state=99999999)
    val, test = train_test_split(val_test, train_size=.5, random_state=88888888)

    print(f"train={len(train)} val={len(val)} test={len(test)}")

    X_train = convert([data[k] for k in train])
    y_train = np.array([data_sentiments[k] for k in train])

    X_val = convert([data[k] for k in val])
    y_val = np.array([data_sentiments[k] for k in val])

    X_test = convert([data[k] for k in test])
    y_test = np.array([data_sentiments[k] for k in test])

    train_ds = tf.data.Dataset.from_tensor_slices((tf.constant(X_train), tf.constant(y_train))) \
        .batch(BATCH_SIZE).repeat(5)

    val_ds = tf.data.Dataset.from_tensor_slices((tf.constant(X_val), tf.constant(y_val))) \
        .batch(BATCH_SIZE)

    test_ds = tf.data.Dataset.from_tensor_slices((tf.constant(X_test), tf.constant(y_test))) \
        .batch(BATCH_SIZE)

    custom_model = CustomSentiCorefModel(vocabulary_size=len(tokenizer.vocab),
                                         embedding_dimensions=EMBEDDING_DIM,
                                         cnn_filters=CNN_FILTERS,
                                         dnn_units=DNN_UNITS,
                                         model_output_classes=OUTPUT_CLASSES,
                                         dropout_rate=DROPOUT_RATE)

    if OUTPUT_CLASSES == 2:
        custom_model.compile(loss="binary_crossentropy",
                             optimizer="adam",
                             metrics=["accuracy"])
    else:
        custom_model.compile(loss="sparse_categorical_crossentropy",
                             optimizer="adam",
                             metrics=["sparse_categorical_accuracy"])

    custom_model.fit(train_ds, epochs=NB_EPOCHS, validation_data=val_ds)

    Y_predicted = np.argmax(custom_model.predict(test_ds), axis=1)
    Y_test = np.array(y_test)

    labels = [0, 1, 2]

    print(f"F1 score: {f1_score(Y_test, Y_predicted, average='macro')}")

    p, r, f1, _ = precision_recall_fscore_support(Y_test, Y_predicted, labels=labels)
    confmat = confusion_matrix(Y_test, Y_predicted, labels=labels, normalize='true')

    print(p)
    print(r)
    print(f1)

    display_confmat(confmat)
    print(classification_report(Y_test, Y_predicted, digits=3))

    cmatdisp = ConfusionMatrixDisplay(confmat, display_labels=['1, 2', '3', '4, 5'])
    cmatdisp.plot(cmap=plt.cm.Blues)
    plt.savefig('report/figures/confmat_CustomSentiCorefModel.pdf', bbox_inches='tight')

    print(textwrap.dedent(f"""
        EMBEDDING_DIM = {EMBEDDING_DIM}
        CNN_FILTERS = {CNN_FILTERS}
        DNN_UNITS = {DNN_UNITS}
        OUTPUT_CLASSES = {OUTPUT_CLASSES}
        DROPOUT_RATE = {DROPOUT_RATE}
        NB_EPOCHS = {NB_EPOCHS}
        BATCH_SIZE = {BATCH_SIZE}
    """))
