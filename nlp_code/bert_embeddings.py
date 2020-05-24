import random
import textwrap
from collections import Counter, defaultdict
from pprint import pprint
import textwrap

from transformers import BertTokenizer, TFBertModel, TFBertPreTrainedModel, TFBertMainLayer
import tensorflow as tf
from tensorflow.keras import layers

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix, f1_score, plot_confusion_matrix, precision_recall_fscore_support
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from nlp_code.articles import ArticleLoader
from nlp_code.models import display_confmat


BERT_MODEL = "models/slo-hr-en-bert-pytorch"

EMBEDDING_DIM = 128
CNN_FILTERS = 100
DNN_UNITS = 256
OUTPUT_CLASSES = 3
DROPOUT_RATE = 0.2
NB_EPOCHS = 6
BATCH_SIZE = 32


#########################################

def convert(word_sequences):
    return bert_tokenizer.batch_encode_plus(
        word_sequences,
        add_special_tokens=True,
        max_length=EMBEDDING_DIM,
        pad_to_max_length=True)["input_ids"]


##########################
class BertEmbeddingsSentiCoref(TFBertPreTrainedModel):
    """
    Provided by Slavko Žitnik in 

    'Transformers architecture and BERT'.

    Jupyter Notebook, materials for Natural Language Processing.
    University of Ljubljana, Faculty of Computer and Information Science
    """

    def __init__(self, config,
                 embedding_dimensions=EMBEDDING_DIM,
                 cnn_filters=100,
                 dnn_units=256,
                 model_output_classes=3,
                 dropout_rate=0.1,
                 training=False,
                 name=("bert_embeddings"),
                 *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.bert = TFBertMainLayer(config, name="bert", trainable = False)

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
    
    def call(self, inputs, training = False, **kwargs):        
        bert_outputs = self.bert(inputs, training = training, **kwargs)
        
        l_1 = self.cnn_layer1(bert_outputs[0]) 
        l_1 = self.pool(l_1) 
        l_2 = self.cnn_layer2(bert_outputs[0]) 
        l_2 = self.pool(l_2)
        l_3 = self.cnn_layer3(bert_outputs[0])
        l_3 = self.pool(l_3) 

        concatenated = tf.concat([l_1, l_2, l_3], axis=-1)
        concatenated = self.dense_1(concatenated)
        concatenated = self.dropout(concatenated, training)
        model_output = self.last_dense(concatenated)

        return model_output

#################################################
def neural_load_articles(article_loader):
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

    return data, data_sentiments


def neural_join_labels(data_sentiments):
    mapper = {
        1: 0,
        2: 0,
        3: 1,
        4: 2,
        5: 2,
    }

    return {k: mapper[v] for k, v in data_sentiments.items()}


def neural_train_val_test_split(data_sentiments: dict):
    random.seed(123)

    datapoints = list(data_sentiments)
    labels = set(data_sentiments.values())

    # Split indexes for train/val/test
    train, val_test = train_test_split(datapoints, train_size=.8, random_state=99999999)
    val, test = train_test_split(val_test, train_size=.5, random_state=88888888)

    counts = Counter([data_sentiments[k] for k in train])
    _, n = counts.most_common()[-1]

    new_train = []
    for l in labels:
        candidates = [k for k in train if data_sentiments[k] == l]
        selected = random.sample(candidates, n)
        new_train.extend(selected)

    return new_train, val, test


def neural_describe_data(data_sentiments, train, val, test):

    counts_train = Counter([data_sentiments[k] for k in train])
    counts_val = Counter([data_sentiments[k] for k in val])
    counts_test = Counter([data_sentiments[k] for k in test])

    print("TRAIN:")
    for k, v in sorted(counts_train.items()):
        print(f"  label={k:2d} x {v}")

    print("VALIDATION:")
    for k, v in sorted(counts_val.items()):
        print(f"  label={k:2d} x {v}")

    print("TEST:")
    for k, v in sorted(counts_test.items()):
        print(f"  label={k:2d} x {v}")

#################################################

if __name__ == "__main__":

    bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True)
    model = TFBertModel.from_pretrained(BERT_MODEL, from_pt=True)

    article_loader = ArticleLoader("data/SentiCoref_1.0")

    # dict {(art_name, entity): [Words]}
    #   where [Words] are sentences with this entity.
    data, data_sentiments = neural_load_articles(article_loader)
    data_sentiments = neural_join_labels(data_sentiments)

    train, val, test = neural_train_val_test_split(data_sentiments)
    neural_describe_data(data_sentiments, train, val, test)

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

#####################################################


    bert_embeddings_model = BertEmbeddingsSentiCoref.from_pretrained(BERT_MODEL,
                            embedding_dimensions=EMBEDDING_DIM,
                            cnn_filters=CNN_FILTERS,
                            dnn_units=DNN_UNITS,
                            model_output_classes=OUTPUT_CLASSES,
                            dropout_rate=DROPOUT_RATE,
                            from_pt = True)

    
    if OUTPUT_CLASSES == 2:
        bert_embeddings_model.compile(loss="binary_crossentropy",
                        optimizer="adam",
                        metrics=["accuracy"])
    else:
        bert_embeddings_model.compile(loss="sparse_categorical_crossentropy",
                        optimizer="adam",
                        metrics=["sparse_categorical_accuracy"])

    bert_embeddings_model.fit(train_ds, epochs=NB_EPOCHS, validation_data=val_ds)
   
    #########################################

    Y_predicted = np.argmax(bert_embeddings_model.predict(test_ds), axis=1)
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
    cmatdisp.plot(cmap=plt.cm.Blues, ax=plt.gca())
    plt.savefig('report/figures/confmat_BertEmbeddingsSentiCoref.pdf', bbox_inches='tight')

    print(textwrap.dedent(f"""
        EMBEDDING_DIM = {EMBEDDING_DIM}
        CNN_FILTERS = {CNN_FILTERS}
        DNN_UNITS = {DNN_UNITS}
        OUTPUT_CLASSES = {OUTPUT_CLASSES}
        DROPOUT_RATE = {DROPOUT_RATE}
        NB_EPOCHS = {NB_EPOCHS}
        BATCH_SIZE = {BATCH_SIZE}
    """))

    