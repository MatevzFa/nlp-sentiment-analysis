import random
from pprint import pprint

from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertModel, TFBertPreTrainedModel, TFBertMainLayer
from tensorflow.keras import layers

from _collections import defaultdict
from nlp_code.articles import ArticleLoader

BERT_MODEL = "models/slo-hr-en-bert-pytorch"


EMBEDDING_DIM = 512
CNN_FILTERS = 100
DNN_UNITS = 128
OUTPUT_CLASSES = 3
DROPOUT_RATE = 0.2
NB_EPOCHS = 3
BATCH_SIZE = 64


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
                 cnn_filters=50,
                 dnn_units=512,
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

if __name__ == "__main__":

    bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True)
    model = TFBertModel.from_pretrained(BERT_MODEL, from_pt=True)

    article_loader = ArticleLoader("data/SentiCoref_1.0")

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

bert_embeddings_model.fit(train_ds, epochs=NB_EPOCHS)

results_predicted = [1 if x>=0.5 else 0 for x in bert_embeddings_model.predict(test_ds) ]
results_true = np.array(y_test)


from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

print(f"F1 score: {f1_score(results_true, results_predicted)}")
print(f"Accuracy score: {accuracy_score(results_true, results_predicted)}")