FROM python:3.7-buster

RUN pip install --no-cache-dir gdown
RUN apt-get update && apt-get install tree zip


RUN mkdir /app
WORKDIR /app

# Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Prepare directories
RUN mkdir -p data/cache data/features data/JOB_1.0 data/SentiCoref_1.0 data/SentiNews_1.0
RUN mkdir models
RUN mkdir -p report/figures


# 
# models
# 
WORKDIR /app/models
# BERT
RUN gdown 'https://drive.google.com/uc?id=102DbPO8lrQn2gBsEYX93tFQxteWYy-0d' && \
    unzip slo-hr-en-bert-pytorch.zip && \
    rm slo-hr-en-bert-pytorch.zip
# lemmatizer
RUN curl https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1286/ssj500k%2bSloleks_lemmatizer.pt --output ssj500k_lemmatizer.pt
# BertEmbeddingsSentiCoref
RUN gdown 'https://drive.google.com/uc?id=1lUXpav0wHxH7m7J_Xae-87kgINxypx0C' && \
    unzip pretrained_bert_embeddings_balanced.zip && \
    rm pretrained_bert_embeddings_balanced.zip && \
    mv model-balanced pretrained_bert_embeddings_balanced


# 
# data
# 
WORKDIR /app/data
# SentiNews
RUN cd SentiNews_1.0 && \
    curl --remote-name-all https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1110{/SentiNews_document-level.txt,/SentiNews_paragraph-level.txt,/SentiNews_sentence-level.txt} && \
    cd ..
# SentiCoref
RUN cd SentiCoref_1.0 && \
    curl --remote-name-all https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1285{/SentiCoref_1.0.zip} && \
    unzip SentiCoref_1.0.zip && \
    rm SentiCoref_1.0.zip && \
    cd ..
# JOB
RUN cd JOB_1.0 && \
    curl --remote-name-all https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1112{/Slovene_sentiment_lexicon_JOB.txt} && \
    cd ..


WORKDIR /app
COPY . .
RUN tree
