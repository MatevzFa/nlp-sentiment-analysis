FROM python:3.7-buster

RUN curl https://github.com/tanaikech/goodls/releases/download/v1.2.6/goodls_darwin_amd64 --output /usr/bin/goodls
RUN chmod +x /usr/bin/goodls


RUN mkdir /app
WORKDIR /app

COPY . .

RUN mkdir -p data/cache data/features data/JOB_1.0 data/SentiCoref_1.0 data/SentiNews_1.0
RUN mkdir models

WORKDIR /app/models
# BERT
RUN goodls -u https://drive.google.com/file/d/102DbPO8lrQn2gBsEYX93tFQxteWYy-0d/view
RUN unzip -d slo-hr-en-bert-pytorch slo-hr-en-bert-pytorch.zip
# lemmatizer
RUN curl --remote-name-all https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1286{/ssj500k%2bSloleks_lemmatizer.pt}

# Dependencies
RUN pip install -r requirements.txt

CMD ["tree" "/app"]