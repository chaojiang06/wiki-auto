#!/usr/bin/env bash


# Script to preprocess the input data. First, the script performs BERT Tokenization and then creates a binarized fairseq dataset.

# sh preprocess.sh <raw data directory> <tokenized data directory>  <binarized data directory>
# The script assumes that the raw data has the following format
# <directory>/
#   |-- train.src
#   |-- train.dst
#   |-- test.src
#   |-- test.dst
#   |-- valid.src
#   |-- valid.dst
# .src files contain complex sentences and .dst files contain simple sentences.


RAW_DATA_PATH=$1  # raw data path
ANON_DATA_PATH=$2  # tokenized data path
DATA_BIN=$3 # binarized data path


# Tokenizes the input data

/usr/bin/python3 preprocess/anonymize_wordpiece.py --input $RAW_DATA_PATH/test.src --vocab preprocess/vocab.txt \
 --output  $ANON_DATA_PATH/test.tok.src
/usr/bin/python3 preprocess/anonymize_wordpiece.py --input $RAW_DATA_PATH/test.dst --vocab preprocess/vocab.txt \
 --output  $ANON_DATA_PATH/test.tok.dst


/usr/bin/python3 preprocess/anonymize_wordpiece.py --input $RAW_DATA_PATH/valid.src --vocab preprocess/vocab.txt \
 --output  $ANON_DATA_PATH/valid.tok.src
/usr/bin/python3 preprocess/anonymize_wordpiece.py --input $RAW_DATA_PATH/valid.dst --vocab preprocess/vocab.txt \
 --output  $ANON_DATA_PATH/valid.tok.dst


/usr/bin/python3 preprocess/anonymize_wordpiece.py --input $RAW_DATA_PATH/train.src --vocab preprocess/vocab.txt \
 --output  $ANON_DATA_PATH/train.tok.src
/usr/bin/python3 preprocess/anonymize_wordpiece.py --input $RAW_DATA_PATH/train.dst --vocab preprocess/vocab.txt \
 --output  $ANON_DATA_PATH/train.tok.dst


# Creates binarized fairseq dataset

/usr/bin/python3 preprocess.py --workers 5 --source-lang src --target-lang dst \
  --trainpref $ANON_DATA_PATH/train.tok --validpref $ANON_DATA_PATH/valid.tok --testpref $ANON_DATA_PATH/test.tok \
  --destdir  $DATA_BIN --padding-factor 1 --joined-dictionary --srcdict preprocess/vocab_count.txt

