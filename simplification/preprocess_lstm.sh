#!/usr/bin/env bash

DATA_PATH=$1
ANON_PATH=$2
DATA_BIN=$3

python3 preprocess/anonymize_spacy.py --dict $ANON_PATH/valid_ne.txt     \
        --src $DATA_PATH/valid.src \
        --dst $DATA_PATH/valid.dst \
        --out_src $ANON_PATH/valid.aner.src     \
        --out_dst $ANON_PATH/valid.aner.dst


python3 preprocess/anonymize_spacy.py --dict $ANON_PATH/test_ne.txt     \
        --src $DATA_PATH/test.src \
        --dst $DATA_PATH/test.dst \
        --out_src $ANON_PATH/test.aner.src     \
        --out_dst $ANON_PATH/test.aner.dst


python3 preprocess/anonymize_spacy.py --dict $ANON_PATH/train_ne.txt     \
        --src $DATA_PATH/train.src \
        --dst $DATA_PATH/train.dst \
        --out_src $ANON_PATH/train.aner.src     \
        --out_dst $ANON_PATH/train.aner.dst


/usr/bin/python3 preprocess.py --workers 5 --source-lang src --target-lang dst \
  --trainpref $ANON_PATH/train.aner --validpref $ANON_PATH/valid.aner --testpref $ANON_PATH/test.aner \
  --destdir  $DATA_BIN --padding-factor 1 --thresholdtgt 4 --thresholdsrc 4
