#!/usr/bin/env bash

DATA_BIN=$1
CP=$2
OUTPUT=$3
SPACY=$4


echo "Checkpoint ${CP}"

CUDA_VISIBLE_DEVICES=$5  python3 generate.py $DATA_BIN --path $CP --batch-size 512 \
  --beam 1 --nbest 1 --user-dir my_model/ --print-alignment > $OUTPUT.aner

python3 postproceess/spacy.py  --dict $SPACY/test_ne.txt \
    --out_anon $OUTPUT.aner  --denon $OUTPUT  --src_anon $SPACY/test.aner.src

rm $OUTPUT.aner
