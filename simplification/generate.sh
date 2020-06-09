#!/usr/bin/env bash



# Script to generate simplified sentences and post-process the output by removing BPE.

# sh generate.sh <binarized data directory> <checkpoint> <output file name> <GPU device id> <split>


DATA_BIN=$1
CP=$2
OUTPUT=$3
SPLIT=$5

echo "Generating using checkpoint ${CP}"
CUDA_VISIBLE_DEVICES=$4 python3 generate.py $DATA_BIN --path $CP \
    --batch-size 32  --beam 1 --nbest 1 --user-dir my_model/ --print-alignment --gen-subset $SPLIT > $OUTPUT.aner

python3 postprocess/bpe.py  --out_anon $OUTPUT.aner --denon $OUTPUT --ignore_lines 5 --wp 1

rm $OUTPUT.aner
