# Introduction

This directory consists of the code for BERT-initialized Transformer and LSTM baselines used in the paper. 


# System Outputs

The outputs of BERT-initialized Transformer and other baselines are shared in `system_output` folder. The code for all the evaluation metrics is also available in the folder.


# Requirements and Installation

The code is based on [fairseq](https://github.com/pytorch/fairseq) toolkit. It requires PyTorch version >= 1.0.0 and 
Python version >= 3.5. For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)

If you do not have PyTorch, please follow the instructions here to install: https://github.com/pytorch/pytorch#installation.

And then install fairseq from the source code shared in the repository.
```
cd simplification
pip install --editable .
```

Please note that the shared version of fairseq is different from the latest version and BERT-initalized Transformer 
is implemented specifically for this version.


# Pre-trained models


**BERT-Initialized Transformer**

1. You need to preprocess the data using the ``preprocess.sh`` script. First, the script performs BERT Tokenization and then creates a binarized fairseq dataset.

```
sh preprocess.sh <raw data directory> <tokenized data directory>  <binarized data directory>

# The script assumes that the raw data has the following format
# <directory>/
#   |-- train.src
#   |-- train.dst
#   |-- test.src
#   |-- test.dst
#   |-- valid.src
#   |-- valid.dst
# .src files contain complex sentences and .dst files contain simple sentences.
```

2. Download the Transformer checkpoint for [newsela](http://web.cse.ohio-state.edu/~maddela.4/acl2020/checkpoint_newsela_auto.pt) or [wiki](http://web.cse.ohio-state.edu/~maddela.4/acl2020/checkpoint_wiki_auto.pt). You can perform generation using the following command.
 
```
sh generate.sh <binarized data directory> <checkpoint> <output file name> <GPU device id> <split>

<checkpoint> refers to the path of the checkpoint
<split> takes one of the following values: train, valid, test
```


# Training 


**BERT-initalized Transformer**

1. Follow the step 1 described in the above section for Transformer.


2. Download the BERT-base checkpoint from [here](https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip) 
and unzip the folder.  Then, train using the following command:

```
CUDA_VISIBLE_DEVICES=<GPU1,GPU2...> python3  train.py <binarized data path from previous step> --save-dir <checkpoint directory>  \
    --lr 0.0001 --optimizer adam  -a bert_rand --max-update 200000 --user-dir my_model --batch-size 32  \
    --lr-scheduler inverse_sqrt --warmup-updates 40000 --max-source-positions 512 --max-target-positions 512 \
    --bert_path uncased_L-12_H-768_A-12/bert_model.ckpt
```

All the model parameters are specified in ``my_model/__init__.py`` file.

3. Follow the step 2 described in the previous section.  You can generate using the best checkpoint according to the cross-entropy loss, i.e. ``checkpoint_best.pt`` in the specified checkpoint directory. Alternatively, you can also choose the best checkpoint according to the SARI score on the validation dataset.


**LSTM**

1. You need to preprocess the data using the ``preprocess_lstm.sh`` script. First, the script anonymizes entities  and then creates a binarized fairseq dataset.

```
sh preprocess_lstm.sh <raw data directory> <anonymized data directory>  <binarized data directory>

# The script assumes that the raw data has the following format
# <directory>/
#   |-- train.src
#   |-- train.dst
#   |-- test.src
#   |-- test.dst
#   |-- valid.src
#   |-- valid.dst
# .src files contain complex sentences and .dst files contain simple sentences.
```

2. Train using the following command


```
CUDA_VISIBLE_DEVICES=<GPU1,GPU2...> python3 train.py  <binarized data directory>  -a encdeca  \
    --max-epoch 30 --lr 0.001 --optimizer adam --save-dir  <checkpoint folder path> \
    --user-dir my_model --batch-size 96 \
    --clip-norm 5 --seed 13
    
```

3. You can perform generation using the following command. You can generate using the best checkpoint according to the cross-entropy loss, i.e. ``checkpoint_best.pt`` in the specified checkpoint directory.
 
```
sh generate_lstm.sh <binarized data directory> <checkpoint> <output file name> <GPU device id> <split>

<checkpoint> refers to the path of the checkpoint
<split> takes one of the following values: train, valid, test
```


# Citation

If you use any of these resources, please cite fairseq and our paper:

```bibtex
@inproceedings{jiang2020neural,
  title={Neural CRF Model for Sentence Alignment in Text Simplification},
  author={Jiang, Chao and Maddela, Mounica and Lan, Wuwei and Zhong, Yang and Xu, Wei},
  booktitle={Proceedings of the Association for Computational Linguistics (ACL)},
  year={2020}
}
```


```bibtex
@inproceedings{ott2019fairseq,
  title = {fairseq: A Fast, Extensible Toolkit for Sequence Modeling},
  author = {Myle Ott and Sergey Edunov and Alexei Baevski and Angela Fan and Sam Gross and Nathan Ng and David Grangier and Michael Auli},
  booktitle = {Proceedings of NAACL-HLT 2019: Demonstrations},
  year = {2019},
}
```
