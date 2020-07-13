# Neural CRF Model for Sentence Alignment in Text Simplification

This repository contains the code and resources from the following [paper](https://arxiv.org/abs/2005.02324)


## Repo Structure: 
1. ```aligner```: Code for neural CRF sentence aligner.

1. ```wiki-manual```: The Wiki-Manual dataset. The definitions of columns are: label, the index of simple sentence, the index of complex sentence, simple sentence, complex sentence.

1. ```wiki-auto```: The Wiki-Auto dataset. ```.src``` are the complex sentences, and ```.dst``` are the simple sentences.
1. ```annotation_tool```: The tool for in-house annotators to annotate the sentence alignment.
1. ```simplification```: Code for text simplification experiments.

## Checkpoints
1. We released the checkpoints of ```BERT``` model fine-tuned on Newsela-Manual and Wiki-Manual datasets. They are trained using the Hugging Face implementation of ```BERT_base``` architecture in the package ```pytorch-transformers==1.1.0```. [```BERT_newsela```](https://drive.google.com/file/d/1xL9KS8A-_g4dcOapW5Z3I-3g8GBqUQkP/view?usp=sharing) and [```BERT_wiki```](https://drive.google.com/file/d/1I43F4OMkCvTUMtTd9Ft3P0hGiQLcFjlT/view?usp=sharing).
1. If you want to align other monolingual parallel data, please try the fine-tuned BERT models. They should be able to achieve competitive performance. The performance boost of adding the neural CRF model is related to the structure of the articles. We have some experience in designing the paragraph alignment algorithm and using neural CRF model to align sentences, feel free to contact us if you want to have a discussion.
1. We also released the code for our neural CRF sentence alignment model, you can use it to train your own model.


## Instructions: 
1. To request the Newsela-Manual and Newsela-Auto datasets, please first obtain access to the [Newsela
corpus](https://newsela.com/data/), then contact the authors.

1. Please use Python 3 to run the code.

1. We also have pre-processed Wikipedia data, alignments between complex and simple Wikipedia articles, and original sentence and paragraph alignments between Wikipedia article pairs, please contact us if you want to use that data.

1. We also have the original sentence and paragraph alignments between the Newsela articles, please contact us if you want to use that data.

## Citation
Please cite if you use the above resources for your research
```
@inproceedings{jiang2020neural,
  title={Neural CRF Model for Sentence Alignment in Text Simplification},
  author={Jiang, Chao and Maddela, Mounica and Lan, Wuwei and Zhong, Yang and Xu, Wei},
  booktitle={Proceedings of the Association for Computational Linguistics (ACL)},
  year={2020}
}
```

