# Neural CRF Model for Sentence Alignment in Text Simplification

This repository contains the code and resources from the following [paper](https://arxiv.org/abs/2005.02324)


## Repo Structure: 
1. ```aligner```: Code for neural CRF sentence aligner.

1. ```wiki-manual```: The Wiki-Manual dataset. The definitions of columns are: label, the index of simple sentence, the index of complex sentence, simple sentence, complex sentence.

1. ```wiki-auto```: The Wiki-Auto dataset. ```.src``` are the complex sentences, and ```.dst``` are the simple sentences.
1. ```annotation_tool```: The tool for in-house annotators to annotate the sentence alignment.


## Instructions: 
1. To request the Newsela-Manual and Newsela-Auto datasets, please first obtain access to the [Newsela
corpus](https://newsela.com/data/), then contact the authors.

1. Please use Python 3 to run the code.

1. We also have pre-processed Wikipedia data, alignments between complex and simple Wikipedia articles, and paragraph alignments between Wikipedia article pairs, please contact us if you want to use that data.

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

