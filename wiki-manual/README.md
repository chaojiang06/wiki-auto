## Overview
Here we provide the full version of `Wiki-manual` dataset for replicating the performance of sentence alignment. Each article pair contains `m*n` sentence pairs, where `m, n` are the number of sentences in `simple, complex` articles. We also provide the sentence level GLEU (Google-BLEU) score for each pair. 


The labels come from crowd-sourcing annotation, and dev / test sets have further been manually corrected by human annotator under binary labels `(aligned + partialAligned v.s. notAligned)`.

## Format
Each line is in the format of `<label>+"\t"+<simple-sent-index>"\t"+<complex-sent-index>+"\t"<simple-sent>+"\t"<complex-sent>+"\t"<GLEU-score>`.

`simple-sent-id` and `complex-sent-id` are both in the format of `articleIndex-level-paragraphIndex-sentIndex`. In the position of `level`, `0` means simple article and `1` means complex article. `paragraphIndex` and `sentIndex` starts from zero.

For example, `399_620677-0-3-7` means the 8th sentence in the 4th paragraph in a simple article. You can even visit the simple article by `https://simple.wikipedia.org/wiki?curid=620677`, and find the complex article through inter-language links at the left-side of the webpage. 

## How to find sentence splitting?

You can only keep the `aligned and partialAligned` sentence pairs and match sentence pairs by `complex-sent-index` (e.g., `118_118165-1-0-1`). If there are more than one simple sentence aligned to the same complex sentence, they together is a sentence splitting case. Basically, it is saying, two (or more) simple sentences are aligned to the same complex sentence. Below is an example from `train.tsv`.

`partialAligned	118_118165-0-0-1	118_118165-1-0-1	About 245 people live there.	About 245 people live there and it has 5.33 km2 (1317.072 acres) of land.	0.2`


`partialAligned	118_118165-0-0-2	118_118165-1-0-1	It covers 5.33 km2.	About 245 people live there and it has 5.33 km2 (1317.072 acres) of land.	0.04`


## How to use GLEU score?

We noticed that some partialAligned sentence pairs have high edit-distance. Moreover, human annotators have manually gone through dev and test sets, but not training set. So there could be a small portion of annotating error. 

Therefore, we provide sentence level GLEU score. If you want to use this dataset for text simplification, we recommend you to filter our sentence pairs with  `< 0.1 GLEU score`. You can aslo customize the filtering based on your need. 

