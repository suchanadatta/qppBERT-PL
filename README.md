# qppBERT-PL

## qppBERT-PL is an end-to-end neural cross-encoder-based approach - trained pointwise on individual queries, but listwise over the top ranked documents (split into chunks).

<p align="center">
  <img src="architecture.png" width="400" height="300">
</p>

## Installation
qppBERT-PL requires Python 3.7 and JAVA 11+ and uses a number of libraries listed in `requirements.txt`.

We recommend creating a conda environment using:

```
conda env create -f conda_env.yml
conda activate qpp-bert
```
## Overview

Using this QPP model on a dataset typically involves the following steps.

`Step 0:` Preprocess your collection. At its simplest, qppBERT-PL works with tab-separated (TSV) files: a file (e.g., collection.tsv) will contain all passages and another (e.g., queries.tsv) will contain a set of queries. Besides, it has the provision of loading collection/ queries/ qrels from [ir-datasets](https://ir-datasets.com/) too. 

`Step 1:` Index your collection to permit fast retrieval. This step encodes all passages into matrices, stores them on disk, and builds data structures for efficient search. We used [PyTerrier-pisa](https://github.com/terrier-org/pyterrier.git) specifically for this work. 

`Step 2:` Search the collection with your queries. Given your model and index, you can issue queries over the collection to retrieve the top-k passages for each query.

Below, we illustrate these steps via an example run on the MS MARCO Passage collection and TREC-DL-2021 topic set.

## Use Model Checkpoint and Evaluate

We provide a [model checkpoint](https://drive.google.com/file/d/1XHcLgpkfWd3XnaQ5YKTkFdCqrU8-wGeO/view?usp=sharing) trained on TREC-DL-2019 + TREC-DL-2020 + a random sample of MS MARCO train set, find [here](https://microsoft.github.io/msmarco/). Run the command below to evaluate the model performance on [TREC-DL-2021](https://microsoft.github.io/msmarco/TREC-Deep-Learning-2021) topic set using the given trained model.

### Data

To do the testing, you need the following:

1. [Index](https://drive.google.com/file/d/1mxP4OMbXXk9BLaePF5pkc3lx4gb4jlNZ/view?usp=sharing)
2. [Collection](https://drive.google.com/file/d/1mxP4OMbXXk9BLaePF5pkc3lx4gb4jlNZ/view?usp=sharing) - MS MARCO Passage (v2) if not loading from [ir-datasets](https://ir-datasets.com/). D/w the [pickle](https://drive.google.com/file/d/12dcXWHiJy0adTuhHWw72DobjZUQDWP0j/view?usp=share_link) dump from here.
3. [Trained model](https://drive.google.com/file/d/1mxP4OMbXXk9BLaePF5pkc3lx4gb4jlNZ/view?usp=sharing)
4. [TREC-DL-2021 topics](https://github.com/suchanadatta/qppBERT-PL/tree/master/data/trec-dl-2021)

### Evaluation

Evaluate the performance of the model on test dataset using pre-trained model by running the following command:

```
python ./qpp_model/eval.py \
-- index <pah of the pisa index> \
-- dataset <'irdataset' for loading from ir-datasets> \
-- collection <path of the .pickle file only if not using ir-datasets> \
-- query <path of the queries.tsv file (test queries), again if not loading from ir-datasets> \
-- checkpoint <path of the pre-trained model> \
-- batch-size <default is set to 4> \
-- chunk-per-query <how many fixed-sized chunks to be taken into account during testing> 
```

### Output

Model's outcome on TREC-DL-2021 topic set can be found [here](https://github.com/suchanadatta/qppBERT-PL). Note that predictions are made on chunk size 4 which can be varied as per the requirement. `eval.py` produces two main output files: 1. `dl21.reldocs` and 2. `dl21.pred.ap` - to be interpreted as follows-

> dl21.reldocs ('qid' \t 'num_rels@100')

- It is a 2-column .tsv file that records the total no. of relevant documents predicted by the model in top 100 BM25 ranked list for each query.

> dl21.pred.ap ('qid' \t 'pred_ap')

- This file contains the predicted AP of each query in the test topic set. AP is measured as a weighted average of the outputs of the network as per the Equation 3 in the [paper](https://github.com/suchanadatta/qppBERT-PL/blob/master/sp1544.pdf).

Besides, eval.py generates two more intermmediate files - 1. `dl21.out` and 2. `dl21.pred`. These files help to analyze the estimated performance of each query at different cut-offs (through chunks).

> dl21.out ('qid' \t 'doc_id' \t 'rank_in_chunk' \t 'chunk_id' \t 'num_rel_docs')

> dl21.pred ('qid' \t 'chunk_id' \t 'num_rels_in_chunk')


## Training

If you want to train a model on your own, run the command below - 

```
python ./qpp_model/train.py \
-- index <pah of the pisa index> \
-- dataset <'irdataset' for loading from ir-datasets> \
-- collection <path of the .pickle file only if not using ir-datasets> \
-- query <path of the queries.tsv file (test queries), if not loading from ir-datasets> \
-- batch-size <default is set to 4> \
-- chunk-per-query <how many fixed-sized chunks to be taken into account during testing> 
```

Once you have the model trained, evaluate it with your desired test set following the steps in 'Use Model Checkpoint and Evaluate`.

## Cite

```
@inproceedings{DBLP:conf/sigir/DattaMGG22,
  author    = {Suchana Datta and
               Sean MacAvaney and
               Debasis Ganguly and
               Derek Greene},
  title     = {A 'Pointwise-Query, Listwise-Document' based Query Performance Prediction
               Approach},
  booktitle = {{SIGIR}},
  pages     = {2148--2153},
  publisher = {{ACM}},
  year      = {2022}
}
```
