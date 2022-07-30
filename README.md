# qppBERT-PL

## qppBERT-PL is an end-to-end neural cross-encoder-based approach - trained pointwise on individual queries, but listwise over the top ranked documents (split into chunks).

<p align="center">
  <img src="architecture.png" width="400" height="300">
</p>

## Installation
qppBERT-PL requires Python 3.7 and JAVA 11+ and uses a number of libraries enlisted in `requirements.txt`.

We recommend creating a conda environment using:

```
conda env create -f conda_env.yml
conda activate qpp-bert
```
## Overview

Using this QPP model on a dataset typically involves the following steps.

`Step 0:` Preprocess your collection. At its simplest, qppBERT-PL works with tab-separated (TSV) files: a file (e.g., collection.tsv) will contain all passages and another (e.g., queries.tsv) will contain a set of queries. Besides, it has the provision of loading collection/ queries/ qrels from `ir-datasets` too.

`Step 2:` Index your collection to permit fast retrieval. This step encodes all passages into matrices, stores them on disk, and builds data structures for efficient search.

`Step 3:` Search the collection with your queries. Given your model and index, you can issue queries over the collection to retrieve the top-k passages for each query.

Below, we illustrate these steps via an example run on the MS MARCO Passage collection and TREC-DL-2021 topic set.
