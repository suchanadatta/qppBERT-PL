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
