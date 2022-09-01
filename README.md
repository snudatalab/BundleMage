# BundleMage
This project is a PyTorch implementation of BundleMage (submitted to BigData 2022).

## Prerequisties
Our implementation is based on Python 3.6 and Pytorch 1.8.1. Please see the full list of packages required to our codes in `requirements.txt`.

## Datasets
We use 3 datasets in our work: Youshu, Netease, and Steam.
We include the preprocessed datasets in the repository: `data/{data_name}`.

## Running the code
You can run the code by `python main.py` with arguments `--data` and `--task`.
Set `--data` argument as one among 'youshu', 'netease', and 'steam'.
Set `--task` argument as 'mat' for bundle matching, or as 'gen' for bundle generation.
We provide `demo.sh`, which reproduces the experiments of our work for bundle matching and generation.
