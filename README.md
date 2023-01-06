# BundleMage
This project is a PyTorch implementation of BundleMage (PLOS ONE 2023).

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

## Citation
```
@article{10.1371/journal.pone.0280630,
    author = {Hyunsik Jeon and 
              Jun{-}Gi Jang and 
              Taehun Kim and
              U Kang},
    journal = {PLOS ONE},
    publisher = {Public Library of Science},
    title = {Accurate Bundle Matching and Generation via Multitask Learning with Partially Shared Parameters},
    year = {2023},
    month = {01}
}
```
