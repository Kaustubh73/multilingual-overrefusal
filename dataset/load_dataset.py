"""Load the released dataset from the Hugging Face Hub.

    python dataset/load_dataset.py [revision]   # revision: tag or branch, default main
"""
import sys

from datasets import load_dataset

REPO = "kaustubhg73/multilingual-multitask-refusal"


def load(revision=None):
    return load_dataset(REPO, revision=revision)


if __name__ == "__main__":
    print(load(*sys.argv[1:2]))
