from collections import Counter
from torch import nn
import torch


def tokenize(d1, d2, d3) -> list:
    return [d1, d2, d3]


if __name__ == '__main__':
    args = {
        "d1": 1,
        "d2": 2,
        "d3": 3
    }

    print(tokenize(**args))
