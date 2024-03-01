from collections import Counter
from torch import nn
import torch


def tokenize(corpus: list) -> list:
    return [sen.split() for sen in corpus]


if __name__ == '__main__':
    corpus = ["he is an old worker", "english is a useful tool", "the cinema is far away"]
    word_list = tokenize(corpus)

    sorted_dict = sorted(Counter(sum(word_list, [])).items(), key=lambda x: x[1], reverse=True)

    word2idx = {
        word: idx
        for idx, (word, freq) in enumerate(sorted_dict)
    }
    token = [[word2idx[w] for w in sen] for sen in word_list]

    emb = nn.Embedding(len(word2idx), 3)
    word_tensor = [torch.tensor(w) for w in word2idx.values()]
    # print(word_tensor)
    for word, idx in word2idx.items():
        word_tensor = torch.tensor(idx)
        print(word, emb(word_tensor).detach().numpy())
    # print(emb(word_tensor[0]))
    # print(emb(word) for word in word_tensor)
