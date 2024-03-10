import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import random
from item2vec import Item2Vec
from torch.utils.data import Dataset, DataLoader
from collections import Counter


def choose_with_prob(discard_prob: float) -> bool:
    p = np.random.uniform(low=0.0, high=1.0)
    if p < discard_prob:
        return False
    else:
        return True


def generate_sample(discard_prob: dict, train_seq: list, context_window: int, vocabulary_size: int,
                    discard=False) -> list:
    samples = []
    for seq in train_seq:
        if discard:
            seq = [w for w in seq if choose_with_prob(discard_prob[w])]
        for i in range(len(seq)):
            target = seq[i]
            context_list = []
            j = i - context_window
            while j < i + context_window and j < len(seq):
                if j >= 0 and j != i:
                    samples.append([(target, seq[j]), 1])
                    context_list.append(seq[j])
                j += 1

            for _ in range(len(context_list)):
                neg_idx = random.randrange(0, vocabulary_size)
                while neg_idx in context_list:
                    neg_idx = random.randrange(0, vocabulary_size)
                samples.append([(target, neg_idx), 0])
    return samples


class Item2VecDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        label = self.data[index][1]
        xi = self.data[index][0][0]
        xj = self.data[index][0][1]
        label = torch.tensor(label, dtype=torch.float32)
        xi = torch.tensor(xi, dtype=torch.long)
        xj = torch.tensor(xj, dtype=torch.long)
        return xi, xj, label

    def __len__(self):
        return len(self.data)


def run_model():
    df = pd.read_csv("data/ml-100k/u.data", sep=" ", header=None, names=['user', 'item', 'rating', 'timestamp'])
    df["user"] -= 1
    df["item"] -= 1
    args = {
        "context_window": 2,
        "vocabulary_size": df["item"].unique()
    }
    train, test = train_test_split(df, test_size=0.2)
    train_seq = train.groupby("user")["item"].agg(list)
    item_frequency = train["item"].value_counts()
    discard_prob = 1 - np.sqrt(args["rho"] / item_frequency)
    sample = generate_sample(discard_prob, train_seq, args["context_window"], args["vocabulary_size"])
    train_set = Item2VecDataset(sample)
    train_generator = DataLoader(train_set, batch_size=args["batch_size"], shuffle=True)
    params = {
        "item_len": df["item"].unique(),
        "embed_dim": 100,
        "epoch": 20,
        "learning_rate": 0.001,
        "early_stop": True
    }
    model = Item2Vec(**params)
    model.fit(train_generator)


def tokenize(corpus: list) -> list:
    return [sen.split() for sen in corpus]


def embedding_utils():
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
