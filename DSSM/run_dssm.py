import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import random
import torch
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from inputs import SparseFeat, VarLenSparseFeat


def gen_data_set(data, neg_sample):
    # (user_id,hist_id,movie_id,label,len(looked),rating)
    data.sort_values(by=['timestamp'], inplace=True)
    item_ids = data['movie_id'].unique()
    train = []
    test = []
    for user, item in tqdm(data.groupby('user_id')):
        looked = item['movie_id'].tolist()
        rating = item['rating'].tolist()

        if neg_sample > 0:
            candidate_list = list(set(item_ids) - set(looked))
            neg_list = np.random.choice(candidate_list, len(looked) * neg_sample, replace=True)
        for i in range(1, len(looked)):
            hist = looked[:i]
            if i != len(looked) - 1:
                train.append((user, hist[::-1], looked[i], 1, len(hist[::-1]), rating[i]))
                for neg in range(neg_sample):
                    train.append((user, hist[::-1], neg_list[i * neg_sample + neg], 0, len(hist[:-1])))
            else:
                test.append((user, hist[::-1], looked[i], 1, len(hist[::-1]), rating[i]))

    random.shuffle(train)
    random.shuffle(test)

    return train, test


def gen_model_input(data_set, user_profile, max_seq_len):
    train_uid = np.array([line[0] for line in data_set])
    train_seq = [line[1] for line in data_set]
    train_iid = np.array([line[2] for line in data_set])
    train_label = np.array([line[3] for line in data_set])
    train_seq_len = np.array([line[4] for line in data_set])

    train_seq_pad = pad_sequences(train_seq, maxlen=max_seq_len, padding='post', truncating='post', value=0)
    train_model_input = {"user_id": train_uid, "movie_id": train_iid, "hist_movie_id": train_seq_pad,
                         "hist_len": train_seq_len}

    keys = ['age', 'gender', 'occupation', 'zip']
    for k in keys:
        train_model_input[k] = user_profile.loc[train_uid][k].values

    return train_model_input, train_label


if __name__ == "__main__":
    # data_path = r"D:\desktop\project\job\algorithm\testHub\recc_system\data\movielens_sample.txt"
    user = pd.read_csv(r"D:\desktop\project\job\algorithm\testHub\recc_system\data\ml-100k\u.user", sep="|",
                       header=None)
    user.columns = ['user_id', 'age', 'gender', 'occupation', 'zip']
    hist = pd.read_csv(r"D:\desktop\project\job\algorithm\testHub\recc_system\data\ml-100k\u.data",
                       delim_whitespace=True,
                       header=None)
    hist.columns = ['user_id', 'movie_id', 'rating', 'timestamp']
    tmp = pd.merge(hist, user, how='right', on=['user_id'])
    movie = pd.read_csv(r"D:\desktop\project\job\algorithm\testHub\recc_system\data\ml-100k\movies.csv", sep=',')
    movie.columns = ['movie_id', 'title', 'genres']
    data = pd.merge(tmp, movie, how='left', on=['movie_id'])

    features = ['user_id', 'movie_id', 'gender', 'age', 'occupation', 'zip']
    feature_max_idx = {}
    for feat in features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat]) + 1
        feature_max_idx[feat] = data[feat].max() + 1
    user_profile = data[['user_id', 'gender', 'age', 'occupation', 'zip']].drop_duplicates(['user_id'])
    item_profile = data[['movie_id']].drop_duplicates(['movie_id'])
    user_profile.set_index('user_id', inplace=True)
    user_item_list = data.groupby('user_id')['movie_id'].apply(list)
    train_set, test_set = gen_data_set(data, neg_sample=3)
    max_len = 100
    train_input, train_labels = gen_model_input(train_set, user_profile, max_len)

    embed_dim = 8
    sparse_features = ['user_id', 'gender', 'age', 'occupation', 'zip']
    sparse_columns = [SparseFeat(feat, feature_max_idx[feat], embedding_dim=embed_dim) for feat in sparse_features]
    varlen_columns = [VarLenSparseFeat(
        SparseFeat('hist_movie_id', feature_max_idx['movie_id'], embedding_dim=embed_dim, embedding_name='movie_id'),
        max_len, 'mean', 'hist_len')]
    user_feature_columns = sparse_columns + varlen_columns

    print(user_feature_columns)
