import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import random
import torch
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from inputs import SparseFeat, VarLenSparseFeat


def process_data(data_path, neg_sample=0, max_len=50):
    df = pd.read_csv(data_path, sep=",")
    features_name = ['user_id', 'movie_id', 'gender', 'age', 'occupation', 'zip']

    feature_max_index_dict = {}
    for feat in features_name:
        lbe = LabelEncoder()
        df[feat] = lbe.fit_transform(df[feat]) + 1
        feature_max_index_dict[feat] = df[feat].max() + 1

    user_profile = df[['user_id', 'gender', 'age', 'occupation', 'zip']].drop_duplicates("user_id")
    item_profile = df[["movie_id"]].drop_duplicates("movie_id")

    df.sort_values("timestamp", inplace=True)
    movie_list = df["movie_id"].unique()
    train_list = []
    test_list = []

    for user_id, hist_df in tqdm(df.groupby("user_id")):
        pos_list = hist_df["movie_id"].to_list()
        rating_list = hist_df["rating"].to_list()
        # train,test的数据格式：user_id, hist_movid_id, pos_movie_id, label, hist_len, pos_rating

        neg_list = []

        if neg_sample > 0:
            candidate_list = list(set(movie_list) - set(pos_list))
            neg_list.append(np.random.choice(candidate_list, size=len(pos_list) * neg_sample, replace=True))

        min_seq_len = 1
        for i in range(min_seq_len, len(pos_list)):
            hist = pos_list[:i]
            if i != len(pos_list) - min_seq_len:
                train_list.append([user_id, hist[::-1], pos_list[i], 1, len(hist[::-1]), rating_list[i]])
                for neg_idx in range(neg_sample):
                    train_list.append(
                        [user_id, hist[::-1], neg_list[i * neg_sample + neg_idx], 0, len(hist[::-1])])
            else:
                test_list.append([user_id, hist[::-1], pos_list[i], 1, len(hist[::-1]), rating_list[i]])

    random.shuffle(train_list)
    random.shuffle(test_list)

    train_dict = {}
    test_dict = {}

    train_dict["user_id"] = np.array(line[0] for line in train_list)
    train_hist_movie_id = [line[1] for line in train_list]
    train_dict["hist_movie_id"] = pad_sequences(train_hist_movie_id, padding="post", truncating="post", maxlen=max_len)
    train_dict["movie_id"] = np.array(line[2] for line in train_list)
    train_dict["label"] = np.array(line[3] for line in train_list)
    train_dict["hist_len"] = np.array(line[4] for line in train_list)
    train_dict["rating"] = np.array(line[5] for line in train_list)

    for key in ['gender', 'age', 'occupation', 'zip']:
        tmp = []
        for user in train_dict["user_id"]:
            tmp.append(user_profile[user_profile["user_id"] == user][key])
        train_dict[key] = np.array(tmp)

    test_dict["user_id"] = np.array(line[0] for line in test_list)
    test_hist_movie_id = [line[1] for line in test_list]
    test_dict["hist_movie_id"] = pad_sequences(test_hist_movie_id, padding="post", truncating="post", maxlen=max_len)
    test_dict["movie_id"] = np.array(line[2] for line in test_list)
    test_dict["label"] = np.array(line[3] for line in test_list)
    test_dict["hist_len"] = np.array(line[4] for line in test_list)
    test_dict["rating"] = np.array(line[5] for line in test_list)

    for key in ['gender', 'age', 'occupation', 'zip']:
        tmp = []
        for user in train_dict["user_id"]:
            tmp.append(user_profile[user_profile["user_id"] == user][key])
        train_dict[key] = np.array(tmp)

    return feature_max_index_dict, train_dict, test_dict


if __name__ == "__main__":
    data_path = r"D:\desktop\project\job\algorithm\testHub\recc_system\data\movielens_sample.txt"
    feature_max_index_dict, train_dict, test_dict = process_data(data_path)
    train_dict.pop("label")
    embedding_dim = 8
    SEQ_LEN = 50
    neg_sample = 3
    user_feature_columns = [SparseFeat("user_id", feature_max_index_dict["user_id"], embedding_dim),
                            SparseFeat("gender", feature_max_index_dict["gender"], embedding_dim),
                            SparseFeat("age", feature_max_index_dict["age"], embedding_dim),
                            SparseFeat("occupation", feature_max_index_dict["occupation"], embedding_dim),
                            SparseFeat("zip", feature_max_index_dict["zip"], embedding_dim),
                            VarLenSparseFeat(
                                SparseFeat("hist_movie_id", feature_max_index_dict["movie_id"], embedding_dim,
                                           embedding_name="movie_id"), SEQ_LEN, "mean", "hist_len")]
    item_feature_columns = [SparseFeat("movie_id", feature_max_index_dict["movie_id"], embedding_dim)]
