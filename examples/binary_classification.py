import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import *
from sklearn.metrics import roc_auc_score

if __name__ == '__main__':
    df = pd.read_csv("data/criteo_sample.txt", sep=",")

    sparse_feature_names = ["C" + str(i) for i in range(1, 27)]
    dense_feature_names = ["I" + str(i) for i in range(1, 14)]
    df[sparse_feature_names] = df[sparse_feature_names].fillna("-1")
    df[dense_feature_names] = df[dense_feature_names].fillna(0)

    for feat in sparse_feature_names:
        lbe = LabelEncoder()
        df[feat] = lbe.fit_transform(df[feat])

    mms = MinMaxScaler(feature_range=(0, 1))
    df[dense_feature_names] = mms.fit_transform(df[dense_feature_names])

    sparse_feature_columns = [SparseFeat(feat, df[feat].nunique(), embedding_dim=4) for feat in sparse_feature_names]
    dense_feature_columns = [DenseFeat(feat, 1) for feat in dense_feature_names]

    linear_feature_columns = sparse_feature_columns + dense_feature_columns
    dnn_feature_columns = sparse_feature_columns + dense_feature_columns
    model = DeepFM(linear_feature_columns, dnn_feature_columns, task="binary", device="cpu", l2_reg_linear=1e-5)
    train, test = train_test_split(df, test_size=0.2)
    feature_names = get_feature_names(dnn_feature_columns)
    train_input = {name: train[name] for name in feature_names}
    test_input = {name: test[name] for name in feature_names}
    model.compile(optimizer="adagrad", loss="binary_crossentropy", metrics=["binary_crossentropy", "auc"])

    model.fit(train_input, train["label"].values, batch_size=64, epochs=10)
    pred = model.predict(test_input, batch_size=32)
    score=roc_auc_score(test["label"].values,pred)
    print("accuracy={}".format(score))

