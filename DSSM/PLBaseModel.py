import torch
from pytorch_lightning import LightningModule
import torch.nn.functional as F
import inputs
import torch.nn as nn
from inputs import SparseFeat, DenseFeat, VarLenSparseFeat


class Linear(nn.Module):
    def __init__(self, feature_columns, feature_index, init_std=0.0001):
        super(Linear, self).__init__()
        self.feature_columns = feature_columns
        self.feature_index = feature_index
        self.init_std = init_std
        self.sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns) if len(feature_columns) else [])
        self.dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns) if len(feature_columns) else [])
        self.varlen_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns) if len(feature_columns) else [])
        self.embedding_dict = inputs.create_embedding_matrix(self.feature_columns, self.init_std, linear=True,
                                                             sparse=False)

        for tensor in self.embedding_dict.values:
            nn.init.normal_(tensor.weight, mean=0, std=self.init_std)

    def forward(self, X):
        sparse_embedding_list = [self.embedding_dict[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for feat in
            self.sparse_feature_columns]
        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1].long()] for feat in
                            self.dense_feature_columns]
        varlen_embedding_dict=inputs.varlen_embedding_lookup(X,self.varlen_feature_columns,self.embedding_dict,self.feature_index)



class PLBaseModel(LightningModule):
    def __init__(self, user_feature_columns,
                 item_feature_columns,
                 optimizer=None,
                 optimizer_args=None,
                 criterion=F.mse_loss,
                 criterion_args=None,
                 scheduler=None,
                 scheduler_args=None,
                 scaler=None,
                 config={},
                 **kwargs):
        super(PLBaseModel, self).__init__()
        self.user_feature_columns = user_feature_columns
        self.item_feature_columns = item_feature_columns
        self.config = config
        self.config.update(kwargs)
        self.mode = self.config.get("mode", "train")
        self.linear_feature_columns = self.user_feature_columns + self.item_feature_columns
        self.dnn_feature_columns = self.linear_feature_columns
        self.reg_loss = torch.zeros((1,))
        self.aux_loss = torch.zeros((1,))
        # build_input_features=OrderedDict: {feature_name: (start, start + dimension)}
        self.feature_index = inputs.build_input_features(self.linear_feature_columns)
        # ModuleDict{embedding_name:nn.Embedding}
        self.embedding_matrix = inputs.create_embedding_matrix(self.feature_index, init_std=self.config.get("std"),
                                                               sparse=False)
