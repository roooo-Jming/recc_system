import torch
from pytorch_lightning import LightningModule
import torch.nn.functional as F
import inputs
import torch.nn as nn
from inputs import SparseFeat, DenseFeat, VarLenSparseFeat


class Linear(nn.Module):
    def __init__(self, feature_columns):
        super(Linear, self).__init__()
        self.feature_columns = feature_columns
        self.sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns) if len(feature_columns) else [])
        self.dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns) if len(feature_columns) else [])
        self.varlen_feature_columns=list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns) if len(feature_columns) else [])


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
        self.feature_index = inputs.build_input_features(self.linear_feature_columns)
        self.embedding_matrix = inputs.create_embedding_matrix(self.feature_index, init_std=self.config.get("std"),
                                                               sparse=False)
