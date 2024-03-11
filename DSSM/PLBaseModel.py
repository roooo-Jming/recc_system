import torch
from pytorch_lightning import LightningModule
import torch.nn.functional as F


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

