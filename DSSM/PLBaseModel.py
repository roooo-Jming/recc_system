from pytorch_lightning import LightningModule

class PLBaseModel(LightningModule):
    def __init__(self):
        super(PLBaseModel, self).__init__()
