import torch.nn as nn


class EmbeddingModule(nn.Module):
    def __init__(self, datatypes, use_sen_et):
        super(EmbeddingModule, self).__init__()
        self.embs = nn.ModuleList()
        for datatype in datatypes:
            if datatype['type'] == 'SparseEncoder' or datatype['type'] == 'BucketSparseEncoder':
                self.embs.append(nn.Embedding(datatype['length'], datatype['emb_dim']))
