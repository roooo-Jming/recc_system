from collections import namedtuple, OrderedDict
import torch.nn as nn

DEFAULT_GROUP_NAME = "default_group"


class SparseFeat(namedtuple("SparseFeat",
                            ["name", "vocabulary_size", "embedding_dim", "use_hash", "dtype", "embedding_name",
                             "group_name"])):
    __slots__ = ()

    def __new__(cls, name, vocabulary_size, embedding_dim=4, use_hash=False, dtype="int32", embedding_name=None,
                group_name=DEFAULT_GROUP_NAME):
        if embedding_name is None:
            embedding_name = name
        if embedding_dim == "auto":
            embedding_dim = 6 * int(pow(vocabulary_size, 0.25))
        if use_hash:
            print("Notice! Feature Hashing on the fly currently is not supported in torch version,you can use "
                  "tensorflow version!")
        return super(SparseFeat, cls).__new__(cls, name, vocabulary_size, embedding_dim, use_hash, dtype,
                                              embedding_name, group_name)

    def __hash__(self):
        return self.name.__hash__()


class VarLenSparseFeat(namedtuple("VarLenSparseFeat", ["sparsefeat", "maxlen", "combiner", "length_name"])):
    def __new__(cls, sparsefeat, maxlen, combiner="mean", length_name=None):
        return super(VarLenSparseFeat, cls).__new__(cls, sparsefeat, maxlen, combiner, length_name)

    @property
    def name(self):
        return self.sparsefeat.name

    @property
    def vocabulary_size(self):
        return self.sparsefeat.vocabulary_size

    @property
    def embedding_dim(self):
        return self.sparsefeat.embedding_dim

    @property
    def use_hash(self):
        return self.sparsefeat.use_hash

    @property
    def dtype(self):
        return self.sparsefeat.dtype

    @property
    def embedding_name(self):
        return self.sparsefeat.embedding_name

    @property
    def group_name(self):
        return self.sparsefeat.group_name

    def __hash__(self):
        self.name.__hash__()


class DenseFeat(namedtuple("DenseFeat", ["name", "dimension", "dtype"])):
    __slots__ = ()

    def __new__(cls, name, dimension=1, dtype="float32"):
        return super(DenseFeat, cls).__new__(cls, name, dimension, dtype)

    def __hash__(self):
        return self.name.__hash__()


def build_input_features(feature_columns):
    # Return OrderedDict: {feature_name:(start, start+dimension)}
    features = OrderedDict()
    start = 0

    for feat in feature_columns:
        if feat.name in features:
            continue
        if isinstance(feat, SparseFeat):
            features[feat.name] = (start, start + 1)
            start += 1
        elif isinstance(feat, DenseFeat):
            features[feat.name] = (start, start + feat.dimension)
            start += feat.dimension
        elif isinstance(feat, VarLenSparseFeat):
            features[feat.name] = (start, start + feat.maxlen)
            start += feat.maxlen
            if feat.length_name is not None and feat.length_name not in features:
                features[feat.length_name] = (start, start + 1)
        else:
            raise TypeError("Invalid feature column type, got", type(feat))

    return features


def create_embedding_matrix(feature_columns, init_std=0.001, linear=False, sparse=False):
    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(
        feature_columns) else []
    varlen_sparse_feature_columns = list(filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if len(
        feature_columns) else []

    embedding_dict = nn.ModuleDict(
        {feat.embedding_name: nn.Embedding(feat.vocabulary_size, feat.embedding_dim if not linear else 1, sparse=sparse)
         for feat in sparse_feature_columns + varlen_sparse_feature_columns})

    for tensor in embedding_dict.values():
        nn.init.normal_(tensor.weight, mean=0, std=init_std)
