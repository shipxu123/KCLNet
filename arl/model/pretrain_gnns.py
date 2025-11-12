import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models import GCN, GraphSAGE, GIN, GAT

from .backbone.mlp import MLP
from .backbone.appnp import APPNP
from .backbone.gat_v2 import GATv2, GATv2Conv
from .backbone.agnn import AsyncGNN

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")

__all__ = ['PretrainModel', 'gcn_pretrain_model', 'gin_pretrain_model', 'sage_pretrain_model',
        'gat_pretrain_model', 'appnp_pretrain_model', 'gat_v2_pretrain_model',
        'agnn_pretrain_model']

class PretrainModel(nn.Module):
    '''
        implement this for sane.
        Actually, sane can be seen as the combination of three cells, node aggregator, skip connection, and layer aggregator
        for sane, we dont need cell, since the DAG is the whole search space, and what we need to do is implement the DAG.
    '''
    def __init__(self, model, is_mlp=False):
        super(PretrainModel, self).__init__()
        # node aggregator op
        self.model = model
        self.is_mlp = is_mlp

    def forward(self, data):
        if self.is_mlp:
            x = data.x.float()
            return self.model(x)
        else:
            x, edge_index = data.x.float(), data.edge_index
            return self.model(x, edge_index)
    # def forward(self, x, edge_index):
    #     if self.is_mlp:
    #         return self.model(x)
    #     else:
    #         return self.model(x, edge_index)


class PretrainModel1(nn.Module):
    '''
        implement this for sane.
        Actually, sane can be seen as the combination of three cells, node aggregator, skip connection, and layer aggregator
        for sane, we dont need cell, since the DAG is the whole search space, and what we need to do is implement the DAG.
    '''
    def __init__(self, model, is_mlp=False):
        super(PretrainModel1, self).__init__()
        # node aggregator op
        self.model = model

    def forward(self, data):
        return self.model(data)

def gcn_pretrain_model(**kwargs):
    """
    Constructs a gcn model.
    """
    model = GCN(**kwargs)
    model.output_channels = kwargs.get('output_channels', 64)
    return PretrainModel(model)


def gin_pretrain_model(**kwargs):
    """
    Constructs a gin model.
    """
    model = GIN(**kwargs)
    model.output_channels = kwargs.get('output_channels', 64)
    return PretrainModel(model)


def sage_pretrain_model(**kwargs):
    """
    Constructs a sage model.
    """
    model = GraphSAGE(**kwargs)
    model.output_channels = kwargs.get('output_channels', 64)
    return PretrainModel(model)


def gat_pretrain_model(**kwargs):
    """
    Constructs a gat model.
    """
    model = GAT(**kwargs)
    model.output_channels = kwargs.get('output_channels', 64)
    return PretrainModel(model)

def appnp_pretrain_model(**kwargs):
    """
    Constructs a appnp model.
    """
    model = APPNP(**kwargs)
    model.output_channels = kwargs.get('output_channels', 64)
    return PretrainModel(model)

def gat_v2_pretrain_model(**kwargs):
    """
    Constructs a gat model.
    """
    # model = GATv2Conv(**kwargs)

    # in_channels = kwargs.get('in_channels')
    # hidden_channels = kwargs.get('hidden_channels', 64)
    # heads = kwargs.get('heads', 8)

    model = GATv2(**kwargs)
    model.output_channels = kwargs.get('output_channels', 64)
    return PretrainModel(model)

def mlp_pretrain_model(**kwargs):
    """
    Constructs a mlp model.
    """
    model = MLP(**kwargs)
    model.output_channels = kwargs.get('output_channels', 64)
    return PretrainModel(model, is_mlp=True)

def agnn_pretrain_model(**kwargs):
    """
    Constructs an agnn model.
    """
    model = AsyncGNN(**kwargs, pretrain=True)
    model.output_channels = kwargs.get('output_channels', 64)
    return PretrainModel1(model)