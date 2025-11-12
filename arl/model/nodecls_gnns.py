import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models import GCN, GraphSAGE, GIN, GAT

from .backbone.mlp import MLP
from .backbone.agnn import AsyncGNN
from .backbone.appnp import APPNP
from .backbone.gat_v2 import GATv2, GATv2Conv

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")

__all__ = ['NodeClsModel', 'gcn_nodecls_model', 'gin_nodecls_model', 'sage_nodecls_model',
        'gat_nodecls_model', 'mlp_nodecls_model', 'gat_v2_nodecls_model',
        'mlp_nodecls_model', 'agnn_nodecls_model']

class NodeClsModel(nn.Module):
    '''
        implement this for sane.
        Actually, sane can be seen as the combination of three cells, node aggregator, skip connection, and layer aggregator
        for sane, we dont need cell, since the DAG is the whole search space, and what we need to do is implement the DAG.
    '''
    def __init__(self,
                model,
                from_pyg=True,
                is_mlp=False):
        super(NodeClsModel, self).__init__()
        # node aggregator op
        self.model = model
        self.is_mlp = is_mlp
        self.from_pyg = from_pyg

    def forward(self, data):
        if self.is_mlp:
            x = data.x.float()
            return self.model(x)
        elif self.from_pyg:
            x, edge_index = data.x.float(), data.edge_index
            return self.model(x, edge_index)
        else:
            return self.model(data)

def gcn_nodecls_model(**kwargs):
    """
    Constructs a gcn model.
    """
    model = GCN(**kwargs)
    return NodeClsModel(model)


def gin_nodecls_model(**kwargs):
    """
    Constructs a gin model.
    """
    model = GIN(**kwargs)
    return NodeClsModel(model)


def sage_nodecls_model(**kwargs):
    """
    Constructs a sage model.
    """
    model = GraphSAGE(**kwargs)
    return NodeClsModel(model)


def gat_nodecls_model(**kwargs):
    """
    Constructs a gat model.
    """
    model = GAT(**kwargs)
    return NodeClsModel(model)

def appnp_nodecls_model( **kwargs):
    """
    Constructs a appnp model.
    """
    model = APPNP(**kwargs)
    return NodeClsModel(model)

def gat_v2_nodecls_model(**kwargs):
    """
    Constructs a gat model.
    """
    # model = GATv2Conv(**kwargs)

    # in_channels = kwargs.get('in_channels')
    # hidden_channels = kwargs.get('hidden_channels', 64)
    # heads = kwargs.get('heads', 8)

    # # Create a multi-layer GATv2 model
    # model = nn.Sequential(
    #     GATv2Conv(in_channels, hidden_channels, heads=heads),
    #     nn.ReLU(),
    #     nn.Dropout(kwargs.get('dropout', 0.5)),
    #     GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads),
    #     nn.ReLU(),
    #     nn.Dropout(kwargs.get('dropout', 0.5)),
    #     GATv2Conv(hidden_channels * heads, n_class, heads=1)
    # )
    model = GATv2(**kwargs)
    model.out_channels = kwargs.get('out_channels', 11)

    return NodeClsModel(model)

def mlp_nodecls_model(**kwargs):
    """
    Constructs a mlp model.
    """
    model = MLP(**kwargs)
    return NodeClsModel(model, is_mlp=True)

def agnn_nodecls_model(**kwargs):
    """
    Constructs an agnn model.
    """
    model = AsyncGNN(**kwargs)
    return NodeClsModel(model, from_pyg=False)