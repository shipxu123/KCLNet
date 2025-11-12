import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models import GCN, GraphSAGE, GIN, GAT


from .backbone.mlp import MLP
from .backbone.agnn import AsyncGNN
from .backbone.appnp import APPNP
from .backbone.gat_v2 import GATv2, GATv2Conv
from .backbone.basic_readout import readout_function

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")

__all__ = ['GraphClsModel', 'gcn_graphcls_model', 'gin_graphcls_model', 'sage_graphcls_model', 
            'gat_graphcls_model', 'appnp_graphcls_model', 'gat_v2_graphcls_model',
            'mlp_graphcls_model', 'agnn_graphcls_model', 'fast_agnn_graphcls_model']

class GraphClsModel(nn.Module):
    '''
        implement this for sane.
        Actually, sane can be seen as the combination of three cells, node aggregator, skip connection, and layer aggregator
        for sane, we dont need cell, since the DAG is the whole search space, and what we need to do is implement the DAG.
    '''
    def __init__(self,
                fc_hidden,
                n_class,
                readout,
                model,
                from_pyg=True,
                is_mlp=False):

        super(GraphClsModel, self).__init__()

        # node aggregator op
        self.readout = readout
        self.fc_hidden = fc_hidden
        self.n_class = n_class
        self.model = model
        self.is_mlp = is_mlp
        self.from_pyg = from_pyg
        # self.fc1 = nn.Linear(self.model.out_channels, fc_hidden)
        # self.fc2 = nn.Linear(fc_hidden, n_class)
        self.fc = nn.Linear(self.model.out_channels, n_class)


    def forward(self, data):
        if self.is_mlp:
            x = data.x.float()
            h = self.model(x)
        elif self.from_pyg:
            x, edge_index = data.x.float(), data.edge_index
            h = self.model(x, edge_index)
        else:
            h = self.model(data)

        # Readout
        h = readout_function(h, self.readout, batch=data.batch)
        
        # Fully-connected layer
        # h = F.relu(self.fc1(h))
        # h = F.softmax(self.fc2(h))

        return self.fc(h)



def gcn_graphcls_model(fc_hidden, n_class, readout, **kwargs):
    """
    Constructs a gcn model.
    """
    model = GCN(**kwargs)
    model.output_channels = kwargs.get('output_channels', 64)
    return GraphClsModel(fc_hidden, n_class, readout, model)


def gin_graphcls_model(fc_hidden, n_class, readout, **kwargs):
    """
    Constructs a gin model.
    """
    model = GIN(**kwargs)
    model.output_channels = kwargs.get('output_channels', 64)
    return GraphClsModel(fc_hidden, n_class, readout, model)


def sage_graphcls_model(fc_hidden, n_class, readout, **kwargs):
    """
    Constructs a sage model.
    """
    model = GraphSAGE(**kwargs)
    model.output_channels = kwargs.get('output_channels', 64)
    return GraphClsModel(fc_hidden, n_class, readout, model)


def gat_graphcls_model(fc_hidden, n_class, readout, **kwargs):
    """
    Constructs a gat model.
    """
    model = GAT(**kwargs)
    model.output_channels = kwargs.get('output_channels', 64)
    return GraphClsModel(fc_hidden, n_class, readout, model)

def appnp_graphcls_model(fc_hidden, n_class, readout, **kwargs):
    """
    Constructs a appnp model.
    """
    model = APPNP(**kwargs)
    model.output_channels = kwargs.get('output_channels', 64)
    return GraphClsModel(fc_hidden, n_class, readout, model)

def gat_v2_graphcls_model(fc_hidden, n_class, readout, **kwargs):
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
    model.output_channels = kwargs.get('output_channels', 64)
    model.out_channels = kwargs.get('hidden_channels', 64)
    return GraphClsModel(fc_hidden, n_class, readout, model)

def mlp_graphcls_model(fc_hidden, n_class, readout, **kwargs):
    """
    Constructs a mlp model.
    """
    model = MLP(**kwargs)
    model.output_channels = kwargs.get('output_channels', 64)
    return GraphClsModel(fc_hidden, n_class, readout, model, is_mlp=True)

def agnn_graphcls_model(fc_hidden, n_class, readout, **kwargs):
    """
    Constructs an agnn model.
    """
    model = AsyncGNN(**kwargs)
    model.output_channels = kwargs.get('output_channels', 64)
    return GraphClsModel(fc_hidden, n_class, readout, model, from_pyg=False)


def fast_agnn_graphcls_model(fc_hidden, n_class, readout, **kwargs):
    """
    Constructs an agnn model.
    """
    model = FastAsyncGNN(**kwargs)
    model.output_channels = kwargs.get('output_channels', 64)
    return GraphClsModel(fc_hidden, n_class, readout, model, from_pyg=False)