import torch
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool


# torch_geometric/nn/pool/sum_pool.py

import torch
from torch_scatter import scatter_add
from torch_geometric.nn.pool import global_add_pool

def global_sum_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    r"""Global sum pooling of node embeddings.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (LongTensor): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example.

    :rtype: :class:`Tensor`
    """
    return scatter_add(x, batch, dim=0)

def readout_function(h, readout, batch):
    if readout == 'sum':
        return global_add_pool(h, batch)
    elif readout == 'mean':
        return global_mean_pool(h, batch)
    elif readout == 'max':
        return global_max_pool(h, batch)
    else:
        raise ValueError(f"Unsupported readout type: {readout}")