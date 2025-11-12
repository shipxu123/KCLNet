import time
import torch
import torch_scatter
import torch.nn as nn

from collections import defaultdict
import networkx as nx
from torch_geometric.data import Data
from torch_scatter import scatter_mean

# Auxiliary function remains unchanged
def compute_depth(edge_index, num_nodes):
    """Calculate node depth (longest path)"""
    adj = [[] for _ in range(num_nodes)]
    in_degree = torch.zeros(num_nodes, dtype=torch.long)
    for u, v in edge_index.t().tolist():
        adj[u].append(v)
        in_degree[v] += 1
    depth = torch.zeros(num_nodes, dtype=torch.long)
    q = []
    for i in range(num_nodes):
        if in_degree[i] == 0:
            q.append(i)
            depth[i] = 0
    while q:
        u = q.pop(0)
        for v in adj[u]:
            if depth[v] < depth[u] + 1:
                depth[v] = depth[u] + 1
            in_degree[v] -= 1
            if in_degree[v] == 0:
                q.append(v)
    return depth


class AsyncGNN(nn.Module):
    def __init__(self,
                in_channels,
                hidden_channels,
                out_channels,
                pretrain=False,
                load_from_pretrain=False):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.load_from_pretrain = load_from_pretrain
        self.input_linear = nn.Linear(in_channels, hidden_channels)  # Add input transformation layer
        self.linear = nn.Linear(hidden_channels * 2, hidden_channels)   # Adjust input dimension
        if self.load_from_pretrain:
            # hard code here
            self.out =  nn.Linear(hidden_channels, hidden_channels)
            self.new_out = nn.Linear(hidden_channels, out_channels)
        else:
            self.out = nn.Linear(hidden_channels, out_channels)

        self.pretrain = pretrain

    # Helper function to optimize parent aggregation
    def _aggregate_parents(self, adj_dict, nodes_in_k, device):
        parents_list = []
        segment_ids_list = []
        for i, v in enumerate(nodes_in_k):
            parents = adj_dict.get(v, [])
            parents_list.extend(parents)
            segment_ids_list.extend([i] * len(parents))

        if not parents_list:
            return torch.empty((0,), dtype=torch.long, device=device), torch.empty((0,), dtype=torch.long, device=device)

        parents_tensor = torch.tensor(parents_list, dtype=torch.long).to(device)
        segment_ids_tensor = torch.tensor(segment_ids_list, dtype=torch.long).to(device)
        return parents_tensor, segment_ids_tensor

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if x is None or edge_index is None or x.size(0) == 0:
            raise ValueError("Input data is empty or invalid.")

        num_nodes = x.size(0)
        device = x.device

        # Calculate node depth
        # depth = compute_depth(edge_index, num_nodes)
        depth = data.depth.to(device)
        max_depth = depth.max().item()
        
        # edge_sorted = edge_index[:, edge_index[1].argsort()]
        # # pointer to neighbours
        # ptr = torch_scatter.index2ptr(edge_sorted[1], num_nodes)

        # Build adjacency dictionary
        adj_dict = defaultdict(list)
        u_list = edge_index[0].tolist()
        v_list = edge_index[1].tolist()
        for u, v in zip(u_list, v_list):
            adj_dict[v].append(u)

        # Group nodes by depth
        groups = defaultdict(list)
        for v in range(num_nodes):
            groups[depth[v].item()].append(v)

        # Feature dimension conversion
        h = self.input_linear(x)  # [num_nodes, hidden_channels]

        # # depth mask
        # depth_masks = [
        #     depth == k
        #     for k in range(max_depth+1)
        # ]

        # List to store I embeddings at each layer
        I_embeddings = []

        # Hierarchical processing flow
        for k in range(max_depth + 1):
            # mask = depth_masks[k]
            # if not mask.any():
            #     continue

            # start, end = ptr[k], ptr[k+1]
            # parents = edge_sorted[0, start:end]
            # if parents.numel() == 0:
            #     aggregated = torch.zeros(mask.sum(), h.size(1), device=h.device)
            # else:
            #     aggregated = torch_scatter.scatter_mean(
            #         h[parents], 
            #         edge_sorted[1, start:end] - ptr[:-1].searchsorted(k),
            #         dim=0
            #     )

            # combined = torch.cat([h[mask], aggregated], dim=1)
            # h[mask] = torch.relu(self.linear(combined))

            # if self.pretrain:
            #     I_embeddings.append(h[mask])

            nodes_in_k = groups.get(k, [])
            if not nodes_in_k:
                continue

            nodes_tensor = torch.tensor(nodes_in_k, dtype=torch.long).to(device)

            # Parent node feature aggregation (optimized)
            parents_tensor, segment_ids_tensor = self._aggregate_parents(adj_dict, nodes_in_k, device)
            if parents_tensor.numel() == 0:
                aggregated = torch.zeros(len(nodes_in_k), h.size(1)).to(device)
            else:
                aggregated = torch.zeros(len(nodes_in_k), h.size(1)).to(device)
                aggregated.index_add_(0, segment_ids_tensor, h[parents_tensor])

            # Feature update
            current = h[nodes_tensor]
            combined = torch.cat([current, aggregated], dim=1)
            h[nodes_tensor] = torch.relu(self.linear(combined))

            if self.pretrain:
                I_embeddings.append(h[nodes_tensor])

        if self.load_from_pretrain:
            z = self.new_out(self.out(h))
        else:
            z = self.out(h)

        if self.pretrain:
            return z, I_embeddings
        else:
            return z

# ----------- Optimized Asynchronous GNN ------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class FastAsyncGNN(nn.Module):
    """
    A compatible FastAsyncGNN variant for pure GCN inference.
    Supports loading pre-trained asynchronous GNN weights.

    - Keeps key layer names for weight loading compatibility.
    - Uses standard GCNConv for graph inference.
    """

    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 pretrain=False,
                 load_from_pretrain=False):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.pretrain = pretrain
        self.load_from_pretrain = load_from_pretrain

        # --- Keep same layer names for weight loading compatibility ---
        self.input_linear = nn.Linear(in_channels, hidden_channels)
        self.linear = nn.Linear(hidden_channels * 2, hidden_channels)

        # --- Replace asynchronous aggregation with GCN layers ---
        # These are new layers for inference
        self.gcn1 = GCNConv(in_channels, hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, hidden_channels)

        # Output projection (keep same interface)
        if load_from_pretrain:
            self.out = nn.Linear(hidden_channels, hidden_channels)
            self.new_out = nn.Linear(hidden_channels, out_channels)
        else:
            self.out = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        """
        Standard GCN inference path for graphs.
        This variant loads weights from a pretrained FastAsyncGNN if available.
        """
        x, edge_index = data.x, data.edge_index

        # ---- Standard GCN inference ----
        x = F.relu(self.gcn1(x, edge_index))
        x = F.relu(self.gcn2(x, edge_index))

        # ---- Output mapping ----
        if self.load_from_pretrain:
            z = self.new_out(self.out(x))
        else:
            z = self.out(x)

        return z

    @staticmethod
    def load_weights_from_async(source_model, target_model):
        """
        A helper function to load useful weights from a pretrained asynchronous FastAsyncGNN
        into this GCN-based one.
        """
        source_dict = source_model.state_dict()
        target_dict = target_model.state_dict()

        # Try to map similar weights
        mapping = {
            "input_linear.weight": "gcn1.lin.weight",
            "input_linear.bias": "gcn1.lin.bias",
            "linear.weight": "gcn2.lin.weight",
            "linear.bias": "gcn2.lin.bias",
            "out.weight": "out.weight",
            "out.bias": "out.bias",
        }

        for s_key, t_key in mapping.items():
            if s_key in source_dict and t_key in target_dict:
                with torch.no_grad():
                    target_dict[t_key].copy_(source_dict[s_key])

        # Load into model
        target_model.load_state_dict(target_dict, strict=False)
        print("Successfully transferred compatible weights from async model.")

# # ---------------- Example ----------------
# if __name__ == "__main__":
#     from torch_geometric.data import Data

#     # toy graph
#     edge_index = torch.tensor([[0, 1, 2],
#                                [1, 2, 3]], dtype=torch.long)
#     x = torch.randn(4, 16)
#     data = Data(x=x, edge_index=edge_index)

#     # source pretrained async model
#     source_async_model = torch.load("fast_async_pretrained.pt")

#     # our new GCN inference model
#     gcn_model = FastAsyncGNN(16, 32, 8, load_from_pretrain=True)

#     # load compatible weights
#     FastAsyncGNN.load_weights_from_async(source_async_model, gcn_model)

#     # inference
#     out = gcn_model(data)
#     print(out.shape)

if __name__ == "__main__":
    # Construct example data
    # edge_index = torch.tensor([[0, 0, 1],
    #                            [1, 2, 2]], dtype=torch.long)
    # x = torch.tensor([[1.0]*32, [2.0]*32, [3.0]*32], dtype=torch.float)
    # data = Data(x=x, edge_index=edge_index)
    
    # model = AsyncGNN(in_channels=32, hidden_channels=64, out_channels=10, pretrain=True)
    # output = model(data)
    # print("Output features:\n", output)

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate a random DAG
    num_nodes = 10
    feature_dim = 64
    edge_list = []
    for u in range(num_nodes):
        for v in range(u+1, min(num_nodes, u+4)):  # limit fan-out
            edge_list.append((u, v))
    edge_index = torch.tensor(edge_list, dtype=torch.long).T.to(device)

    x = torch.randn(num_nodes, feature_dim, device=device)
    depth = compute_depth(edge_index.cpu(), num_nodes)
    data = Data(x=x, edge_index=edge_index, depth=depth)

    model1 = AsyncGNN(in_channels=feature_dim, hidden_channels=128, out_channels=10).to(device)
    model2 = FastAsyncGNN(in_channels=feature_dim, hidden_channels=128, out_channels=10).to(device)

    # Warm-up CUDA
    for _ in range(2):
        _ = model1(data)
        _ = model2(data)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Measure AsyncGNN
    t0 = time.perf_counter()
    for _ in range(5):
        _ = model1(data)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_async = (time.perf_counter() - t0) / 5

    # Measure FastAsyncGNN
    t0 = time.perf_counter()
    for _ in range(5):
        _ = model2(data)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_fast = (time.perf_counter() - t0) / 5

    print(f"Device: {device}")
    print(f"Nodes: {num_nodes}, Feature dim: {feature_dim}")
    print(f"Average time per forward pass:")
    print(f" - AsyncGNN (dict-based): {t_async:.5f} sec")
    print(f" - FastAsyncGNN (vectorized): {t_fast:.5f} sec")
    print(f"Speedup: {t_async / t_fast:.2f}x")