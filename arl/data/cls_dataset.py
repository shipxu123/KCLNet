import os
import json
import torch
import random
import pickle
import itertools

import numpy as np
import os.path as osp
import networkx as nx
import scipy.sparse as sp
import multiprocessing as mp
import torch.nn.functional as F

from typing import List, Callable, Optional
from torch_geometric.data import Data, InMemoryDataset
from networkx.algorithms.similarity import graph_edit_distance


device_mapping = {
    "cap": 0,    # Passive device (capacitor)
    "diode": 0,  # Passive device (diode)
    "nmos": 1,   # Active device (N-type MOSFET)
    "npn": 1,    # Active device (NPN bipolar junction transistor)
    "pmos": 1,   # Active device (P-type MOSFET)
    "pnp": 1,    # Active device (PNP bipolar junction transistor)
    "res": 0,    # Passive device (resistor)
    "ind": 0,     # Passive device (inductor)
    "net": -1,
    "voltage": -2,
    "gnd": -3
}


# circuit_mapping = {
    # "new_case10cg.json": 8,
    # "new_case11cg.json": 8,
    # "new_case1cg.json": 0,
    # "new_case2cg.json": 1,
    # "new_case3cg.json": 2,
    # "new_case4cg-2.json": 3,
    # "new_case4.json": 4,
    # "new_case5.json": 5,
    # "new_case6.json": 6,
    # "new_case7.json": 7,
    # "new_case8.json": 7,
    # "new_case9.json": 7,
    # "new_case10.json": 8,
    # "new_case11.json": 8,
    # "new_case1.json": 0,
    # "new_case2.json": 1,
    # "new_case3.json": 2,
    # "new_case4cg.json": 3,
    # "new_case5cg.json": 4,
    # "new_case6cg.json": 5,
    # "new_case7cg.json": 6,
    # "new_case8cg.json": 7,
    # "new_case9cg.json": 7
# }

# device_types = ["cap", "diode", "nmos", "npn", "pmos", "pnp", "res", "ind", "net", "voltage"]
circuit_mapping = {
    # 0类：数据转换器
    "dac_r2r.json": 0,
    "dac_r2r-fix.json": 0,
    "mux4_sar_x1.json": 0,
    "mux4_sar_x1-fix.json": 0,
    "mux4_sar_x2.json": 0,
    "mux4_sar_x2-fix.json": 0,

    # 1类：数字逻辑电路
    "dec3to7.json": 1,
    "dec3to7-fix.json": 1,
    "mux4_dig_18.json": 1,
    "mux4_dig_18-fix.json": 1,
    "R8_pnp10_3.json": 1,
    "R8_pnp10_3-fix.json": 1,
    "signbit.json": 1,
    "signbit-fix.json": 1,

    # 2类：电源管理
    "ldo_r180k.json": 2,
    "ldo_r180k-fix.json": 2,

    # 3类：模拟电路组件-电流基准
    "IREF5U_SEL.json": 3,
    "IREF5U_SEL-fix.json": 3,

    # 4类：模拟电路组件-运算放大器
    "op_fd_1st.json": 4,
    "op_fd_1st-fix.json": 4,

    # 5类：模拟电路组件-电阻分压器
    "resdiv.json": 5,
    "resdiv-fix.json": 5,

    # 6类：模拟电路组件-带隙基准
    "mux4_bg.json": 6,
    "mux4_bg-fix.json": 6,

    # 7类：时钟电路
    "osc_40m_top.json": 7,
    "osc_40m_top-fix.json": 7,
    "pll_pfd.json": 7,
    "pll_pfd-fix.json": 7,
    "pll_post.json": 7,
    "pll_post-fix.json": 7,

    # 8类：I/O电路
    "op_buf_pad.json": 8,
    "op_buf_pad-fix.json": 8,

    # 9类：缓冲器
    "op_buf_cal.json": 9,
    "op_buf_cal-fix.json": 9,

    # 10类：备用单元
    "spare_cell_33.json": 10,
    "spare_cell_33-fix.json": 10
}


device_types = ["cap", "diode", "nmos", "npn", "pmos", "pnp", "res", "ind", "net", "voltage", "gnd"]
type2index = {v: k for k, v in enumerate(device_types)}


def topological_sort_with_depth(G, start_nodes):
    depth = {node: -1 for node in G.nodes()}
    for node in start_nodes:
        depth[node] = 0

    queue = start_nodes[:]
    visited = set(start_nodes)

    while queue:
        current = queue.pop(0)
        for neighbor in G.neighbors(current):
            if neighbor not in visited:
                depth[neighbor] = depth[current] + 1
                visited.add(neighbor)
                queue.append(neighbor)

    depth_tensor = torch.tensor([depth[node] for node in sorted(G.nodes())], dtype=torch.long)
    return depth_tensor


def load_base_graph(json_path):
    print(f"load graph from {json_path}")
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    G = nx.json_graph.node_link_graph(json_data)

    nodes = sorted(G.nodes())
    num_nodes = len(nodes)
    adj = nx.adjacency_matrix(G, nodelist=nodes)
    adj_coo = adj.tocoo()

    row = torch.from_numpy(adj_coo.row).to(torch.long)
    col = torch.from_numpy(adj_coo.col).to(torch.long)
    edge_index = torch.stack([row, col], dim=0)

    if all('feature' in G.nodes[node] for node in nodes):
        x = [G.nodes[node]['feature'] for node in nodes]
        x = torch.tensor(x, dtype=torch.float)
    else:
        x = torch.eye(num_nodes, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)

    if adj_coo.data is not None:
        edge_weight = torch.tensor(adj_coo.data, dtype=torch.float).unsqueeze(1)
        data.edge_attr = edge_weight

    if all('label' in G.nodes[node] for node in nodes):
        y = [G.nodes[node]['label'] for node in nodes]
        data.y = torch.tensor(y, dtype=torch.long)

    has_source = []
    for node in nodes:
        print(node)
        inst_type = G.nodes[node].get('inst_type', 'unknown').lower()
        has_source.append(device_mapping[inst_type])
    data.has_source = torch.tensor(has_source, dtype=torch.long)

    node_types = [G.nodes[node].get('inst_type', 'unknown').lower() for node in nodes]
    data.node_types = node_types

    voltage_nodes = [node for node, attr in G.nodes(data=True) if attr.get('inst_type') == 'voltage']
    depth_tensor = topological_sort_with_depth(G, voltage_nodes)
    data.depth = depth_tensor

    return data


def load_fft_data(inst_type, base_path):
    supported_types = ["cap", "diode", "nmos", "npn", "pmos", "pnp", "res", "ind"]

    if inst_type == "voltage":
        return np.ones((1, 131))
    elif inst_type == "gnd":
        return np.ones((1, 131)) * -1.0
    elif inst_type == "net":
        return np.zeros((1, 131))
    else:
        if inst_type not in supported_types:
            raise ValueError(f"Unsupported instance type: {inst_type}")

    path = f"{base_path}/{inst_type}_fft.npy"
    return np.load(path)


def generate_graph_dataset(base_graph, fft_base_path, total_samples, graph_label, padding_types=("cap", "ind", "res", "ind")):
    node_types = [base_graph.node_types[i].lower() for i in range(base_graph.num_nodes)]
    edge_index = base_graph.edge_index

    fft_features = {}
    max_dim = 0
    
    unique_types = set(node_types)
    used_indexes = set()

    def get_sample_indexes():
        sampled_indexes = []
        sampled_features = []
        for nt in node_types:
            features = fft_features.get(nt, np.ones((1, max_dim)))
            num_features = len(features)
            idx = np.random.randint(num_features)
            sampled_indexes.append(idx)
            sampled_features.append(features[idx])
        return sampled_indexes, sampled_features


    for inst_type in unique_types:
        try:
            arr = load_fft_data(inst_type, fft_base_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load FFT data for {inst_type} - {str(e)}")

        if inst_type not in padding_types:
            dim = arr.shape[1]
            if dim > max_dim:
                max_dim = dim

        fft_features[inst_type] = arr

    for inst_type in padding_types:
        if inst_type in fft_features:
            arr = fft_features[inst_type]
            required_padding = max_dim - arr.shape[1]
            
            if required_padding > 0:
                fft_features[inst_type] = np.pad(
                    arr, 
                    [(0, 0), (0, required_padding)],
                    mode='constant'
                )
            elif required_padding < 0:
                raise ValueError(f"Feature dimension anomaly: {inst_type} dimension exceeds others")

    dataset = []
    for _ in range(total_samples):
        sampled_indexes, sampled_features = get_sample_indexes()
        while str(sampled_indexes) in used_indexes:
            sampled_indexes, sampled_features = get_sample_indexes()

        x = torch.tensor(np.vstack(sampled_features), dtype=torch.float32)

        data = Data(x=x, edge_index=edge_index)
        data.node_label  = torch.tensor(base_graph.has_source, dtype=torch.long)
        data.graph_label = torch.tensor([graph_label], dtype=torch.long)
        # TODO: change to cluster_label later
        data.cluster_label = torch.tensor([type2index[node_type] for node_type
                                             in base_graph.node_types], dtype=torch.long)

        device_type_indices = [device_types.index(nt) if nt in device_types else len(device_types) - 1 for nt in node_types]
        device_type_one_hot = F.one_hot(torch.tensor(device_type_indices), num_classes=len(device_types)).float()

        data.x = torch.cat([device_type_one_hot, data.x], dim=1)

        data.depth = torch.tensor(base_graph.depth, dtype=torch.long)

        dataset.append(data)

    return dataset


class ClsDataset(InMemoryDataset):
    '''
        A base dataset class for node and graph classification tasks, inheriting from InMemoryDataset.
        This class is used to load, process, and manage graph data for machine learning models.
    '''

    def __init__(self, root, name, transform=None, pre_transform=None, pre_filter=None, args=None):
        '''
        Initializes the dataset with specific configurations.
        
        Parameters:
            root (str): The root directory of the dataset.
            transform (callable, optional): A function/transform that takes in an object and returns a transformed version.
            pre_transform (callable, optional): A function/transform that takes in an object and returns a transformed version, applied before saving to disk.
            pre_filter (callable, optional): A function that takes in an object and returns a boolean, used for filtering objects before saving to disk.
            args (namespace, optional): Additional arguments for dataset processing.
        '''
        self.args = args
        self.name = name
        self.feature_path = os.path.join(root, 'raw', 'features')
        self.graph_path   = os.path.join(root, 'raw', 'graphs')
        super().__init__(root, transform, pre_transform, pre_filter)
        self.process()
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        '''
        Specifies the list of raw file names.
        
        Returns:
            list: A list of raw file names.
        '''
        return []

    @property
    def processed_file_names(self):
        '''
        Specifies the name of the processed file.
        
        Returns:
            str: The name of the processed file.
        '''
        return self.name + '_pyg.pt'

    def download(self):
        pass

    def process(self):
        '''
        Processes the raw data and saves it to disk if not already processed.
        It reads graph data from JSON files, applies transformations, and filters before saving.
        '''
        processed_path = self.processed_paths[0]
        if os.path.exists(processed_path):
            print('Processed file exists, skipping...')
            return

        if not os.path.exists(self.feature_path):
            os.makedirs(self.feature_path)

        if not os.path.exists(self.graph_path):
            os.makedirs(self.graph_path)

        json_files = [os.path.join(self.graph_path, f) for f in os.listdir(self.graph_path) if f.endswith('.json')]
        dataset = []

        for json_file in json_files:
            base_graph = load_base_graph(json_file)
            sub_dataset = generate_graph_dataset(
                base_graph=base_graph,
                fft_base_path=self.feature_path,
                total_samples=self.args.n_samples,
                graph_label=circuit_mapping[os.path.basename(json_file)]
            )
            dataset.extend(sub_dataset)

        data_list = dataset
        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print('Saving...')

    def has_cache(self):
        '''
        Checks if the processed dataset file exists in the specified location.
        
        Returns:
            bool: True if the cache file exists, False otherwise.
        '''
        if os.path.exists(self.processed_paths[0]):
            print('cache found')
            return True
        else:
            print('cache not found')
            return False

    def load(self):
        '''
        Loads the processed dataset from disk.
        '''
        print('loading dataset from ' + self.processed_paths[0])
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        # self.data, self.slices = torch.load(self.processed_paths[0])

    def save(self):
        '''
        Saves the processed dataset to disk.
        '''
        print('saving dataset to ' + self.processed_paths[0])
        torch.save((self.data, self.slices), self.processed_paths[0])

    def __getitem__(self, idx):
        if self.slices is None:
            return self.data
        else:
            return self.get(idx)

    def __len__(self):
        if self.slices is None:
            return 1
        else:
            return len(self.slices['x']) - 1


class NodeClsDataset(ClsDataset):

    def __init__(self, root, name, transform=None, pre_transform=None, pre_filter=None, args=None):
        super().__init__(root, name, transform, pre_transform, pre_filter, args)


class GraphClsDataset(ClsDataset):
    def __init__(self, root, name, transform=None, pre_transform=None, pre_filter=None, args=None):
        super().__init__(root, name, transform, pre_transform, pre_filter, args)


class ClusterClsDataset(ClsDataset):
    def __init__(self, root, name, transform=None, pre_transform=None, pre_filter=None, args=None):
        super().__init__(root, name, transform, pre_transform, pre_filter, args)


if __name__ == "__main__":
    args = type('', (), {})()
    args.pretrained_model = 'model_name'
    args.n_samples = 10000
    args.gpu = 0

    def show_dataset(dataset):
        print(f"Dataset saved, contains {len(dataset)} samples")
        print("Sample structure example:", dataset[0])

        # Print graph feature example
        print("Graph feature example:", dataset[0].x.shape)
        print("Graph feature example:", dataset[0].x)

        # Print graph label example
        print("Graph label example:", dataset[0].graph_label)
        print("Node source label example:", dataset[0].node_label)

        # Print graph feature example
        print("Graph feature example:", dataset[1].x.shape)
        print("Graph feature example:", dataset[1].x)

        # Print graph label example
        print("Graph label example:", dataset[1].graph_label)
        print("Node label example:", dataset[1].node_label)


    #node_cls_dataset  = NodeClsDataset(root='./dataset', name='train', args=args)
    #graph_cls_dataset = GraphClsDataset(root='./dataset', name='train', args=args)
    #cluster_cls_dataset = ClusterClsDataset(root='./dataset', name='train', args=args)
    #show_dataset(node_cls_dataset)
    #show_dataset(graph_cls_dataset)
    #show_dataset(cluster_cls_dataset)

    args.n_samples = 800
    #node_cls_dataset  = NodeClsDataset(root='./dataset', name='val', args=args)
    graph_cls_dataset = GraphClsDataset(root='./dataset_timing', name='val', args=args)
    #cluster_cls_dataset = ClusterClsDataset(root='./dataset', name='val', args=args)
    #show_dataset(node_cls_dataset)
    show_dataset(graph_cls_dataset)
    #show_dataset(cluster_cls_dataset)

    #args.n_samples = 1100
    #node_cls_dataset  = NodeClsDataset(root='./dataset', name='test', args=args)
    #graph_cls_dataset = GraphClsDataset(root='./dataset', name='test', args=args)
    #cluster_cls_dataset = ClusterClsDataset(root='./dataset', name='test', args=args)
    #show_dataset(node_cls_dataset)
    #show_dataset(graph_cls_dataset)
    #show_dataset(cluster_cls_dataset)
