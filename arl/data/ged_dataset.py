import os
import json
import torch
import random

import numpy as np
import os.path as osp
import networkx as nx
import scipy.sparse as sp
import multiprocessing as mp
import torch.nn.functional as F

from typing import List, Optional
from torch_geometric.data import Data, InMemoryDataset, Batch, DataLoader
from torch_geometric.utils import from_networkx

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

    ged_labels = []
    for node in nodes:
        print(node)
        ged_label = G.nodes[node].get('ged_label', -1)
        ged_labels.append(ged_label)
    data.ged_label = torch.tensor(ged_labels, dtype=torch.long)

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
            return np.zeros((1, 131))
        # else:
            # raise ValueError(f"Unsupported instance type: {inst_type}")

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
        # data.node_label  = torch.tensor(base_graph.has_source, dtype=torch.long)
        data.graph_label = torch.tensor([graph_label], dtype=torch.long)
        device_type_indices = [device_types.index(nt) if nt in device_types else len(device_types) - 1 for nt in node_types]
        device_type_one_hot = F.one_hot(torch.tensor(device_type_indices), num_classes=len(device_types)).float()
        data.x = torch.cat([device_type_one_hot, data.x], dim=1)
        data.depth = torch.tensor(base_graph.depth, dtype=torch.long)
        dataset.append(data)
    return dataset


class GEDDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None, pre_filter=None, args=None):
        self.args = args
        self.name = name
        self.feature_path = os.path.join(root, 'raw', 'features')
        self.graph_path = os.path.join(root, 'raw', 'graphs')
        self.ged_records_path = os.path.join(root, 'raw', 'mutated_graphs/ged_records.json')
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self) -> List[str]:
        return ['ged_records.json']

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

        # Load mutation records
        with open(self.ged_records_path, 'r') as f:
            ged_records = json.load(f)
        mutation_records = ged_records['mutation_records']

        dataset = []
        for record in mutation_records:
            base_id        = record['base_graph_id']
            mutated_id     = record['mutated_graph_id']
            edit_distance = torch.tensor([record['edit_distance']['total']], dtype=torch.float)

            # Load graph structures
            base_graph = load_base_graph(os.path.join(self.graph_path, f"{base_id}.json"))
            mutated_graph = load_base_graph(os.path.join(self.graph_path, f"{mutated_id}.json"))

            # Generate feature variations
            base_samples = generate_graph_dataset(
                base_graph=base_graph,
                fft_base_path=self.feature_path,
                total_samples=self.args.n_samples,
                graph_label=-1  # Placeholder
            )
            mutated_samples = generate_graph_dataset(
                base_graph=mutated_graph,
                fft_base_path=self.feature_path,
                total_samples=self.args.n_samples,
                graph_label=-1
            )

            # Create graph pairs
            for _ in range(self.args.n_samples_per_pair):
                base_sample = random.choice(base_samples)
                mutated_sample = random.choice(mutated_samples)

                data = Data(
                    x=mutated_sample.x,             # Mutated graph features
                    edge_index=mutated_sample.edge_index,
                    depth=mutated_sample.depth,
                    x2=base_sample.x,
                    edge_index2=base_sample.edge_index,
                    depth2=base_sample.depth,
                    # x=base_sample.x,
                    # edge_index=base_sample.edge_index,
                    # depth=base_sample.depth,
                    # x2=mutated_sample.x,             # Mutated graph features
                    # edge_index2=mutated_sample.edge_index,
                    # depth2=mutated_sample.depth,
                    y=edit_distance                 # GED label
                )

                dataset.append(data)

        data_list = dataset
        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        # Save processed data
        data, slices = self.collate(dataset)
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


if __name__ == "__main__":
    args = type('', (), {})()
    args.pretrained_model = 'model_name'
    args.gpu = 0

    def show_dataset(dataset):
        print(f"Dataset saved, contains {len(dataset)} samples")

        # # Print graph feature example
        # print("Sample structure example:", dataset[0])
        # print("Graph feature example 1 : Graph 1 - ", dataset[0].x.shape)
        # print("Graph feature example 1 : Graph 2 - ", dataset[0].x2.shape)
        # print("Graph label example 1 : ", dataset[0].y)


        # Print graph feature example
        # print("Sample structure example:", dataset[1])
        # print("Graph feature example 2 : Graph 1 - ", dataset[1].x.shape)
        # print("Graph feature example 2 : Graph 2 - ", dataset[1].x2.shape)
        # print("Graph label example 2 : ", dataset[1].y)


    # args.n_samples = 70
    # args.n_samples_per_pair = 70
    # ged_cls_dataset = GEDDataset(root='./ged_dataset', name='train', args=args)
    # show_dataset(ged_cls_dataset)

    # args.n_samples = 10
    # args.n_samples_per_pair = 10
    # ged_cls_dataset = GEDDataset(root='./ged_dataset', name='val', args=args)
    # show_dataset(ged_cls_dataset)

    # args.n_samples = 20
    # args.n_samples_per_pair = 20
    # ged_cls_dataset = GEDDataset(root='./ged_dataset', name='test', args=args)
    # show_dataset(ged_cls_dataset)

    #args.n_samples = 350
    #args.n_samples_per_pair = 350
    #ged_cls_dataset = GEDDataset(root='./ged_dataset_new', name='train', args=args)
    #show_dataset(ged_cls_dataset)

    args.n_samples = 50
    args.n_samples_per_pair = 50
    ged_cls_dataset = GEDDataset(root='./ged_dataset_timing', name='val', args=args)
    #show_dataset(ged_cls_dataset)

    #args.n_samples = 100
    #args.n_samples_per_pair = 100
    #ged_cls_dataset = GEDDataset(root='./ged_dataset_new', name='test', args=args)
    #show_dataset(ged_cls_dataset)

    # loader = DataLoader(ged_cls_dataset, batch_size=32, shuffle=True)

    # for i, data in enumerate(loader):
    #     print(i, data)
