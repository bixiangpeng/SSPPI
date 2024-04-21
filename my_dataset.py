# -*- coding:utf-8 -*-
"""
作者：毕宸浩
日期：2023年07月24日
"""
from torch.utils.data import Dataset
import torch
from torch_geometric import data as DATA
from torch_geometric.data import Batch
import torch_geometric.utils as utils
from torch_scatter import scatter_add
import numpy as np
import pickle as pkl

def collate_dgl(batch):
    lable_list, p1, p2, p1_index, p2_index = list(map(list, zip(*batch)))
    return torch.tensor(lable_list), p1, p2, torch.tensor(p1_index,dtype=torch.long), torch.tensor(p2_index, dtype=torch.long)


def collate_pyg(data):
    pro1_batch = Batch.from_data_list([item[0] for item in data])
    pro2_batch = Batch.from_data_list([item[1] for item in data])

    return pro1_batch, pro2_batch


def normalize_adj(edge_index, edge_weight=None, num_nodes=None):
    edge_index, edge_weight = utils.remove_self_loops(edge_index, edge_weight)
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1),
                                 device=edge_index.device)
    num_nodes = utils.num_nodes.maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv = 1.0 / deg
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)
    edge_weight = deg_inv[row] * edge_weight
    return utils.to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes=num_nodes)


def compute_pe(graph,pos_enc_dim):
    W0 = normalize_adj(graph.edge_index, num_nodes=graph.c_size).tocsc()
    W = W0
    vector = torch.zeros((graph.c_size, pos_enc_dim))
    vector[:, 0] = torch.from_numpy(W0.diagonal())
    for i in range(pos_enc_dim - 1):
        W = W.dot(W0)
        vector[:, i + 1] = torch.from_numpy(W.diagonal())
    return vector.float()

def padding_seq(pid,dataset_name):
    esm_feature = np.load(f'/media/ST-18T/xiangpeng/SSPPI/data/{dataset_name}/pretrained-emb-esm-2/{pid}.npy')
    # esm_feature = np.load(f'/media/ST-18T/xiangpeng/my_PPI/data/pretrained-emb-esm-2/{pid}.npy') #yeast
    if esm_feature.shape[0] < 1200:  # and esm_feature.shape[0]>500:
        pass
    else:
        esm_feature = esm_feature[:1200, ]
    esm_feature = esm_feature[:1200]
    seq_feature = np.zeros((1200, 1280), dtype=np.float32)
    for idx, feature in enumerate(esm_feature):
        seq_feature[idx] = feature
    seq_feature = torch.Tensor(seq_feature)
    return seq_feature

def read_graph(pid,datasetname):
    with open(f'./data/{datasetname}/graph_data/{pid}.pkl', 'rb') as file:
        c_size, features, edge_index, rwpe, edge_index_high = pkl.load(file)
    return  c_size, features, edge_index, rwpe, edge_index_high

def read_seq(pid,datasetname):
    with open(f'./data/{datasetname}/seq_data/{pid}.pkl', 'rb') as file:
        seq_features = pkl.load(file)
    return  seq_features

class PygDataset(Dataset):
    def __init__(self, dataset_name = 'yeast', pns = None, map_dict = None):
        super(PygDataset,self).__init__()
        self.pns = pns
        self.map_dict = map_dict
        self.dt_name = dataset_name

    def __len__(self):
        return len(self.pns)

    def __getitem__(self, index):
        pid1, pid2, label = self.pns[index]
        pid1_index = self.map_dict[pid1]
        pid2_index = self.map_dict[pid2]
        c_size1, features1, edge_index1, rwpe1, edge_index_high1 = read_graph(pid1,self.dt_name)
        c_size2, features2, edge_index2, rwpe2, edge_index_high2 = read_graph(pid2, self.dt_name)
        seq_data1 = read_seq(pid1, self.dt_name)
        seq_data2 = read_seq(pid2, self.dt_name)
        GCNData1 = DATA.Data(x=torch.Tensor(features1), edge_index=torch.LongTensor(edge_index1),y=torch.FloatTensor([label]), rwpe = rwpe1, edge_index_high = torch.LongTensor(edge_index_high1),map_id = torch.LongTensor([pid1_index]), seq_data = seq_data1.unsqueeze(0))
        GCNData1.__setitem__('c_size', torch.LongTensor([c_size1]))
        GCNData2 = DATA.Data(x=torch.Tensor(features2), edge_index=torch.LongTensor(edge_index2),y=torch.FloatTensor([label]), rwpe = rwpe2, edge_index_high = torch.LongTensor(edge_index_high2), map_id = torch.LongTensor([pid2_index]), seq_data = seq_data2.unsqueeze(0))
        GCNData2.__setitem__('c_size', torch.LongTensor([c_size2]))

        return GCNData1, GCNData2