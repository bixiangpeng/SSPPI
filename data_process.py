from tqdm import tqdm
from multiprocessing import Pool
from torch_geometric.utils import to_dense_adj
import torch_geometric.utils as utils
from torch_scatter import scatter_add
import numpy as np
import pickle as pkl
import os
import torch

def normalize_adj(edge_index, edge_weight=None, num_nodes=None):
    edge_index, edge_weight = utils.remove_self_loops(edge_index, edge_weight)
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.shape[1])
    num_nodes = utils.num_nodes.maybe_num_nodes(edge_index, num_nodes)
    edge_index = torch.LongTensor(edge_index)
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv = 1.0 / deg
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)
    edge_weight = deg_inv[row] * edge_weight
    return utils.to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes=num_nodes)

def compute_pe(edge_index,c_size, pos_enc_dim):
    W0 = normalize_adj(edge_index, num_nodes=c_size).tocsc()
    W = W0
    vector = torch.zeros((c_size, pos_enc_dim))
    vector[:, 0] = torch.from_numpy(W0.diagonal())
    for i in range(pos_enc_dim - 1):
        W = W.dot(W0)
        vector[:, i + 1] = torch.from_numpy(W.diagonal())
    return vector.float()

def compute_edgeindex_high(edge_index):
    edge_index_ = torch.LongTensor(edge_index).to('cuda:0')
    adj_matrix_np = to_dense_adj(edge_index_)[0].cpu().numpy()
    np.fill_diagonal(adj_matrix_np, 0)
    adj_matrix_tensor = torch.Tensor(adj_matrix_np).to('cuda:0')
    adj_matrix_high = torch.matmul(adj_matrix_tensor, adj_matrix_tensor).cpu().numpy()
    np.fill_diagonal(adj_matrix_high, 0)
    adj_matrix_high = (adj_matrix_high >= 1) + 0
    edge_index_high = np.argwhere(adj_matrix_high == 1).transpose()
    return edge_index_high


def generate_graph_data(pid):
    esm_feature = np.load(f'./data/{dataset}/pretrained-emb-esm-2/{pid}.npy')
    contact_map = np.load(f'./data/{dataset}/contact_map_8.0/{pid}.npy')
    edge_index = np.argwhere(contact_map == 1).transpose(1, 0)
    c_size = contact_map.shape[0]
    rwpe = compute_pe(edge_index, c_size, 20)
    edge_index_high = compute_edgeindex_high(edge_index)
    assert esm_feature.shape[0] == contact_map.shape[0]
    graph_data = (c_size, esm_feature, edge_index, rwpe, edge_index_high)
    with open(f'./data/{dataset}/graph_data/{pid}.pkl', 'wb') as file:
        pkl.dump(graph_data, file)

def generate_seq_data(pid):
    esm_feature = np.load(f'./data/{dataset}/pretrained-emb-esm-2/{pid}.npy')
    if esm_feature.shape[0] < 1200:
        pass
    else:
        esm_feature = esm_feature[:1200, ]
    esm_feature = esm_feature[:1200]
    seq_features = np.zeros((1200, 1280), dtype=np.float32)
    for idx, feature in enumerate(esm_feature):
        seq_features[idx] = feature
    seq_features = torch.Tensor(seq_features)
    with open(f'./data/{dataset}/seq_data/{pid}.pkl', 'wb') as file:
        pkl.dump(seq_features, file)

def generate_data(data_type):
    num_processes = 30
    pool = Pool(processes=num_processes)
    if data_type == 'graph':
        with tqdm(total=len(files)) as pbar:
            for i, _ in enumerate(pool.imap_unordered(generate_graph_data, pid_list)):
                pbar.update()
    else:
        with tqdm(total=len(files)) as pbar:
            for i, _ in enumerate(pool.imap_unordered(generate_seq_data, pid_list)):
                pbar.update()
    pool.close()
    pool.join()

if __name__ == '__main__':
    dataset = 'multi_species'
    read_path = f'./data/{dataset}/pretrained-emb-esm-2'
    files = os.listdir(read_path)
    pid_list = [file[:-4] for file in files]
    generate_data('graph')
    generate_data('seq')




