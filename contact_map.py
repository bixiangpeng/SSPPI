# -*- coding:utf-8 -*-
"""
作者：毕宸浩
日期：2023年07月24日
"""
# import Bio.PDB
import numpy as np
import scipy.sparse as sp
import os
from tqdm import tqdm
import torch
import pickle as pkl
from sklearn.decomposition import PCA
import torch_geometric.utils as utils
from torch_scatter import scatter_add
# import Bio
from multiprocessing import Pool
from torch_geometric.utils import degree, to_dense_adj

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

aa_codes = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
        'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'LYS': 'K',
        'ILE': 'I', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
        'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
        'THR': 'T', 'VAL': 'V', 'TYR': 'Y', 'TRP': 'W',
    }

def get_center_atom(residue):
    if residue.has_id('CA'):
        c_atom = 'CA'
    elif residue.has_id('N'):
        c_atom = 'N'
    elif residue.has_id('C'):
        c_atom = 'C'
    elif residue.has_id('O'):
        c_atom = 'O'
    elif residue.has_id('CB'):
        c_atom = 'CB'
    elif residue.has_id('CD'):
        c_atom = 'CD'
    else:
        c_atom = 'CG'
    return c_atom

def calc_residue_dist(residue_one, residue_two) :
    """Returns the C-alpha distance between two residues"""

    c_atom1 = get_center_atom(residue_one)
    c_atom2 = get_center_atom(residue_two)
    diff_vector  = residue_one[c_atom1].coord - residue_two[c_atom2].coord
    return np.sqrt(np.sum(diff_vector * diff_vector))

def calc_dist_matrix(chain_one, chain_two) :
    """Returns a matrix of C-alpha distances between two chains"""
    residue_seq = ''
    coord_list = []
    residue_len = 0
    for row, residue_one in enumerate(chain_one):
        hetfield = residue_one.get_id()[0]
        hetname = residue_one.get_resname()
        if hetfield == " " and hetname in aa_codes.keys():
            atom_ = get_center_atom(residue_one)
            coord_ = residue_one[atom_].coord
            coord_list.append(coord_)
            residue_len = residue_len + 1
            residue_seq = residue_seq + aa_codes[hetname]
    dist_matrix = np.zeros((residue_len, residue_len), dtype=float)

    x = -1
    for residue_one in chain_one:
        y = -1
        hetfield1 = residue_one.get_id()[0]
        hetname1 = residue_one.get_resname()
        if hetfield1 == ' ' and hetname1 in aa_codes.keys():
            x = x + 1
            for residue_two in chain_two:
                hetfield2 = residue_two.get_id()[0]
                hetname2 = residue_two.get_resname()
                if hetfield2 == ' ' and hetname2 in aa_codes.keys():
                    y = y + 1
                    dist_matrix[x, y]= calc_residue_dist(residue_one, residue_two)
    coord_array = np.array(coord_list)
    for i in range(residue_len):
        dist_matrix[i,i] = 100
    return dist_matrix,residue_seq,coord_array


# def calc_contact_map(pdb_id,chain_id):
#     pdb_path = pdb_id + '.pdb'
#     structure = Bio.PDB.PDBParser().get_structure(pdb_id, pdb_path)
#     model = structure[0]
#     dist_matrix,res_seq,coord_array = calc_dist_matrix(model[chain_id], model[chain_id])
#     contact_map = (dist_matrix < 8.0).astype(int)
#     #print('contact map shape:',contact_map.shape)
#     return contact_map,res_seq,coord_array


'''generate data file, including (contact_map, sequence, and coord)'''
# def generate_data(pid_list,path):
#     '''读取可获得的PDB文件'''
#
#     for pid in tqdm(pid_list):
#         contact_map, sequence, coord_array = calc_contact_map(path + pid, 'A')
#         np.savez(f'/media/ST-18T/xiangpeng/EGNNPPI/data/yeast/contact/{pid}.npz', seq=sequence, contact=contact_map, coord=coord_array)

    # for pid in tqdm(pid_list):
    #     try:
    #         contact_map,sequence,coord_array = calc_contact_map(path+pid,'A')
    #         np.savez(f'/media/ST-18T/xiangpeng/EGNNPPI/data/yeast/contact/{pid}.npz',seq=sequence,contact=contact_map,coord=coord_array)
    #     except:
    #         print(pid)




def merge_data(pid_list):
    contact_dict = {}
    seq_dict = {}
    coord_dict = {}
    graph_dict = {}
    for pid in tqdm(pid_list):
        with np.load(f'/media/ST-18T/xiangpeng/EGNNPPI/data/yeast/contact/{pid}.npz') as f:
            seq = f['seq']
            contact_map = f['contact']
            coord = f['coord']
            contact_dict[pid] = contact_map
            seq_dict[pid] = seq
            coord_dict[pid] = coord
            f.close()
    with open('/media/ST-18T/xiangpeng/EGNNPPI/data/yeast/contact_dict.pkl','wb') as file:
        pkl.dump(contact_dict,file)
    with open('/media/ST-18T/xiangpeng/EGNNPPI/data/yeast/seq_dict.pkl','wb') as file:
        pkl.dump(seq_dict,file)
    with open('/media/ST-18T/xiangpeng/EGNNPPI/data/yeast/coord_dict.pkl', 'wb') as file:
        pkl.dump(coord_dict, file)


def generate_and_merge_graph(pid_list, dataset):
    graph_dict = {}
    seq_dict = {}
    # with open('/media/ST-18T/xiangpeng/EGNNPPI/data/yeast/seq_feature_dict.pkl','rb') as file:
    #     seq_feature_dict = pkl.load(file)
    # with open('/media/ST-18T/xiangpeng/EGNNPPI/data/yeast/esm_feature_dict.pkl','rb') as file:
    #     esm_feature_dict = pkl.load(file)
    # seqvec_feat_dict = np.load('/media/ST-18T/xiangpeng/EGNNPPI/data/yeast/embeddings.npz')
    # for pid in tqdm(pid_list):
    #     with np.load(f'/media/ST-18T/xiangpeng/EGNNPPI/data/yeast/contact/{pid}.npz') as f:
    #         seq = f['seq']
    #         contact_map = f['contact']
    #         coord = f['coord']
    #         contact_map = np.add(contact_map, np.eye(contact_map.shape[0])).astype(int)
    #         c_size = contact_map.shape[0]
    #         contact_map = np.argwhere(contact_map == 1).transpose(1, 0)
    #
    #         # '''法一'''
    #         # # edge_index = np.argwhere(contact_map == 1).transpose()
    #         # # src_ids = torch.tensor(edge_index[0])
    #         # # target_ids = torch.tensor(edge_index[1])
    #         # # g = dgl.graph((src_ids, target_ids), num_nodes=contact_map.shape[0], idtype=torch.int32)
    #         # '''法二'''
    #         # sp_mat = sp.coo_matrix(contact_map)
    #         # g = dgl.from_scipy(sp_mat,idtype=torch.long)
    #         # # g.ndata['feature'] = torch.tensor(seq_feature_dict[pid],dtype=torch.float)
    #         # # g.ndata['feature'] = esm_feature_dict[pid][1:-1]
    #         # g.ndata['feature'] = torch.tensor(seq_feature_dict[pid],dtype=torch.float)
    #         # g.ndata['coord'] = torch.tensor(coord,dtype=torch.float)
    #         features = seqvec_feat_dict[pid]
    #         # pca = PCA(n_components=32)
    #         # new_features = pca.fit_transform(features)
    #         graph_dict[pid] = (c_size, features, contact_map)
    #         seq_dict[pid] = str(seq)
    #         f.close()
    num_processes = 30
    pool = Pool(processes=num_processes)
    with tqdm(total=len(files)) as pbar:
        for i, _ in enumerate(pool.imap_unordered(generate, pid_list)):
            pbar.update()
    pool.close()
    pool.join()
###全部读入内存
    #     graph_dict[pid] = (c_size, esm_feature, edge_index, rwpe)
    # with open(f'/media/ST-18T/xiangpeng/SSPPI/data/{dataset}/graph_dict_8.0_PE.pkl','wb') as file:
    #     pkl.dump(graph_dict,file)


    # with open('/media/ST-18T/xiangpeng/EGNNPPI/data/yeast/graph_dict_with_esmfeature.pkl','wb') as file:
    #     pkl.dump(graph_dict,file)
    # with open('/media/ST-18T/xiangpeng/EGNNPPI/data/yeast/seq_dict.pkl','wb') as file:
    #     pkl.dump(seq_dict,file)

def generate(pid):
    esm_feature = np.load(f'/media/ST-18T/xiangpeng/SSPPI/data/{dataset}/pretrained-emb-esm-2/{pid}.npy')
    contact_map = np.load(f'/media/ST-18T/xiangpeng/SSPPI/data/{dataset}/contact_map_8.0/{pid}.npy')
    edge_index = np.argwhere(contact_map == 1).transpose(1, 0)
    c_size = contact_map.shape[0]
    rwpe = compute_pe(edge_index, c_size, 20)
    edge_index_high = compute_edgeindex_high(edge_index)
    assert esm_feature.shape[0] == contact_map.shape[0]
    ###分文件读写
    graph_dict = (c_size, esm_feature, edge_index, rwpe, edge_index_high)
    with open(f'/media/ST-18T/xiangpeng/SSPPI/data/{dataset}/graph_data/{pid}.pkl', 'wb') as file:
        pkl.dump(graph_dict, file)



def main(pid_list, dataset):
    # generate_data(pid_list,path)
    # merge_data(pid_list)
    generate_and_merge_graph(pid_list, dataset)

if __name__ == '__main__':

    ###(1) for pdb collected from AlphaFold2 DB
    # path = '/media/ST-18T/xiangpeng/EGNNPPI/data/yeast/pdb/'
    # pid_list = []
    # files = os.listdir(path)
    # for file in files:
    #     pid_list.append(file.strip().split('.')[0])

    ###(2) for missing pdb predicted from AlphaFold2
    # path = '/media/ST-18T/xiangpeng/EGNNPPI/data/yeast/missing_pdb/Alphafold2/'
    # pid_list = []
    # files = os.listdir(path)
    # for file in files:
    #     pid_list.append(file.strip().split('.')[0])
    # main(pid_list, path)

    ###(3) for missing pdb collected from RCSPDB
    # path = '/media/ST-18T/xiangpeng/EGNNPPI/data/yeast/missing_pdb/RCPDB/'
    # pid_list = []
    # files = os.listdir(path)
    # for file in files:
    #     pid_list.append(file.strip().split('.')[0])
    # main(pid_list,path)

    ###(4) for generate_and_merge_graph
    dataset = 'multi_class'
    # dataset = 'multi_species'
    path = f'/media/ST-18T/xiangpeng/SSPPI/data/{dataset}/contact_map_8.0/'
    pid_list = []
    files = os.listdir(path)
    for file in files:
        pid_list.append(file[:-4])

    # index_map_dict = {}
    # for i, index in enumerate(pid_list):
    #     index_map_dict[index] = i
    # with open(f'/media/ST-18T/xiangpeng/SSPPI/data/{dataset}/index_map_dict.pkl', 'wb') as file:
    #     pkl.dump(index_map_dict, file)
    main(pid_list, dataset)
