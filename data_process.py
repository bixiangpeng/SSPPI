import json
from rdkit import Chem
import numpy as np
import networkx as nx
import pickle
import pandas as pd
from collections import OrderedDict
import argparse
from tqdm import tqdm
import os
from sklearn.decomposition import PCA
import torch
from multiprocessing import Pool

def atom_features(atom):
    # Generating initial atomic descriptors based on atomic properties.
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def dic_normalize(dic):
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0
    return dic

pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y','X']
pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
pro_res_aromatic_table = ['F', 'W', 'Y']
pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
pro_res_acidic_charged_table = ['D', 'E']
pro_res_basic_charged_table = ['H', 'K', 'R']

res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                    'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                    'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}

res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
                 'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
                 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}

res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17,
                 'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13,
                 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}

res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00,
                 'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00,
                 'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}

res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59,
                'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65,
                'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}

res_hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100,
                             'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7,
                             'T': 13, 'V': 79, 'W': 84, 'Y': 49}

res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99,
                             'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5,
                             'T': 13, 'V': 76, 'W': 97, 'Y': 63}

#normalization
res_weight_table = dic_normalize(res_weight_table)
res_pka_table = dic_normalize(res_pka_table)
res_pkb_table = dic_normalize(res_pkb_table)
res_pkx_table = dic_normalize(res_pkx_table)
res_pl_table = dic_normalize(res_pl_table)
res_hydrophobic_ph2_table = dic_normalize(res_hydrophobic_ph2_table)
res_hydrophobic_ph7_table = dic_normalize(res_hydrophobic_ph7_table)


def residue_features(residue):
    if residue not in pro_res_table:
        residue = 'X'
    res_property1 = [1 if residue in pro_res_aliphatic_table else 0, 1 if residue in pro_res_aromatic_table else 0,
                     1 if residue in pro_res_polar_neutral_table else 0,
                     1 if residue in pro_res_acidic_charged_table else 0,
                     1 if residue in pro_res_basic_charged_table else 0]
    res_property2 = [res_weight_table[residue], res_pka_table[residue], res_pkb_table[residue], res_pkx_table[residue],
                     res_pl_table[residue], res_hydrophobic_ph2_table[residue], res_hydrophobic_ph7_table[residue]]
    return np.array(res_property1 + res_property2)

def seq_feature(pro_seq):
    pro_hot = np.zeros((len(pro_seq), len(pro_res_table)))
    pro_property = np.zeros((len(pro_seq), 12))
    for i in range(len(pro_seq)):
        pro_hot[i,] = one_of_k_encoding_unk(pro_seq[i], pro_res_table)
        pro_property[i,] = residue_features(pro_seq[i])
    return np.concatenate((pro_hot, pro_property), axis=1)

def seq_to_graph(pro_id, seq):
    npz_data = np.load(f'/media/ST-18T/xiangpeng/TAGPPI-main/data/yeast/alphafold_cmap/{pro_id}.npz')
    npz_seq = str(npz_data['seq'])
    npz_contact = npz_data['contact']
    try:
        assert seq == npz_seq
        assert len(seq) == len(npz_seq)
    except:
        print(pro_id)
        print('seq:',seq)
        print('npz_seq:',npz_seq)
    edge_index = np.argwhere(npz_contact == 1).transpose(1, 0).astype(int)
    c_size = len(npz_seq)
    features = seq_feature(npz_seq).astype(np.float32)
    return c_size, features, edge_index


'''生成具有普通33维表征的PyG蛋白质图数据'''
# seq_dict = {}
# with open('/media/ST-18T/xiangpeng/TAGPPI-main/data/yeast/dictionary/sequences.fasta') as file:
#     lines = file.readlines()
#
# for i in range(0, len(lines), 2):
#     protein_id = lines[i].strip()[1:]
#     protein_seq = lines[i + 1].strip()
#     seq_dict[protein_id] = protein_seq
#
# seq_graph = {}
# for pro_id, seq in tqdm(seq_dict.items()):
#     g = seq_to_graph(pro_id, seq)
#     seq_graph[pro_id] = g
# print('protein graph is constructed successfully!')
#
# with open('/media/ST-18T/xiangpeng/my_PPI/data/yeast/graph_dict_in_pyg.pkl','wb') as file:
#     pickle.dump(seq_graph,file)

def generate_seq_data(pid):
    esm_feature = np.load(f'/media/ST-18T/xiangpeng/SSPPI/data/{dataset}/pretrained-emb-esm-2/{pid}.npy')
    if esm_feature.shape[0] < 1200:  # and esm_feature.shape[0]>500:
        # count = count + 1
        pass
    else:
        esm_feature = esm_feature[:1200, ]
    esm_feature = esm_feature[:1200]
    seq_features = np.zeros((1200, 1280), dtype=np.float32)
    for idx, feature in enumerate(esm_feature):
        seq_features[idx] = feature
    seq_features = torch.Tensor(seq_features)
    with open(f'/media/ST-18T/xiangpeng/SSPPI/data/{dataset}/seq_data/{pid}.pkl', 'wb') as file:
        pickle.dump(seq_features, file)

dataset = 'multi_class'
'''以ESM2(1200,1280),生成序列数据，统一padding到1200'''
seq_esm_dict = {}
path = f'/media/ST-18T/xiangpeng/SSPPI/data/{dataset}/pretrained-emb-esm-2'
files = os.listdir(path)
pid_list = [file[:-4] for file in files]
count = 0

num_processes = 30
pool = Pool(processes=num_processes)
with tqdm(total=len(files)) as pbar:
    for i, _ in enumerate(pool.imap_unordered(generate_seq_data, pid_list)):
        pbar.update()
pool.close()
pool.join()
# print(f'{count/len(pid_list)*100} %  of proteins were included !')

# with open(f'/media/ST-18T/xiangpeng/SSPPI/data/{dataset}/seq_esm_tensor_dict.pkl', 'wb') as file:
#     pickle.dump(seq_esm_dict, file)