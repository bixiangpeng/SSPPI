import pandas as pd
from collections import Counter
import numpy as np
import pickle as pkl
import torch
import esm
from tqdm import tqdm
import torch_geometric.utils as utils
from torch_scatter import scatter_add
from torch_geometric.utils import degree, to_dense_adj

""""初步处理得到Gene name并形成相应的数据集"""
###One-Core
# protein_name = []
# protein_name.append('CD9')
# with open('one-core/CD9.txt','r') as file:
#     lines = file.readlines()
#     for line in lines:
#         line = line.strip().split(':')
#         protein_name.append(line[1])
# with open('one-core/gene_name.txt','w') as file:
#     for item in protein_name:
#         file.write(item)
#         file.write('\n')
####Crossover
# protein_nameA = []
# with open('crossover/Wnt_proteinA.txt','r') as file:
#     lines = file.readlines()
#     for line in lines:
#         if line.startswith('>'):
#             line = line.lstrip('>').rstrip()
#             protein_nameA.append(line)
#
# protein_nameB = []
# with open('crossover/Wnt_proteinB.txt','r') as file:
#     lines = file.readlines()
#     for line in lines:
#         if line.startswith('>'):
#             line = line.lstrip('>').rstrip()
#             protein_nameB.append(line)
#
# dataset = {
#         'proteinA': protein_nameA,
#         'proteinB': protein_nameB,
#         'label': [1]*len(protein_nameA),
# }
# df_dataset = pd.DataFrame(dataset)
# df_dataset.to_csv(f'/media/ST-18T/xiangpeng/SSPPI/data/cancer/crossover/test_dataset.csv',index=False)
#
# protein_name = protein_nameA + protein_nameB
# protein_name = list(set(protein_name))
# print(len(protein_name))
# with open('crossover/gene_name.txt','w') as file:
#     for item in protein_name:
#         file.write(item)
#         file.write('\n')

'''进一步处理根据Gene name从uniprot下载的TSV'''
####One-core
# tsv_file_path = 'one-core/idmapping_reviewed_true_AND_model_organ_2024_03_02.tsv'
# df = pd.read_csv(tsv_file_path, sep='\t')
# df.to_csv('one-core/final_onecore.csv',index=None)

####Crossover
# tsv_file_path = 'crossover/idmapping_reviewed_true_AND_model_organ_2024_02_29.tsv'
# df = pd.read_csv(tsv_file_path, sep='\t')
# From = list(df['From'])
# element_count = Counter(From)
# duplication = []
# for element, count in element_count.items():
#     if count > 1:
#         duplication.append(element)
#         print(f"{element} 重复 {count} 次")
# ###去除掉重复匹配的值
# df = df[~df['From'].isin(duplication)]
# ###合并上手工匹配的值
# not_match = {'PROCN':'Q6P2Q9','RAC4':'Q67VP4','RSTS':'Q5U248'}
# redundant_match = {'DVL1':'O14640','TAK1':'P49116','TBL1':'O60907','APC2':'Q9UJX6','PRKACA':'P17612'}
# other_entry = {}
# for key,value in not_match.items():
#     other_entry[key] = value
# for key,value in redundant_match.items():
#     other_entry[key] = value
# tsv_file_path = 'crossover/other_entry.tsv'
# df2 = pd.read_csv(tsv_file_path, sep='\t')
# ###输出最终文件
# df_total = pd.concat([df, df2], axis=0)
# df_total.to_csv('crossover/final_crossover.csv',index=None)



'''处理完后的验证以及生成对应的从alphafold下载蛋白质结构的文件'''
####one-core
# df = pd.read_csv('one-core/final_onecore.csv')
# uniprotID_list = list(df['Entry'])
# with open('one-core/AlphaFold2_PDB/ID_list.txt','w') as file:
#     for i in uniprotID_list:
#         file.write(i)
#         file.write('\n')

####Crossover
# origin_entry = []
# with open('crossover/gene_name.txt','r') as file:
#     lines = file.readlines()
#     for line in lines:
#         origin_entry.append(line.strip())
#
# df = pd.read_csv('crossover/final_crossover.csv')
# processed_entry = list(df['From'])
# for item in origin_entry:
#     if item not in processed_entry:
#         print(item)
# uniprotID_list = list(df['Entry'])
# with open('crossover/AlphaFold2_PDB/ID_list.txt','w') as file:
#     for i in uniprotID_list:
#         file.write(i)
#         file.write('\n')
'''根据PDB文件生成contact map'''
# from Bio import PDB
# import os
# from tqdm import tqdm
# import numpy
# import warnings
# def calc_residue_dist(residue_one, residue_two) :
#     """Returns the C-alpha distance between two residues"""
#     diff_vector  = residue_one["CA"].coord - residue_two["CA"].coord
#     return numpy.sqrt(numpy.sum(diff_vector * diff_vector))
# def calc_dist_matrix(chain_one, chain_two) :
#     """Returns a matrix of C-alpha distances between two chains"""
#     answer = numpy.zeros((len(chain_one), len(chain_two)), float)
#     for row, residue_one in enumerate(chain_one) :
#         for col, residue_two in enumerate(chain_two) :
#             answer[row, col] = calc_residue_dist(residue_one, residue_two)
#     return answer
#
# df = pd.read_csv('one-core/final_onecore.csv')
# P_ID = list(df['Entry'])
# Sequence = list(df['Sequence'])
# Item = dict(zip(P_ID,Sequence))
#
# warnings.filterwarnings('ignore')
# path = '/media/ST-18T/xiangpeng/SSPPI/data/cancer/one-core/AlphaFold2_PDB/PDB/'
# pdb_files = os.listdir(path)
# for pdb_id_ in tqdm(pdb_files):
#     pdb_id = pdb_id_[:-4]
#     p = PDB.PDBParser()
#     structure_id = pdb_id
#     filename = path + f'{pdb_id}.pdb'
#     try:
#         structure = p.get_structure(structure_id, filename)
#     except:
#         print(pdb_id)
#         pass
#     model = structure[0]
#     dist_matrix = calc_dist_matrix(model['A'], model['A'])
#     contact_map = (dist_matrix<8.0)+0
#     assert len(Item[pdb_id]) == contact_map.shape[0]
#     numpy.save('/media/ST-18T/xiangpeng/SSPPI/data/cancer/one-core/contact_map_8.0/'+pdb_id+'.npy',contact_map)

''''根据序列生成相应的ESM表征'''


# Load ESM-2 model
# model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
# batch_converter = alphabet.get_batch_converter()
# model.eval()  # disables dropout for deterministic results
#
# df = pd.read_csv('one-core/final_onecore.csv')
# P_ID = list(df['Entry'])
# Sequence = list(df['Sequence'])
# Item = dict(zip(P_ID,Sequence))
#
# esm_feature_dict = {}
# for pid, seq in tqdm(Item.items()):
#     data = [(pid,str(seq))]
#     batch_labels, batch_strs, batch_tokens = batch_converter(data)
#     batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
#
#     # Extract per-residue representations (on CPU)
#     with torch.no_grad():
#         results = model(batch_tokens, repr_layers=[33], return_contacts=False)
#     token_representations = results["representations"][33]
#
#     token_representations = token_representations.squeeze(0)[1:-1]
#     assert len(seq) == token_representations.shape[0]
#     np.save(f'/media/ST-18T/xiangpeng/SSPPI/data/cancer/one-core/pretrained-emb-esm-2/{pid}.npy',token_representations.numpy())


'''合并contact_map与ESM表征'''
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


def compute_pe(edge_index, c_size, pos_enc_dim):
    W0 = normalize_adj(edge_index, num_nodes=c_size).tocsc()
    W = W0
    vector = torch.zeros((c_size, pos_enc_dim))
    vector[:, 0] = torch.from_numpy(W0.diagonal())
    for i in range(pos_enc_dim - 1):
        W = W.dot(W0)
        vector[:, i + 1] = torch.from_numpy(W.diagonal())
    return vector.float()
df = pd.read_csv('one-core/final_onecore.csv')
P_ID = list(df['Entry'])
Sequence = list(df['Sequence'])
Item = dict(zip(P_ID,Sequence))

for pid, seq in tqdm(Item.items()):
    contact_map = np.load(f'/media/ST-18T/xiangpeng/SSPPI/data/cancer/one-core/contact_map_8.0/{pid}.npy')
    feature = np.load(f'/media/ST-18T/xiangpeng/SSPPI/data/cancer/one-core/pretrained-emb-esm-2/{pid}.npy')
    edge_index = np.argwhere(contact_map == 1).transpose(1, 0)
    edge_index_ = torch.LongTensor(edge_index).to('cuda:0')
    adj_matrix_np = to_dense_adj(edge_index_)[0].cpu().numpy()
    np.fill_diagonal(adj_matrix_np, 0)
    adj_matrix_tensor = torch.Tensor(adj_matrix_np).to('cuda:0')
    adj_matrix_high = torch.matmul(adj_matrix_tensor, adj_matrix_tensor).cpu().numpy()
    np.fill_diagonal(adj_matrix_high, 0)
    adj_matrix_high = (adj_matrix_high >= 1) + 0
    edge_index_high = np.argwhere(adj_matrix_high == 1).transpose()

    c_size = len(seq)
    rwpe = compute_pe(edge_index, c_size, 20)

    esm_feature = feature
    seq_feature = np.zeros((1200, 1280), dtype=np.float32)
    if esm_feature.shape[0] < 1200:  # and esm_feature.shape[0]>500:
        pass
    else:
        esm_feature = esm_feature[:1200, ]
    for idx, feature_temp in enumerate(esm_feature):
        seq_feature[idx] = feature_temp
    seq_feature = torch.Tensor(seq_feature)

    with open(f'/media/ST-18T/xiangpeng/SSPPI/data/cancer/one-core/graph_data/{pid}.pkl', 'wb') as file:
        pkl.dump((c_size, torch.Tensor(feature), edge_index, rwpe, edge_index_high), file)

    with open(f'/media/ST-18T/xiangpeng/SSPPI/data/cancer/one-core/seq_data/{pid}.pkl', 'wb') as file:
        pkl.dump(seq_feature, file)



