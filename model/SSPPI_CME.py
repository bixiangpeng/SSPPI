# -*- coding:utf-8 -*-
"""
作者：毕宸浩
日期：2023年11月20日
"""
import torch.nn as nn
import torch
from torch_geometric.nn import global_add_pool, global_mean_pool, GCNConv,GATConv,SuperGATConv
from contrast import Contrast
from .utils import FFN, PositionalEncoding
from .test_GT import Graphormer
from .test_CT import Convormer
from .utils import SA_block, my_MultiHeadAttention
from torch_geometric.utils import to_dense_batch
from .bilinear_attention import BilinearAttention

class CME(nn.Module):
    def __init__(self,input_dim = 1280, hidden_dim = 64, output_dim = 1 ): #多分类的输出维度为7
        super(CME, self).__init__()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.hidden_dim = hidden_dim

        self.pre_transform = nn.Linear(input_dim,hidden_dim)
        self.pe_embedding = nn.Linear(20, hidden_dim)

        self.GT1 = Graphormer(hidden_dim, n_heads = 1)
        self.GT2 = Graphormer(hidden_dim, n_heads = 1)

        self.CT1 = Convormer(input_dim = input_dim, hidden_dim = hidden_dim, kernel_size = [3, 5, 7], patch_size = 25)

        self.self_attn = SA_block( layer=1, hidden_dim=hidden_dim, num_heads=8, dropout=0, batch_first=True)
        self.align_fc = nn.Linear(hidden_dim * 2, hidden_dim)

        ###Bilinear Attention
        self.bilinear_attention = BilinearAttention(hidden_dim, num_heads=8)

        self.fc1 = nn.Linear(hidden_dim * 2, 512)
        self.fc2 = nn.Linear(512, 256)
        self.out = nn.Linear(256, output_dim)


    def forward(self,  pro1 , pro2,device):

        x1,edge_index1,edge_index_high1, rwpe1,batch1, p1_index, c_size1 = pro1.x, pro1.edge_index, pro1.edge_index_high, pro1.rwpe, pro1.batch, pro1.ppi_id,pro1.c_size
        x1 = self.pre_transform(x1)
        rwpe1 = self.pe_embedding(rwpe1)
        x1 = x1 + rwpe1

        x2, edge_index2, edge_index_high2, rwpe2,batch2, p2_index, c_size2 = pro2.x, pro2.edge_index, pro2.edge_index_high, pro2.rwpe, pro2.batch, pro2.ppi_id, pro2.c_size
        x2 = self.pre_transform(x2)
        rwpe2 = self.pe_embedding(rwpe2)
        x2 = x2 + rwpe2

        '''GraphFormer'''
        x1 = self.GT1(x1, edge_index1, edge_index_high1, c_size1, batch1, device)
        x1 = self.GT2(x1, edge_index1, edge_index_high1, c_size1, batch1, device)
        x2 = self.GT1(x2, edge_index2, edge_index_high2, c_size2, batch2, device)
        x2 = self.GT2(x2, edge_index2, edge_index_high2, c_size2, batch2, device)

        '''ConvFormer'''
        seq1 = self.CT1(pro1)
        seq2 = self.CT1(pro2)

        '''Feature combination'''
        x1_c = global_mean_pool(x1, batch1).unsqueeze(1).expand(seq1.shape[0], seq1.shape[1], seq1.shape[2])
        x2_c = global_mean_pool(x2, batch2).unsqueeze(1).expand(seq2.shape[0], seq2.shape[1], seq2.shape[2])
        seq1_ = torch.cat((torch.add(seq1, x1_c), torch.sub(seq1, x1_c)), -1) # feature combination
        seq2_ = torch.cat((torch.add(seq2, x2_c), torch.sub(seq2, x2_c)), -1)
        seq1_ = self.relu(self.align_fc(seq1_))
        seq2_ = self.relu(self.align_fc(seq2_))

        '''Cross-protein Fusion'''
        hc, _, _ = self.bilinear_attention(seq1_, seq2_)

        hc = self.fc1(hc)
        hc = self.relu(hc)
        hc = self.dropout(hc)
        hc = self.fc2(hc)
        hc = self.relu(hc)
        hc = self.dropout(hc)
        out = self.out(hc)
        return out, torch.tensor(0, device= device), torch.tensor(0, device= device)



