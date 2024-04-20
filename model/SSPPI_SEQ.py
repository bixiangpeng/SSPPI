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
from .bilinear_attention import BilinearAttention2

class SEQ(nn.Module):
    def __init__(self,input_dim = 1280, hidden_dim = 64, output_dim = 1 ): #多分类的输出维度为7
        super(SEQ, self).__init__()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        self.hidden_dim = hidden_dim
        self.pre_transform = nn.Linear(input_dim,hidden_dim)
        self.pe_embedding = nn.Linear(20, hidden_dim)

        self.GT1 = Graphormer(hidden_dim, n_heads = 1)
        self.GT2 = Graphormer(hidden_dim, n_heads = 1)

        ###Bilinear Attention
        self.bilinear_attention = BilinearAttention2(hidden_dim, num_heads=1)

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


        '''Cross-protein Fusion'''
        hc = self.bilinear_attention(x1,batch1, x2,batch2)

        hc = self.fc1(hc)
        hc = self.relu(hc)
        hc = self.dropout(hc)
        hc = self.fc2(hc)
        hc = self.relu(hc)
        hc = self.dropout(hc)
        out = self.out(hc)
        return out, torch.tensor(0, device= device), torch.tensor(0, device= device)




