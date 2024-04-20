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

class STR(nn.Module):
    def __init__(self,input_dim = 1280, hidden_dim = 64, output_dim = 1 ): #多分类的输出维度为7
        super(STR, self).__init__()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.hidden_dim = hidden_dim
        self.CT1 = Convormer(input_dim = input_dim, hidden_dim = hidden_dim, kernel_size = [3, 5, 7], patch_size = 25)

        ###Bilinear Attention
        self.bilinear_attention = BilinearAttention(hidden_dim, num_heads=8)

        self.fc1 = nn.Linear(hidden_dim * 2, 512)
        self.fc2 = nn.Linear(512, 256)
        self.out = nn.Linear(256, output_dim)


    def forward(self,  pro1 , pro2,device):

        '''ConvFormer'''
        seq1 = self.CT1(pro1)
        seq2 = self.CT1(pro2)



        '''Cross-protein Fusion'''
        hc, _, _ = self.bilinear_attention(seq1, seq2)

        hc = self.fc1(hc)
        hc = self.relu(hc)
        hc = self.dropout(hc)
        hc = self.fc2(hc)
        hc = self.relu(hc)
        hc = self.dropout(hc)
        out = self.out(hc)
        return out, torch.tensor(0, device= device), torch.tensor(0, device= device)




