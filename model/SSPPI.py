# -*- coding:utf-8 -*-
"""
作者：毕宸浩
日期：2023年11月20日
"""
import torch.nn as nn
import torch
from torch_geometric.nn import global_mean_pool
from contrast import Contrast
from .Graphormer import Graphormer
from .Convormer import Convormer
from .utils import SA_block, my_MultiHeadAttention, BilinearAttention
from torch_geometric.utils import to_dense_batch

class SSPPI(nn.Module):
    def __init__(self,input_dim = 1280, hidden_dim = 64, output_dim = 1 ):
        super(SSPPI, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.constrast = Contrast(hidden_dim, 0.5, ['str', 'seq'])
        self.hidden_dim = hidden_dim
        self.pre_transform = nn.Linear(input_dim,hidden_dim)
        self.pe_embedding = nn.Linear(20, hidden_dim)

        self.GT1 = Graphormer(hidden_dim, n_heads = 1)
        self.GT2 = Graphormer(hidden_dim, n_heads = 1)
        self.CT1 = Convormer(input_dim = input_dim, hidden_dim = hidden_dim, kernel_size = [3, 5, 7], patch_size = 25)

        self.self_attn = SA_block( layer=1, hidden_dim=hidden_dim, num_heads=8, dropout=0, batch_first=True)
        self.cross_attn1 = my_MultiHeadAttention(hidden_dim = hidden_dim, num_heads = 8,dropout = 0, batch_first = True)
        self.cross_attn2 = my_MultiHeadAttention(hidden_dim =hidden_dim, num_heads= 8, dropout = 0, batch_first = True)

        ###Bilinear Attention
        self.bilinear_attention = BilinearAttention(hidden_dim, num_heads=8)
        self.fc1 = nn.Linear(hidden_dim * 2, 512)
        self.fc2 = nn.Linear(512, 256)
        self.out = nn.Linear(256, output_dim)

    def forward(self,  pro1 , pro2,device):
        x1, edge_index1, edge_index_high1, rwpe1, batch1, p1_index, c_size1 = pro1.x, pro1.edge_index, pro1.edge_index_high, pro1.rwpe, pro1.batch, pro1.map_id, pro1.c_size
        x1 = self.pre_transform(x1)
        rwpe1 = self.pe_embedding(rwpe1)
        x1 = x1 + rwpe1

        x2, edge_index2, edge_index_high2, rwpe2, batch2, p2_index, c_size2 = pro2.x, pro2.edge_index, pro2.edge_index_high, pro2.rwpe, pro2.batch, pro2.map_id, pro2.c_size
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

        '''Cross-modality Enhancer'''
        x1_c = global_mean_pool(x1, batch1)
        x2_c = global_mean_pool(x2, batch2)

        seq1_c = torch.mean(seq1, 1 )
        seq2_c = torch.mean(seq2, 1 )

        cl_loss = self.constrast(x1_c, x2_c, seq1_c, seq2_c, p1_index, p2_index, device)

        seq1 = self.self_attn(seq1, attn_mask=None, key_padding_mask=None)
        seq2 = self.self_attn(seq2, attn_mask = None, key_padding_mask = None)
        x1_dense, mask1 = to_dense_batch(x1, batch1)
        x2_dense, mask2 = to_dense_batch(x2, batch2)
        seq1_ = self.cross_attn1(seq1, x1_dense, x1_dense, attn_mask=None, key_padding_mask = ~mask1) #[32,130,64] 序列增强的结构
        seq2_ = self.cross_attn1(seq2, x2_dense, x2_dense, attn_mask=None, key_padding_mask = ~mask2)

        '''Cross-protein Fusion'''
        hc, _, _ = self.bilinear_attention(seq1_, seq2_)

        hc = self.fc1(hc)
        hc = self.relu(hc)
        hc = self.dropout(hc)
        hc = self.fc2(hc)
        hc = self.relu(hc)
        hc = self.dropout(hc)
        out = self.out(hc)
        return out, cl_loss




