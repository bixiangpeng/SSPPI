# import torch.nn as nn
# import torch
# from torch_geometric.utils import batched_negative_sampling,  negative_sampling
# from torch_geometric.utils.dropout import dropout_edge
# from torch_geometric.nn import global_add_pool, global_mean_pool, GCNConv,GATConv
# from torch_geometric.utils import to_dense_batch
# from .utils import MSA, FFN
# import torch.nn.functional as F
#
#
# class Graphormer(nn.Module):
#     def __init__(self, hidden_dim = 64, n_heads = 8):
#         super(Graphormer, self).__init__()
#         self.GCN = GATConv(hidden_dim, hidden_dim,1)
#         self.MHA = MSA( hidden_dim = hidden_dim, n_heads = n_heads )
#         self.norm1 = nn.LayerNorm(hidden_dim)
#         self.norm2 = nn.LayerNorm(hidden_dim)
#         self.dropout = nn.Dropout(0.2)
#         self.relu = nn.ReLU()
#
#     def forward(self, x, edge_index,c_size, batch, device):
#         x_ = self.dropout(self.relu(self.GCN(x, edge_index)))  #图卷积
#         x_ = self.norm1( x + x_ )
#         x_dense, mask = to_dense_batch(x_, batch) # Padding
#
#         attn_mask = (~mask).unsqueeze(1).repeat(1, x_dense.size(1), 1)
#         x_dense, scores = self.MHA(x_dense, x_dense, x_dense, attn_mask = attn_mask) # 多头注意力
#         x_ = self.norm2( x_ + self.dropout(self.relu(x_dense[mask]) ))
#
#         recon_loss = None
#         if self.training:
#             scores = scores[mask]
#             # pos_edge_index = self.positive_sampling(edge_index)
#             pos_edge_index = edge_index
#             count = torch.tensor([torch.sum(c_size[:i]).item() for i in range(0, c_size.size(0))]).to(device)
#             pos_row, pos_col = pos_edge_index
#             pos_index = pos_col - count[batch[pos_col]]
#             pos_scores = scores[pos_row][torch.arange(pos_row.size(0)), pos_index]
#             label = scores.new_zeros(pos_scores.size(0))
#             label[:pos_edge_index.size(1)] = 1.
#             recon_loss = F.binary_cross_entropy_with_logits(pos_scores, label)
#
#         return x_, recon_loss
#
#
#     def positive_sampling(self, edge_index):
#         pos_edge_index, _ = dropout_edge(edge_index, p=0.1, training=self.training)
#         return pos_edge_index
#
#     def negative_sampling(self, edge_index, num_nodes, batch=None):
#         num_neg_samples = None
#         if batch is None:
#             neg_edge_index = negative_sampling(edge_index, num_nodes,num_neg_samples=num_neg_samples)
#         else:
#             neg_edge_index = batched_negative_sampling(edge_index, batch, num_neg_samples=num_neg_samples)
#         return neg_edge_index
#
import torch.nn as nn
import torch
from torch_geometric.utils import batched_negative_sampling,  negative_sampling
from torch_geometric.utils.dropout import dropout_edge
from torch_geometric.nn import global_add_pool, global_mean_pool, GCNConv,GATConv
from torch_geometric.utils import to_dense_batch, to_dense_adj
from .utils import MSA, FFN
import torch.nn.functional as F
from torch_geometric.utils import degree


class Graphormer(nn.Module):
    def __init__(self, hidden_dim = 64, n_heads = 8):
        super(Graphormer, self).__init__()
        self.GCN = GATConv(hidden_dim, hidden_dim,1)
        self.MHA = MSA( hidden_dim = hidden_dim, n_heads = n_heads )
        # self.deg_embedding = nn.Embedding(60, hidden_dim)
        self.edge_embedding = nn.Linear(1, 1)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index,edge_index_high, c_size, batch, device):
        ###度编码没有用啊
        # x_deg = degree(edge_index[0], num_nodes=x.shape[0]).type(torch.LongTensor).to(device)
        # deg_embedding = self.deg_embedding(x_deg)
        # x = x + deg_embedding
        x_dense, mask = to_dense_batch(x, batch)  # Padding

        edge_encoding = to_dense_adj(edge_index, batch, max_num_nodes=x_dense.shape[1])
        edge_encoding_high = to_dense_adj(edge_index_high, batch, max_num_nodes=x_dense.shape[1])
        edge_encoding_new = 0.7 * edge_encoding + 0.3 * edge_encoding_high
        edge_emb = self.edge_embedding(edge_encoding_new.unsqueeze(-1)).squeeze(-1)
        attn_mask = (~mask).unsqueeze(1).repeat(1, x_dense.size(1), 1)
        x_dense, scores = self.MHA(x_dense, x_dense, x_dense, attn_mask=attn_mask, edge_encoding=edge_emb)  # 多头注意力
        x_ = self.norm2(x + self.dropout(self.relu(x_dense[mask])))

        x_ = self.dropout(self.relu(self.GCN(x_, edge_index)))  # 图卷积



        # x_ = self.norm1(x + x_)


        #原始
        # x_ = self.dropout(self.relu(self.GCN(x, edge_index)))  #图卷积
        # x_ = self.norm1( x + x_ )
        #
        # x_dense, mask = to_dense_batch(x_, batch) # Padding
        # edge_encoding = to_dense_adj(edge_index, batch, max_num_nodes= x_dense.shape[1])
        #
        # attn_mask = (~mask).unsqueeze(1).repeat(1, x_dense.size(1), 1)
        # x_dense, scores = self.MHA(x_dense, x_dense, x_dense, attn_mask = attn_mask, edge_encoding = edge_encoding) # 多头注意力
        # x_ = self.norm2( x_ + self.dropout(self.relu(x_dense[mask]) ))

        # recon_loss = None
        # if self.training:
        #     scores = scores[mask]
        #     # pos_edge_index = self.positive_sampling(edge_index)
        #     pos_edge_index = edge_index
        #     count = torch.tensor([torch.sum(c_size[:i]).item() for i in range(0, c_size.size(0))]).to(device)
        #     pos_row, pos_col = pos_edge_index
        #     pos_index = pos_col - count[batch[pos_col]]
        #     pos_scores = scores[pos_row][torch.arange(pos_row.size(0)), pos_index]
        #     label = scores.new_zeros(pos_scores.size(0))
        #     label[:pos_edge_index.size(1)] = 1.
        #     recon_loss = F.binary_cross_entropy_with_logits(pos_scores, label)

        return x_


    def positive_sampling(self, edge_index):
        pos_edge_index, _ = dropout_edge(edge_index, p=0.1, training=self.training)
        return pos_edge_index

    def negative_sampling(self, edge_index, num_nodes, batch=None):
        num_neg_samples = None
        if batch is None:
            neg_edge_index = negative_sampling(edge_index, num_nodes,num_neg_samples=num_neg_samples)
        else:
            neg_edge_index = batched_negative_sampling(edge_index, batch, num_neg_samples=num_neg_samples)
        return neg_edge_index


