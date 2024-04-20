import torch.nn as nn
import torch
from torch_geometric.utils import batched_negative_sampling,  negative_sampling
from torch_geometric.utils.dropout import dropout_edge
from torch_geometric.nn import global_add_pool, global_mean_pool, GCNConv,GATConv
from torch_geometric.utils import to_dense_batch
from .utils import MSA, FFN
import torch.nn.functional as F


class GT_block(nn.Module):
    def __init__(self, hidden_dim = 64, n_heads = 8):
        super(GT_block, self).__init__()
        # self.neg_sample_ratio = 0.5
        # self.edge_sample_ratio = 0.9
        # self.GCN = GCNConv( hidden_dim, hidden_dim )
        self.GCN = GATConv(hidden_dim, hidden_dim,1)
        self.MHA = MSA( hidden_dim = hidden_dim, n_heads = n_heads )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
    def forward(self, x, edge_index,c_size, batch, device):
        x_ = self.dropout(self.relu(self.GCN(x, edge_index)))  #图卷积
        x_ = self.norm1( x + x_ )
        x_dense, mask = to_dense_batch(x_, batch) # Padding

        attn_mask = (~mask).unsqueeze(1).repeat(1, x_dense.size(1), 1)
        x_dense, scores = self.MHA(x_dense, x_dense, x_dense, attn_mask = attn_mask) # 多头注意力
        x_ = self.norm2( x_ + self.dropout(self.relu(x_dense[mask]) ))# test##############################################

        recon_loss = None
        if self.training:
            scores = scores[mask]
            # pos_edge_index = edge_index
            pos_edge_index = self.positive_sampling(edge_index)
            # neg_edge_index = self.negative_sampling(edge_index, None, batch) # batch现加的，看看还有没有段错误

            count = torch.tensor([torch.sum(c_size[:i]).item() for i in range(0, c_size.size(0))]).to(device)
            pos_row, pos_col = pos_edge_index
            # neg_row, neg_col = neg_edge_index
            pos_index = pos_col - count[batch[pos_col]]
            # neg_index = neg_col - count[batch[neg_col]]

            pos_scores = scores[pos_row][torch.arange(pos_row.size(0)), pos_index]
            # neg_scores = scores[neg_row][torch.arange(neg_row.size(0)), neg_index]

            # scores = torch.cat([pos_scores, neg_scores], dim=0)  # .unsqueeze(-1)
            label = scores.new_zeros(pos_scores.size(0))
            label[:pos_edge_index.size(1)] = 1.

            recon_loss = F.binary_cross_entropy_with_logits(pos_scores, label)
        # if self.training:
        #     scores = scores[mask]
        #     pos_edge_index = self.positive_sampling(edge_index)
        #     neg_edge_index = self.negative_sampling(edge_index, None, batch) # batch现加的，看看还有没有段错误
        #
        #     count = torch.tensor([torch.sum(c_size[:i]).item() for i in range(0, c_size.size(0))]).to(device)
        #     pos_row, pos_col = pos_edge_index
        #     neg_row, neg_col = neg_edge_index
        #     pos_index = pos_col - count[batch[pos_col]]
        #     neg_index = neg_col - count[batch[neg_col]]
        #
        #     pos_scores = scores[pos_row][torch.arange(pos_row.size(0)), pos_index]
        #     neg_scores = scores[neg_row][torch.arange(neg_row.size(0)), neg_index]
        #
        #     scores = torch.cat([pos_scores, neg_scores], dim=0)  # .unsqueeze(-1)
        #     label = scores.new_zeros(scores.size(0))
        #     label[:pos_edge_index.size(1)] = 1.
        #
        #     recon_loss = F.binary_cross_entropy_with_logits(scores, label)

        return x_, recon_loss

        # return recon_loss

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

