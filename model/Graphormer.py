import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_dense_batch, to_dense_adj
from .utils import MSA

class Graphormer(nn.Module):
    def __init__(self, hidden_dim = 64, n_heads = 8):
        super(Graphormer, self).__init__()
        self.GCN = GATConv(hidden_dim, hidden_dim,1)
        self.MHA = MSA( hidden_dim = hidden_dim, n_heads = n_heads )
        self.edge_embedding = nn.Linear(1, 1)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index,edge_index_high, c_size, batch, device):
        x_dense, mask = to_dense_batch(x, batch)
        edge_encoding = to_dense_adj(edge_index, batch, max_num_nodes=x_dense.shape[1])
        edge_encoding_high = to_dense_adj(edge_index_high, batch, max_num_nodes=x_dense.shape[1])
        edge_encoding_new = 0.7 * edge_encoding + 0.3 * edge_encoding_high
        edge_emb = self.edge_embedding(edge_encoding_new.unsqueeze(-1)).squeeze(-1)
        attn_mask = (~mask).unsqueeze(1).repeat(1, x_dense.size(1), 1)
        x_dense, scores = self.MHA(x_dense, x_dense, x_dense, attn_mask=attn_mask, edge_encoding=edge_emb)
        x_ = self.norm2(x + self.dropout(self.relu(x_dense[mask])))
        x_ = self.dropout(self.relu(self.GCN(x_, edge_index)))
        return x_

