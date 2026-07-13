import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv

class DimensionNN_V2(nn.Module):
    def __init__(self, n_in, n_h, n_out, activator):
        super(DimensionNN_V2, self).__init__()
        self.act = activator()
        self.lin_in = nn.Linear(n_in, n_h)
        self.lin_h1 = nn.Linear(n_h, n_h)
        self.lin_out = nn.Linear(n_h, n_out)
        self.sample = []

    def encode(self, x):
        z = self.act(self.lin_in(x))
        z = self.act(self.lin_h1(z))
        return self.lin_out(z)

    def forward(self, x: torch.Tensor):
        self.sample = x
        self.out = F.normalize(self.encode(x.T))
        return self.out

    def dimensional_loss(self):
        return self.out.mean(dim=0).pow(2).mean()

class MLP_encoder(nn.Module):
    def __init__(self, n_in, n_h, activator):
        super(MLP_encoder, self).__init__()
        self.act = activator()
        self.lin1 = nn.Linear(n_in, n_h)
        self.lin2 = nn.Linear(n_h, n_h)

    def encode(self, x, edge_index=None):
        out = self.act(self.lin1(x))
        out = self.act(self.lin2(out))
        return out

    def proj(self, z):
        return self.lin2(self.act(self.lin1(z)))

    def forward(self, x, edge_index=None):
        return self.encode(x)

    def embed(self, x, edge_index=None):
        self.eval()
        return self.encode(x)

class GraphTransformer_encoder(nn.Module):
    def __init__(self, n_in, n_h, activator, heads=3, dropout=0.2):
        super(GraphTransformer_encoder, self).__init__()
        self.transformer_in = TransformerConv(n_in, n_h, heads=heads, dropout=dropout)
        self.transformer_out = TransformerConv(n_h * heads, n_h, heads=heads, dropout=dropout)
        self.act = activator()

    def encode(self, x, edge_index):
        out = self.act(self.transformer_in(x, edge_index))
        out = self.act(self.transformer_out(out, edge_index))
        return out

    def proj(self, z):
        return self.lin_2(self.act(self.lin_1(z)))

    def forward(self, x, edge_index):
        return self.encode(x, edge_index)

    def embed(self, x, edge_index):
        self.eval()
        return self.encode(x, edge_index)

class LSAS(nn.Module):
    def __init__(self, D_NN, MLP, Trans, S_mtd, sample_size):
        super(LSAS, self).__init__()
        self.dnn = D_NN
        self.mlp = MLP
        self.trans = Trans
        self.trans_proj = nn.Linear(3072, 1024)
        self.smtd = S_mtd
        self.sample_size = sample_size
        self.d_sample_matrix = []

    def update_sample(self, x, edge_index, if_rand=False):
        with torch.no_grad():
            self.d_sample_matrix = self.smtd(self.sample_size, x, edge_index, if_rand)

    def forward(self, x, edge_index):
        dimension_sig = self.dnn(self.d_sample_matrix)
        x = self.feature_sig_propagate(x, dimension_sig)
        z_g = self.mlp(x)
        z_t = self.trans(x, edge_index)
        z_t = self.trans_proj(z_t)
        return z_g, z_t

    def embed(self, x, edge_index):
        with torch.no_grad():
            self.eval()
            dimension_sig = self.dnn(self.d_sample_matrix)
            x = self.feature_sig_propagate(x, dimension_sig)
            z_g = self.mlp.embed(x)
            z_t = self.trans.embed(x, edge_index)
            z = torch.cat((z_g, z_t), dim=1)
            return z

    def infonce_loss(self, z_g, z_t, temperature=0.2):
        z_g = F.normalize(z_g, dim=1)
        z_t = F.normalize(z_t, dim=1)
        sim = torch.matmul(z_g, z_t.t()) / temperature
        labels = torch.arange(sim.size(0), device=sim.device)
        loss = F.cross_entropy(sim, labels)
        return loss

    def dim_loss_fn(self):
        return self.dnn.dimensional_loss()

    def feature_sig_propagate(self, x, dimension_sig):
        return F.normalize(x @ dimension_sig)
