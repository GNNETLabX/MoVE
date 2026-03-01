from torch import Tensor
from torch.nn import Module
from typing import Optional, Tuple

from torch_geometric.nn.inits import glorot
from torch import nn
from torch.nn import Parameter
# from torch_geometric.utils import softmax
# from torch_geometric.data import Data
# from torch_geometric.utils.convert import to_networkx
# from torch_scatter import scatter

from torch.nn import MultiheadAttention
import torch.nn.functional as F
import networkx as nx
import numpy as np
import torch
import math
from torch_geometric.nn import GCNConv, GATConv # Import GCNConv from PyG
from torch_geometric.nn import GAE

EPS = 1e-15
MAX_LOGSTD = 2


class InnerProductDecoder(torch.nn.Module):
    r"""The inner product decoder from the `"Variational Graph Auto-Encoders"
    <https://arxiv.org/abs/1611.07308>`_ paper

    .. math::
        \sigma(\mathbf{Z}\mathbf{Z}^{\top})

    where :math:`\mathbf{Z} \in \mathbb{R}^{N \times d}` denotes the latent
    space produced by the encoder."""
    def forward(self, z: Tensor, edge_index: Tensor,
                sigmoid: bool = True) -> Tensor:
        r"""Decodes the latent variables :obj:`z` into edge probabilities for
        the given node-pairs :obj:`edge_index`.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, z: Tensor, sigmoid: bool = True) -> Tensor:
        r"""Decodes the latent variables :obj:`z` into a probabilistic dense
        adjacency matrix.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj


class SparseInputLinear(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(SparseInputLinear, self).__init__()

        weight = np.zeros((inp_dim, out_dim), dtype=np.float32)
        weight = nn.Parameter(torch.from_numpy(weight))
        bias = np.zeros(out_dim, dtype=np.float32)
        bias = nn.Parameter(torch.from_numpy(bias))
        self.inp_dim, self.out_dim = inp_dim, out_dim
        self.weight, self.bias = weight, bias
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return torch.mm(x, self.weight) + self.bias

class RelTemporalEncoding(nn.Module):
    def __init__(self, n_hid, max_len=50, dropout=0.2):
        super(RelTemporalEncoding, self).__init__()

        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_hid, 2) * -(math.log(10000.0) / n_hid))
        emb = nn.Embedding(max_len, n_hid)
        emb.weight.data[:, 0::2] = torch.sin(position * div_term) / math.sqrt(n_hid)
        emb.weight.data[:, 1::2] = torch.cos(position * div_term) / math.sqrt(n_hid)
        emb.requires_grad = False
        self.emb = emb
        self.lin = nn.Linear(n_hid, n_hid)

    def forward(self, x, t):
        return x + self.lin(self.emb(t))


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, z, e):
        x_i = z[e[0]]
        x_j = z[e[1]]
        x = torch.cat([x_i, x_j], dim=1)
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x).squeeze()


class MultiplyPredictor(torch.nn.Module):
    def __init__(self):
        super(MultiplyPredictor, self).__init__()

    def forward(self, z, e):
        x_i = z[e[0]]
        x_j = z[e[1]]
        x = (x_i * x_j).sum(dim=1)
        return torch.sigmoid(x)


class MergeMultiplyPredictor(torch.nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_dim): # Removed output_dim as it's effectively 1 after sum
        super(MergeMultiplyPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim1, hidden_dim)
        self.fc2 = nn.Linear(input_dim2, hidden_dim)
        self.act = nn.ReLU()
    def forward(self, z, e):
        x_i = self.act(z[e[0]])
        x_j = self.act(z[e[1]])
        x_merged = x_i * x_j
        x_summed = x_merged.sum(dim=1)
        return torch.sigmoid(x_summed)

class MergeLayer(nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_dim, output_dim):
        super(MergeLayer, self).__init__()

        super().__init__()
        self.fc1 = nn.Linear(input_dim1 + input_dim2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()

    def forward(self, z, e):
        x_i = z[e[0]]
        x_j = z[e[1]]
        x = torch.cat([x_i, x_j], dim=1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return torch.sigmoid(x).squeeze()

class NodeClf(nn.Module):
    def __init__(self, args):
        super().__init__()
        clf = nn.ModuleList()
        hid_dim = args.nfeat
        clf_layers = args.clf_layers
        num_classes = args.num_classes
        clf.append(nn.Linear(hid_dim, num_classes))
        self.clf = clf

    def forward(self, x):
        for layer in self.clf:
            x = layer(x)
        return x



class GCNEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.in_dim = args.nfeat
        self.dropout = getattr(args, "dropout", 0.4)

        in_channels = self.in_dim
        out_channels = self.in_dim
        hidden_channels = 4 * in_channels
        self.conv_mu = GCNConv(in_channels, out_channels)
        self.conv_logvar = GCNConv(in_channels, out_channels)

        if self.args.dataset == 'aminer':
            self.classifier = NodeClf(args)

        self.feat = Parameter(
            (torch.ones(args.num_nodes, args.nfeat)).to(args.device),
            requires_grad=True
        )
        self.edge_decoder = MultiplyPredictor()

    def forward(self, x_list, edge_index):
        n = x_list.size(0)
        mu = self.conv_mu(x_list, edge_index)
        mu = F.relu(mu)
        mu = F.dropout(mu, p=self.dropout, training=self.training)  # 可选，是否对mu加dropout
        logvar = self.conv_logvar(x_list, edge_index)
        logvar = F.relu(logvar)
        logvar = F.dropout(logvar, p=self.dropout, training=self.training)  # 可选
        return mu, logvar


class Environment_estminator(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int, k: int):

        super().__init__()
        self.k = k

        self.mu_heads = nn.ModuleList([
            nn.Linear(hidden_dim, feat_dim) for _ in range(k)
        ])
        self.logvar_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, feat_dim), nn.Softplus()) for _ in range(k)
        ])
        self.reset_parameters(hidden_dim)

    def reset_parameters(self, hidden_dim):
        stdv = 1. / math.sqrt(hidden_dim)
        for m in self.mu_heads:
            if isinstance(m, nn.Linear):
                m.weight.data.uniform_(-stdv, stdv)
                if m.bias is not None:
                    m.bias.data.zero_()
        for m in self.logvar_heads:
            for layer in m:
                if isinstance(layer, nn.Linear):
                    layer.weight.data.uniform_(-stdv, stdv)
                    if layer.bias is not None:
                        layer.bias.data.zero_()

    def forward(self, mu: torch.Tensor, log_var: torch.Tensor):
        delta_mus = []
        delta_logvars = []
        for i in range(self.k):
            delta_mu = self.mu_heads[i](mu.detach().clone())  # (K, N, D)
            delta_logvar = self.logvar_heads[i](log_var.detach().clone())  # (K, N, D)
            delta_mus.append(delta_mu)
            delta_logvars.append(delta_logvar)
        return delta_mus, delta_logvars

    def shift_loss(self, delta_mus, delta_logvars, beta, gamma_mu, gamma_sigma):
        """
        delta_mus: Tensor [K, N, D]
        delta_logvars: Tensor [K, N, D]
        """

        mus_var = delta_mus.var(dim=0, unbiased=False).mean()
        logvars_var = delta_logvars.var(dim=0, unbiased=False).mean()

        reg_term = beta * (delta_mus.pow(2) + delta_logvars.pow(2))
        reg_term = reg_term.mean()

        loss = torch.exp(-gamma_mu * mus_var) + torch.exp(-gamma_sigma * logvars_var)
        loss = loss + reg_term

        return loss


class GatingNetwork_Node(nn.Module):
    def __init__(self, node_dim, num_experts):
        super(GatingNetwork_Node, self).__init__()
        self.num_experts = num_experts

        self.gating_network = nn.Sequential(
            nn.Linear(node_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [K, num_nodes, node_dim]
        K, num_nodes, node_dim = x.size()
        x_flat = x.view(K * num_nodes, node_dim)   # [K*num_nodes, node_dim]
        out = self.gating_network(x_flat)          # [K*num_nodes, 1]
        out = out.view(K, num_nodes)               # [K, num_nodes]
        return out

class GatingNetwork(nn.Module):
    def __init__(self, node_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.num_experts = num_experts
        self.readout = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.Tanh()
        )
        self.decision_mlp = nn.Sequential(
            nn.Linear(num_experts * node_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_experts)
        )

    def forward(self, z_list):
        summaries = []
        for z in z_list:
            # Simple readout: apply linear layer and take mean
            node_summaries = self.readout(z)
            graph_summary = node_summaries.mean(dim=0)  # [node_dim]
            summaries.append(graph_summary)
        concatenated_summary = torch.cat(summaries, dim=0)
        logits = self.decision_mlp(concatenated_summary)
        weights = torch.softmax(logits, dim=0).unsqueeze(1)  # Shape [K+1]
        return weights


class VGAE(GAE):
    def __init__(self, encoder: Module, decoder: Optional[Module] = None):
        super().__init__(encoder, decoder)

    def reparametrize(self, mu: Tensor, logstd: Tensor) -> Tensor:
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def encode(self, *args, **kwargs) -> Tensor:
        self.__mu__, self.__logstd__ = self.encoder(*args, **kwargs)
        self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD)
        z = self.reparametrize(self.__mu__, self.__logstd__)
        return z

    def kl_loss(self, mu: Optional[Tensor] = None, logstd: Optional[Tensor] = None) -> Tensor:

        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd.clamp(
            max=MAX_LOGSTD)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))


class VGAE_MoE(GAE):

    def __init__(self, args, encoder: Module):
        super().__init__(encoder)
        self.args = args
        self.k = args.k

        self.latent_dim = args.dim
        self.num_experts = args.k + 1

        self.environment_estimator = Environment_estminator(feat_dim=self.latent_dim, hidden_dim=self.latent_dim, k=args.k)
        self.gating_network = GatingNetwork(node_dim=self.latent_dim, num_experts=self.num_experts)

        self.expert_linears = nn.ModuleList([
            nn.Linear(self.latent_dim, self.latent_dim) for _ in range(self.num_experts)
        ])

        self.all_mus: Optional[Tensor] = None
        self.all_logstds: Optional[Tensor] = None
        self.beta = args.beta
        self.gamma_mu = args.gamma_mu
        self.gamma_std = args.gamma_std

    def reparametrize(self, mu: Tensor, logstd: Tensor) -> Tensor:
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def encode(self, *args, **kwargs):
        original_mu, original_logstd = self.encoder(*args, **kwargs)
        original_logstd = original_logstd.clamp(max=MAX_LOGSTD)

        perturbed_mus, perturbed_logstds = self.environment_estimator(original_mu, original_logstd)

        perturbed_mus = torch.stack(perturbed_mus, dim=0)  # list -> tensor
        all_mus = torch.cat([original_mu.unsqueeze(0), perturbed_mus], dim=0)

        perturbed_logstds = torch.stack(perturbed_logstds, dim=0)  # list -> tensor
        all_logstds = torch.cat([original_logstd.unsqueeze(0), perturbed_logstds.clamp(max=MAX_LOGSTD)], dim=0)

        all_zs = self.reparametrize(all_mus, all_logstds)
        self.all_mus = all_mus
        self.all_logstds = all_logstds

        invariant_matrix = [[] for _ in range(self.num_experts - 1)]
        expert_outputs = []
        for i in range(self.num_experts):
            expert_output = self.expert_linears[i](all_zs[i])  # (num_nodes, latent_dim)
            expert_outputs.append(expert_output)

        referent_output = self.expert_linears[0](all_zs[0])
        gating_weight = self.gating_network(all_zs)  # (num_nodes, k+1)

        expert_outputs_tensor = torch.stack(expert_outputs, dim=1)

        z_aggregated = torch.sum(expert_outputs_tensor * gating_weight, dim=1)

        n = z_aggregated.shape[0]
        if self.args.normalize:
            z_aggregated = F.normalize(z_aggregated.view(n, self.args.n_factors, self.args.delta_d), dim=2).view(z_aggregated.shape[0], z_aggregated.shape[1])

        ref = expert_outputs[0]
        others = torch.stack(expert_outputs[1:], dim=0)
        align_loss = torch.mean(
            torch.norm(ref.unsqueeze(0) - others, p='fro', dim=(1, 2))
        )
        env_loss = self.environment_estimator.shift_loss(perturbed_mus, perturbed_logstds, beta=self.beta, gamma_mu=self.gamma_mu, gamma_sigma=self.gamma_std)
        return z_aggregated, env_loss, align_loss

    def kl_loss(self) -> Tensor:
        if self.all_mus is None or self.all_logstds is None:
            raise RuntimeError("You must call the `encode` method before calculating the KL loss.")
        kl_values = []
        for mu, logstd in zip(self.all_mus, self.all_logstds):
            kl = -0.5 * (
                1 + 2 * logstd - mu.pow(2) - logstd.exp().pow(2)
            )
            kl = kl.sum(dim=-1).mean()
            kl_values.append(kl)

        return torch.stack(kl_values).mean()

