import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .layers import MLPLayers
from .rq import ResidualVectorQuantizer


class RQVAE(nn.Module):
    def __init__(self,
                 in_dim=768,
                 # num_emb_list=[256,256,256,256],
                 num_emb_list=None,
                 e_dim=64,
                 # layers=[512,256,128],
                 layers=None,
                 dropout_prob=0.0,
                 bn=False,
                 loss_type="mse",
                 quant_loss_weight=1.0,
                 kmeans_init=False,
                 kmeans_iters=100,
                 # sk_epsilons=[0,0,0.003,0.01]],
                 sk_epsilons=None,
                 sk_iters=100,
                 use_linear=0,
                 use_orth_loss=False,
                 orth_loss_weight=0.0
        ):
        super(RQVAE, self).__init__()

        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim

        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight=quant_loss_weight
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        
        self.use_orth_loss = use_orth_loss
        self.orth_loss_weight = orth_loss_weight


        self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        self.encoder = MLPLayers(layers=self.encode_layer_dims,
                                 dropout=self.dropout_prob,bn=self.bn)

        self.rq = ResidualVectorQuantizer(num_emb_list, e_dim,
                                          kmeans_init = self.kmeans_init,
                                          kmeans_iters = self.kmeans_iters,
                                          sk_epsilons=self.sk_epsilons,
                                          sk_iters=self.sk_iters,
                                          use_linear=use_linear)

        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLPLayers(layers=self.decode_layer_dims,
                                       dropout=self.dropout_prob,bn=self.bn)
        
    def compute_orth_loss(self, codes):
        """
        codes: [*X, L, D]  其中 *X = x.shape[:-1] (token/item 维度)
        返回：标量 loss
        """
        # 展平 token/item 维度 => [N, L, D]
        L = codes.shape[-2]
        if L < 2:
            return codes.new_tensor(0.0)

        codes = codes.reshape(-1, L, codes.shape[-1])           # [N, L, D]
        codes = F.normalize(codes, dim=-1, eps=1e-12)           # 归一化得到 \tilde e
        G = torch.matmul(codes, codes.transpose(-1, -2))        # [N, L, L]

        # 取 off-diagonal 的平方和
        eye = torch.eye(L, device=codes.device, dtype=torch.bool)
        off_diag = ~eye
        # sum_{i != j} (dot)^2 / (L(L-1))
        loss_per = (G.pow(2)[:, off_diag]).sum(dim=-1) / (L * (L - 1))  # [N]
        return loss_per.mean()


    def forward(self, x, use_sk=True):
        x = self.encoder(x)
        # x_q, rq_loss, indices, distances = self.rq(x,use_sk=use_sk)
        x_q, rq_loss, indices, distances, codes = self.rq(x, use_sk=use_sk)
        # print(indices.shape)
        if self.use_orth_loss:
            orth_loss = self.compute_orth_loss(codes)
        else:
            orth_loss = x.new_tensor(0.0)
        out = self.decoder(x_q)

        return out, rq_loss, orth_loss, indices

    @torch.no_grad()
    def get_indices(self, xs, use_sk=False):
        x_e = self.encoder(xs)
        _, _, indices, distances, _ = self.rq(x_e, use_sk=use_sk)
        return indices, distances
        # _, _, indices, distances = self.rq(x_e, use_sk=use_sk)
        # return indices, distances

    def compute_loss(self, out, quant_loss, xs=None):

        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(out, xs, reduction='mean')
        elif self.loss_type == 'l1':
            loss_recon = F.l1_loss(out, xs, reduction='mean')
        else:
            raise ValueError('incompatible loss type')

        loss_total = loss_recon + self.quant_loss_weight * quant_loss

        return loss_total, loss_recon