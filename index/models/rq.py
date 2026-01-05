import torch
import torch.nn as nn
import torch.nn.functional as F
from .vq import VectorQuantizer


class ResidualVectorQuantizer(nn.Module):
    """ References:
        SoundStream: An End-to-End Neural Audio Codec
        https://arxiv.org/pdf/2107.03312.pdf
    """

    def __init__(self, n_e_list, e_dim, sk_epsilons,
                 kmeans_init = False, kmeans_iters = 100, sk_iters=100, use_linear=0):
        super().__init__()
        self.n_e_list = n_e_list
        self.e_dim = e_dim
        self.num_quantizers = len(n_e_list)
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        self.vq_layers = nn.ModuleList([VectorQuantizer(n_e, e_dim,
                                                        kmeans_init = self.kmeans_init,
                                                        kmeans_iters = self.kmeans_iters,
                                                        sk_epsilon=sk_epsilon,
                                                        sk_iters=sk_iters,
                                                        use_linear=use_linear)
                                        for n_e, sk_epsilon in zip(n_e_list,sk_epsilons) ])

    def get_codebook(self):
        all_codebook = []
        for quantizer in self.vq_layers:
            codebook = quantizer.get_codebook()
            all_codebook.append(codebook)
        return torch.stack(all_codebook)

    def forward(self, x, use_sk=True):
        all_losses = []
        all_indices = []
        all_distances = []
        all_codes = []

        x_q = 0
        residual = x
        for quantizer in self.vq_layers:
            x_res, loss, indices, distance = quantizer(residual, use_sk=use_sk)
            residual = residual - x_res
            x_q = x_q + x_res

            all_losses.append(loss)
            all_indices.append(indices)
            all_distances.append(distance)
            
            # ===== 新增：把每层选中的码字向量取出来 =====
            # indices shape: x.shape[:-1]
            flat_idx = indices.reshape(-1)  # [N]
            # embeddings_weight 要与 quantizer.forward 里保持一致（考虑 use_linear）
            if quantizer.use_linear == 1:
                embeddings_weight = quantizer.codebook_projection(quantizer.embedding.weight)
                code_vec = F.embedding(flat_idx, embeddings_weight)  # [N, e_dim]
            else:
                code_vec = quantizer.embedding(flat_idx)             # [N, e_dim]
            code_vec = code_vec.view(*indices.shape, self.e_dim)     # [*x.shape[:-1], e_dim]
            all_codes.append(code_vec)


        mean_losses = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)
        all_distances = torch.stack(all_distances, dim=1)
        
        # 新增：codes 堆叠，shape: [*x.shape[:-1], L, e_dim]
        all_codes = torch.stack(all_codes, dim=-2)

        return x_q, mean_losses, all_indices, all_distances, all_codes