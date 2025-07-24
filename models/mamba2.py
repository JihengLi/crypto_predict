import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import *
from .blocks import *


class Mamba2Multitask(nn.Module):
    def __init__(
        self,
        input_dim: int = 46,
        d_model: int = 256,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        num_layers: int = 4,
        dropout: float = 0.1,
        drop_path_prob: float = 0.1,
        max_len: int = 10000,
        cls_weight: float = 2.0,
        reg_weight: float = 1.0,
        enable_mhsa: bool = True,
        mhsa_every: int = 2,
        num_heads: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model))

        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            use_mhsa = enable_mhsa and ((i % mhsa_every) == (mhsa_every - 1))
            self.blocks.append(
                Mamba2EnhancedBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    drop_path_prob=drop_path_prob,
                    use_mhsa=use_mhsa,
                    num_heads=num_heads,
                )
            )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dir_head = nn.Linear(d_model, 3)
        self.reg_head = nn.Linear(d_model, 3)

        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.log_var_cls = nn.Parameter(torch.zeros(()))
        self.log_var_reg = nn.Parameter(torch.zeros(()))

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def heads(self, h_pooled):
        h = self.dropout(self.norm(h_pooled))
        if h.dim() == 3:
            h = h.mean(dim=1)
        return self.dir_head(h), self.reg_head(h)

    def forward(self, x: torch.Tensor):
        _, L, _ = x.shape
        h = self.input_proj(x) + self.pos_emb[:, :L, :]
        for blk in self.blocks:
            h = blk(h)
        dir_logits, reg_out = self.heads(h)
        return dir_logits, reg_out

    @torch.no_grad()
    def allocate_inference_cache(self, batch_size: int):
        caches = []
        for blk in self.blocks:
            caches.append(blk.allocate_inference_cache(batch_size, self.max_len))
        return caches

    def step(self, x_t, caches):
        h_t = self.input_proj(x_t) + self.pos_emb[:, :1]

        new_caches = []
        for blk, state in zip(self.blocks, caches):
            h_t, new_state = blk.step(h_t, state)
            new_caches.append(new_state)

        h_t = self.dropout(self.norm(h_t))
        return h_t, new_caches

    def multi_task_loss(self, dir_logits, reg_out, cls, p90, p10, sigma):
        cls_loss = F.cross_entropy(dir_logits, cls)
        mse = nn.MSELoss()
        reg_loss = (
            mse(reg_out[:, 0], p90)
            + mse(reg_out[:, 1], p10)
            + mse(reg_out[:, 2], sigma)
        ) / 3.0
        loss = (
            self.cls_weight
            * (cls_loss * torch.exp(-self.log_var_cls) + self.log_var_cls)
            + self.reg_weight
            * (reg_loss * torch.exp(-self.log_var_reg) + self.log_var_reg)
        ) * 0.5
        return loss, cls_loss, reg_loss
