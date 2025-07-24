import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import *
from .blocks import *


class GatedMLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff * 2)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x1, x2 = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(F.gelu(x1) * x2)


class Mamba2EnhancedBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        drop_path_prob: float = 0.1,
        use_mhsa: bool = False,
        num_heads: int = 4,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.ssm = Mamba2Block(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )
        self.drop1 = DropPath(drop_path_prob) if drop_path_prob > 0 else nn.Identity()

        self.use_mhsa = use_mhsa
        if use_mhsa:
            self.norm_attn = nn.LayerNorm(d_model)
            self.attn = nn.MultiheadAttention(
                embed_dim=d_model, num_heads=num_heads, batch_first=True
            )
            self.drop_attn = DropPath(drop_path_prob)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = GatedMLP(d_model, d_model * expand)
        self.drop2 = DropPath(drop_path_prob)

    def forward(self, x: torch.Tensor):
        h = self.norm1(x)
        y = self.ssm(h)
        x = x + self.drop1(y)
        if self.use_mhsa:
            h2 = self.norm_attn(x)
            attn_out, _ = self.attn(h2, h2, h2)
            x = x + self.drop_attn(attn_out)
        h3 = self.norm2(x)
        y2 = self.ffn(h3)
        x = x + self.drop2(y2)
        return x


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
        max_len: int = 10000,
        cls_weight: float = 2.0,
        reg_weight: float = 1.0,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model))

        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            use_mhsa = i % 2 == 1
            self.blocks.append(
                Mamba2EnhancedBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    drop_path_prob=0.1,
                    use_mhsa=use_mhsa,
                    num_heads=4,
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

    def forward(self, x: torch.Tensor):
        _, L, _ = x.shape
        h = self.input_proj(x) + self.pos_emb[:, :L, :]

        for blk in self.blocks:
            h = blk(h)

        h = self.norm(h)
        h = h.mean(dim=1)
        h = self.dropout(h)

        dir_logits = self.dir_head(h)
        reg_out = self.reg_head(h)
        return dir_logits, reg_out

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
