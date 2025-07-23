import torch
import torch.nn as nn
from .blocks import Mamba2Block


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
    ):
        super(Mamba2Multitask, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)

        self.blocks = nn.ModuleList(
            [
                Mamba2Block(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.dir_head = nn.Linear(d_model, 1)
        self.reg_head = nn.Linear(d_model, 3)

    def forward(self, x: torch.Tensor):
        h = self.input_proj(x)

        for block in self.blocks:
            h = h + block(h)

        h = self.norm(h)
        h = h.mean(dim=1)
        h = self.dropout(h)

        dir_logits = self.dir_head(h)
        reg_out = self.reg_head(h)
        return dir_logits, reg_out
