import torch
from torch import nn


class DropPath(nn.Module):
    """Per-sample stochastic depth (a.k.a DropPath)"""

    __constants__ = ["scale_by_keep"]

    def __init__(self, drop_connect_rate: float = 0.2, *, scale_by_keep: bool = True):
        super().__init__()
        assert 0.0 <= drop_connect_rate < 1.0
        self.drop_connect_rate = float(drop_connect_rate)
        self.scale_by_keep = scale_by_keep

    def forward(
        self, x: torch.Tensor, tmp_drop_connect_rate: float | None = None
    ) -> torch.Tensor:
        p = (
            self.drop_connect_rate
            if tmp_drop_connect_rate is None
            else tmp_drop_connect_rate
        )
        if p == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - p
        mask_shape = (x.shape[0],) + (1,) * (x.dim() - 1)
        mask = x.new_empty(mask_shape).bernoulli_(keep_prob)

        if keep_prob > 0.0 and self.scale_by_keep:
            mask.div_(keep_prob)

        return x * mask

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.3f}"
