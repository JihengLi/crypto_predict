import torch
import torch.nn as nn
import torch.nn.functional as F


def layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    *,
    eps: float = 1e-6,
    z: torch.Tensor = None,
    group_size: int = None,
    norm_before_gate: bool = True,
    is_rms_norm: bool = False,
) -> torch.Tensor:
    orig_shape = x.shape
    N = orig_shape[-1]
    x = x.reshape(-1, N)
    if z is not None:
        z = z.reshape(-1, N)
    if z is not None and not norm_before_gate:
        x = x * F.silu(z)
    if x.stride(-1) != 1:
        x = x.contiguous()
    if group_size is None or group_size == N:
        x32 = x.float()
        if not is_rms_norm:
            mean = x32.mean(dim=-1, keepdim=True)
            var = (x32 - mean).pow(2).mean(dim=-1, keepdim=True)
            xhat = (x32 - mean) / torch.sqrt(var + eps)
        else:
            var = x32.pow(2).mean(dim=-1, keepdim=True)
            xhat = x32 / torch.sqrt(var + eps)
    else:
        assert N % group_size == 0, "N must be divisible by group_size"
        G = N // group_size
        xg = x.reshape(-1, G, group_size).float()  # upcast
        if not is_rms_norm:
            mean = xg.mean(dim=-1, keepdim=True)
            var = (xg - mean).pow(2).mean(dim=-1, keepdim=True)
            xhat = (xg - mean) / torch.sqrt(var + eps)
        else:
            var = xg.pow(2).mean(dim=-1, keepdim=True)
            xhat = xg / torch.sqrt(var + eps)
        xhat = xhat.reshape(-1, N)

    w = weight.contiguous()
    y = xhat.to(x.dtype) * w
    if bias is not None:
        b = bias.contiguous()
        y = y + b

    if z is not None and norm_before_gate:
        y = y * F.silu(z)

    return y.reshape(orig_shape)


class LayerNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        *,
        eps: float = 1e-6,
        group_size: int = None,
        norm_before_gate: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.eps = eps
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate
        self.weight = nn.Parameter(torch.ones(hidden_size, **factory_kwargs))
        self.bias = nn.Parameter(torch.zeros(hidden_size, **factory_kwargs))

    def forward(self, x: torch.Tensor, z: torch.Tensor = None) -> torch.Tensor:
        return layer_norm(
            x,
            self.weight,
            self.bias,
            eps=self.eps,
            z=z,
            group_size=self.group_size,
            norm_before_gate=self.norm_before_gate,
            is_rms_norm=False,
        )


class RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        *,
        eps: float = 1e-6,
        group_size: int = None,
        norm_before_gate: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.eps = eps
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate
        self.weight = nn.Parameter(torch.ones(hidden_size, **factory_kwargs))

    def forward(self, x: torch.Tensor, z: torch.Tensor = None) -> torch.Tensor:
        return layer_norm(
            x,
            self.weight,
            None,
            eps=self.eps,
            z=z,
            group_size=self.group_size,
            norm_before_gate=self.norm_before_gate,
            is_rms_norm=True,
        )
