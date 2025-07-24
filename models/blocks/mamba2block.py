import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from ..utils import RMSNorm
from huggingface_hub import PyTorchModelHubMixin


def mamba_chunk_scan_combined(
    x,
    dt,
    A,
    B_param,
    C_param,
    chunk_size=256,
    D=None,
    z=None,
    dt_bias=None,
    dt_softplus=True,
    dt_limit=(0.0, float("inf")),
):
    Bsz, L, h, p = x.shape
    device, dtype = x.device, x.dtype
    if dt_softplus:
        dt_dis = F.softplus(dt.unsqueeze(-1) + dt_bias.reshape(1, h)).clamp(
            *dt_limit
        )  # (B,h,1)
    else:
        dt_dis = dt.unsqueeze(-1)  # (B,h,1)
    dt_dis = dt_dis.unsqueeze(-1)  # (B,h,1,1)
    d_state = C_param.shape[-1]
    state = torch.zeros(Bsz, h, p, d_state, device=device, dtype=dtype)
    y = torch.zeros_like(x)
    for start in range(0, L, chunk_size):
        end = min(start + chunk_size, L)
        for t in range(start, end):
            dt_t = dt[:, t, :].unsqueeze(-1).unsqueeze(-1)  # (B, h, 1, 1)
            if dt_softplus:
                dt_t = F.softplus(dt_t + dt_bias.reshape(1, h, 1, 1)).clamp(*dt_limit)
            x_t = x[:, t, :, :].unsqueeze(-1)  # (B, h, p, 1)
            B_t = B_param[:, t, :].unsqueeze(1).unsqueeze(2)  # (B, 1, 1, d_state)
            C_t = C_param[:, t, :].unsqueeze(1).unsqueeze(2)  # (B, 1, 1, d_state)
            dA = torch.exp(dt_t * A.reshape(1, h, 1, 1))  # (B, h, 1, 1)
            dBx = dt_t * B_t * x_t  # (B, h, p, d_state)
            state = state * dA + dBx
            state = state.detach()
            y_ssm = torch.einsum("bhpd,bhpd->bhp", state, C_t.expand_as(state))
            if D is not None:
                D_exp = D if D.ndim == 2 else D.view(h, 1)
                y_t = y_ssm + x_t.squeeze(-1) * D_exp
            else:
                y_t = y_ssm
            if z is not None:
                z_t = z[:, t, :, :]
                y_t = y_t * F.silu(z_t)
            y[:, t, :, :] = y_t
    return y  # (B, L, h, p)


class Mamba2Block(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        d_model,
        d_state=128,
        d_conv=4,
        conv_init=None,
        expand=2,
        headdim=64,
        d_ssm=None,
        ngroups=1,
        A_init_range=(1, 16),
        D_has_hdim=False,
        rmsnorm=True,
        norm_before_gate=False,
        chunksize=256,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        bias=False,
        conv_bias=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        assert self.d_inner == self.expand * self.d_model
        self.headdim = headdim
        self.d_ssm = self.d_inner if d_ssm is None else d_ssm
        self.ngroups = ngroups
        assert self.d_ssm % self.headdim == 0
        self.nheads = self.d_ssm // self.headdim
        self.D_has_hdim = D_has_hdim
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate
        self.chunksize = chunksize
        self.dt_limit = dt_limit
        self.activation = "silu"

        # [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)

        conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)

        self.act = nn.SiLU()
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(
            *A_init_range
        )
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(
            torch.ones(self.d_ssm if self.D_has_hdim else self.nheads, device=device)
        )
        self.D._no_weight_decay = True

        if self.rmsnorm:
            assert RMSNorm is not None
            self.norm = RMSNorm(
                self.d_ssm,
                eps=1e-5,
                norm_before_gate=self.norm_before_gate,
                group_size=self.d_ssm // ngroups,
                **factory_kwargs,
            )

        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )

    def forward(self, u):
        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj) or (B * L, d_in_proj)

        A = -torch.exp(self.A_log.float())  # (nheads) or (d_inner, d_state)
        d_mlp = (
            zxbcdt.shape[-1]
            - 2 * self.d_ssm
            - 2 * self.ngroups * self.d_state
            - self.nheads
        ) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [
                d_mlp,
                d_mlp,
                self.d_ssm,
                self.d_ssm + 2 * self.ngroups * self.d_state,
                self.nheads,
            ],
            dim=-1,
        )
        assert self.activation in ["silu", "swish"]

        xBC = self.act(
            self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, : -(self.d_conv - 1)]
        )  # (B, L, self.d_ssm + 2 * ngroups * d_state)

        x, B_param, C_param = torch.split(
            xBC,
            [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state],
            dim=-1,
        )
        x_hp = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
        z_hp = (
            rearrange(z, "b l (h p) -> b l h p", p=self.headdim)
            if (z is not None and not self.rmsnorm)
            else None
        )
        D_param = (
            rearrange(self.D, "(h p) -> h p", p=self.headdim)
            if self.D_has_hdim
            else self.D
        )

        y = mamba_chunk_scan_combined(
            x_hp,
            dt,
            A,
            B_param,
            C_param,
            chunk_size=self.chunksize,
            D=D_param,
            z=z_hp,
            dt_bias=self.dt_bias,
            dt_softplus=True,
            dt_limit=self.dt_limit,
        )
        y = rearrange(y, "b l h p -> b l (h p)")
        if self.rmsnorm:
            y = self.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert (
            hidden_states.shape[1] == 1
        ), "Only support decoding with 1 token at a time for now"
        zxbcdt = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        d_mlp = (
            zxbcdt.shape[-1]
            - 2 * self.d_ssm
            - 2 * self.ngroups * self.d_state
            - self.nheads
        ) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [
                d_mlp,
                d_mlp,
                self.d_ssm,
                self.d_ssm + 2 * self.ngroups * self.d_state,
                self.nheads,
            ],
            dim=-1,
        )

        # Conv step
        conv_state.copy_(
            torch.roll(conv_state, shifts=-1, dims=-1)
        )  # Update state (B D W)
        conv_state[:, :, -1] = xBC
        xBC = torch.sum(
            conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1
        )  # (B D)
        if self.conv1d.bias is not None:
            xBC = xBC + self.conv1d.bias
        xBC = self.act(xBC).to(dtype=dtype)

        x, B, C = torch.split(
            xBC,
            [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state],
            dim=-1,
        )
        A = -torch.exp(self.A_log.float())  # (nheads,)

        # SSM step
        assert self.ngroups == 1, "Only support ngroups=1 for this inference code path"
        # Discretize A and B
        dt = F.softplus(dt + self.dt_bias.to(dtype=dt.dtype))  # (batch, nheads)
        dA = torch.exp(dt * A)  # (batch, nheads)
        x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
        dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x)
        ssm_state.copy_(ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
        y = torch.einsum("bhpn,bn->bhp", ssm_state.to(dtype), C)
        y = y + rearrange(self.D.to(dtype), "h -> h 1") * x
        y = rearrange(y, "b h p -> b (h p)")
        if not self.rmsnorm:
            y = y * self.act(z)  # (B D)
        if self.rmsnorm:
            y = self.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state
