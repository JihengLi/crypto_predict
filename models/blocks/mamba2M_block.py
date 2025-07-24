import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import *
from .mamba2_block import *


def mha_step(
    q_new: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    t_ptr: torch.Tensor,
    num_heads: int,
):
    B, _, D = q_new.shape
    H = num_heads
    d_h = D // H

    with torch.no_grad():
        k_cache.scatter_(
            1, t_ptr[:, None, None].expand_as(q_new), q_new.to(k_cache.dtype)
        )
        v_cache.scatter_(
            1, t_ptr[:, None, None].expand_as(q_new), q_new.to(v_cache.dtype)
        )
        t_ptr += 1

    max_L = int(t_ptr.max().item())
    k_eff = k_cache[:, :max_L].to(q_new.dtype)
    v_eff = v_cache[:, :max_L].to(q_new.dtype)

    q = q_new.view(B, 1, H, d_h).transpose(1, 2)
    k = k_eff.view(B, max_L, H, d_h).transpose(1, 2)
    v = v_eff.view(B, max_L, H, d_h).transpose(1, 2)

    attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    out = attn.transpose(1, 2).reshape(B, 1, D)
    return out, k_cache, v_cache, t_ptr


class GatedMLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        activation=F.silu,
        bias=False,
        multiple_of=128,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        out_features = out_features if out_features is not None else in_features
        hidden_features = (
            hidden_features if hidden_features is not None else int(8 * in_features / 3)
        )
        hidden_features = (
            (hidden_features + multiple_of - 1) // multiple_of * multiple_of
        )
        self.fc1 = nn.Linear(
            in_features, 2 * hidden_features, bias=bias, **factory_kwargs
        )
        self.activation = activation
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, **factory_kwargs)

    def forward(self, x):
        y = self.fc1(x)
        y, gate = y.chunk(2, dim=-1)
        y = y * self.activation(gate)
        y = self.fc2(y)
        return y


class Mamba2EnhancedBlock(nn.Module):
    def __init__(
        self,
        d_model,
        d_state,
        d_conv,
        expand,
        drop_path_prob=0.1,
        use_mhsa=False,
        num_heads=4,
    ):
        super().__init__()
        self.use_mhsa = use_mhsa
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.ssm = Mamba2Block(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )
        self.drop1 = DropPath(drop_path_prob) if drop_path_prob > 0 else nn.Identity()

        if use_mhsa:
            self.norm_attn = nn.LayerNorm(d_model)
            self.attn_proj_qkv = nn.Linear(d_model, d_model * 3, bias=False)
            self.out_proj_attn = nn.Linear(d_model, d_model, bias=False)
            self.drop_attn = DropPath(drop_path_prob)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = GatedMLP(
            in_features=d_model,
            hidden_features=d_model * expand,
        )
        self.drop2 = DropPath(drop_path_prob)

    def forward(self, x: torch.Tensor):
        y = self.ssm(self.norm1(x))
        x = x + self.drop1(y)

        if self.use_mhsa:
            h = self.norm_attn(x)
            qkv = self.attn_proj_qkv(h)
            q, k, v = qkv.chunk(3, dim=-1)
            attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            attn_out = self.out_proj_attn(attn_out)
            x = x + self.drop_attn(attn_out)

        x = x + self.drop2(self.ffn(self.norm2(x)))
        return x

    @torch.no_grad()
    def allocate_inference_cache(self, batch_size: int, max_len: int):
        param_ref = next(self.parameters())
        device = param_ref.device
        dtype = param_ref.dtype
        conv_state, ssm_state = self.ssm.allocate_inference_cache(batch_size)
        if self.use_mhsa:
            d_model = self.ssm.d_model
            k_cache = torch.empty(
                batch_size, max_len, d_model, device=device, dtype=dtype
            )
            v_cache = torch.empty_like(k_cache)
            t_ptr = torch.zeros(batch_size, dtype=torch.long, device=device)
        else:
            k_cache = v_cache = t_ptr = None
        return (conv_state, ssm_state, k_cache, v_cache, t_ptr)

    def step(self, x_t, states):
        assert x_t.size(1) == 1, f"got seq={x_t.size(1)} in {self}"
        conv_state, ssm_state, k_cache, v_cache, t_ptr = states
        y_t, conv_state, ssm_state = self.ssm.step(x_t, conv_state, ssm_state)
        x_t = x_t + self.drop1(y_t)
        if self.use_mhsa:
            h_t = self.norm_attn(x_t)
            qkv = self.attn_proj_qkv(h_t)
            q_new, _, _ = qkv.chunk(3, dim=-1)

            attn_out, k_cache, v_cache, t_ptr = mha_step(
                q_new, k_cache, v_cache, t_ptr, self.num_heads
            )
            x_t = x_t + self.drop_attn(self.out_proj_attn(attn_out))
        x_t = x_t + self.drop2(self.ffn(self.norm2(x_t)))
        return x_t, (conv_state, ssm_state, k_cache, v_cache, t_ptr)
