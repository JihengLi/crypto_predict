import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention and cross-attention"""

    def __init__(
        self,
        embed_dim,
        num_heads,
        num_heads_kv=None,
        head_dim=None,
        mlp_dim=0,
        qkv_proj_bias=True,
        out_proj_bias=True,
        softmax_scale=None,
        causal=False,
        layer_idx=None,
        d_conv=0,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.layer_idx = layer_idx
        self.d_conv = d_conv
        self.softmax_scale = softmax_scale
        self.causal = causal

        self.num_heads = num_heads
        self.num_heads_kv = num_heads_kv if num_heads_kv is not None else num_heads
        assert (
            self.num_heads % self.num_heads_kv == 0
        ), "num_heads must be divisible by num_heads_kv"
        if head_dim is None:
            assert (
                self.embed_dim % num_heads == 0
            ), "embed_dim must be divisible by num_heads"
        self.head_dim = (
            head_dim if head_dim is not None else self.embed_dim // num_heads
        )
        self.mlp_dim = math.ceil(mlp_dim / 256) * 256
        qkv_dim = self.head_dim * (self.num_heads + 2 * self.num_heads_kv)
        out_dim = self.head_dim * self.num_heads

        self.in_proj = nn.Linear(
            embed_dim, qkv_dim + self.mlp_dim, bias=qkv_proj_bias, **factory_kwargs
        )
        if self.d_conv > 0:
            self.conv1d = nn.Conv1d(
                qkv_dim,
                qkv_dim,
                kernel_size=self.d_conv,
                padding=self.d_conv - 1,
                groups=qkv_dim,
                **factory_kwargs
            )
        self.out_proj = nn.Linear(
            out_dim + self.mlp_dim // 2, embed_dim, bias=out_proj_bias, **factory_kwargs
        )

    @torch.no_grad()
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        dtype = self.out_proj.weight.dtype if dtype is None else dtype
        device = self.out_proj.weight.device
        if self.d_conv > 0:
            conv_state = torch.zeros(
                batch_size,
                self.conv1d.weight.shape[0],
                self.d_conv,
                device=device,
                dtype=dtype,
            )
        else:
            conv_state = None
        kv_cache = torch.empty(
            batch_size,
            max_seqlen,
            2,
            self.num_heads_kv,
            self.head_dim,
            device=device,
            dtype=dtype,
        )
        t_ptr = torch.zeros(batch_size, dtype=torch.long, device=device)
        return kv_cache, conv_state, t_ptr

    def forward(self, x):
        qkv = self.in_proj(x)
        if self.mlp_dim > 0:
            qkv, x_mlp = qkv.split([qkv.shape[-1] - self.mlp_dim, self.mlp_dim], dim=-1)
            x_mlp_up, x_mlp_gate = x_mlp.chunk(2, dim=-1)
            x_mlp = x_mlp_up * F.silu(x_mlp_gate)
        if self.d_conv > 0:
            qkv = rearrange(
                self.conv1d(rearrange(qkv, "b s d -> b d s"))[
                    ..., : -(self.d_conv - 1)
                ],
                "b d s -> b s d",
            ).contiguous()
        q, kv = qkv.split(
            [self.num_heads * self.head_dim, self.num_heads_kv * 2 * self.head_dim],
            dim=-1,
        )
        q = rearrange(q, "... (h d) -> ... h d", d=self.head_dim)
        kv = rearrange(kv, "... (two hkv d) -> ... two hkv d", two=2, d=self.head_dim)
        k, v = kv.unbind(dim=-3)
        k = torch.repeat_interleave(
            k, dim=2, repeats=self.num_heads // self.num_heads_kv
        )
        v = torch.repeat_interleave(
            v, dim=2, repeats=self.num_heads // self.num_heads_kv
        )
        context = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=self.causal,
            scale=self.softmax_scale,
        ).transpose(1, 2)
        context = rearrange(context, "... h d -> ... (h d)")
        if self.mlp_dim > 0:
            context = torch.cat([context, x_mlp], dim=-1)
        out = self.out_proj(context)
        return out

    def step(self, x_t: torch.Tensor, cache):
        """
        x_t:  (B,1,D_model)
        cache: (kv_cache, conv_state, t_ptr)
        return: (out_t, new_cache)
        """
        kv_cache, conv_state, t_ptr = cache  # kv_cache:(B,L_max,2,Hkv,dh)
        B, _, _ = x_t.shape
        dtype = x_t.dtype
        device = x_t.device
        H, H_kv, dh = self.num_heads, self.num_heads_kv, self.head_dim
        qkv = self.in_proj(x_t)  # (B,1, qkv_dim)

        if self.d_conv > 0:
            qkv_flat = qkv.squeeze(1)  # (B,D)
            # roll & write conv_state
            conv_state = torch.roll(conv_state, shifts=-1, dims=-1)  # (B,D,K)
            conv_state[..., -1] = qkv_flat.to(conv_state.dtype)
            w = self.conv1d.weight.squeeze(1)  # (D,K)
            qkv_flat = torch.einsum("bdw,dw->bd", conv_state, w)  # (B,D)
            if self.conv1d.bias is not None:
                qkv_flat = qkv_flat + self.conv1d.bias
            qkv = qkv_flat.unsqueeze(1)

        q, k_new, v_new = qkv.split([H * dh, H_kv * dh, H_kv * dh], dim=-1)
        q = q.view(B, 1, H, dh)
        k_new = k_new.view(B, 1, H_kv, dh)
        v_new = v_new.view(B, 1, H_kv, dh)

        batch_idx = torch.arange(B, device=device)
        kv_cache[batch_idx, t_ptr, 0] = k_new.squeeze(1).to(kv_cache.dtype)
        kv_cache[batch_idx, t_ptr, 1] = v_new.squeeze(1).to(kv_cache.dtype)
        t_ptr_new = t_ptr + 1  # (B,)

        max_L = int(t_ptr_new.max().item())
        k_eff = kv_cache[:, :max_L, 0].to(dtype)  # (B,L_eff,Hkv,dh)
        v_eff = kv_cache[:, :max_L, 1].to(dtype)

        repeat = H // H_kv
        k_eff = k_eff.repeat_interleave(repeat, dim=2)  # (B,L_eff,H,dh)
        v_eff = v_eff.repeat_interleave(repeat, dim=2)

        q = q.transpose(1, 2)  # (B,H,1,dh)
        k = k_eff.transpose(1, 2)  # (B,H,L_eff,dh)
        v = v_eff.transpose(1, 2)
        context = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=self.softmax_scale
        )  # (B,H,1,dh)
        context = context.transpose(1, 2).reshape(B, 1, -1)  # (B,1,D)

        out_t = self.out_proj(context)  # (B,1,D_model)
        new_cache = (kv_cache, conv_state, t_ptr_new)
        return out_t, new_cache
