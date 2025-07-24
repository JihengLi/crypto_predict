import torch
import torch.nn as nn

from ..modules import *
from .mamba2_block import *


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
            self.mha = MultiHeadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                causal=True,
                d_conv=d_conv,
            )
            self.drop_attn = DropPath(drop_path_prob)
        else:
            self.norm_attn = nn.Identity()
            self.attn_proj_qkv = nn.Identity()
            self.out_proj_attn = nn.Identity()
            self.drop_attn = nn.Identity()

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
            attn_out = self.mha(self.norm_attn(x))
            x = x + self.drop_attn(attn_out)

        x = x + self.drop2(self.ffn(self.norm2(x)))
        return x

    @torch.no_grad()
    def allocate_inference_cache(self, batch_size: int, max_len: int):
        conv_state, ssm_state = self.ssm.allocate_inference_cache(batch_size)
        if self.use_mhsa:
            kv_cache, attn_conv_state, t_ptr = self.mha.allocate_inference_cache(
                batch_size=batch_size,
                max_seqlen=max_len,
            )
        else:
            kv_cache = attn_conv_state = t_ptr = None
        return (conv_state, ssm_state, kv_cache, attn_conv_state, t_ptr)

    def step(self, x_t, states):
        assert x_t.size(1) == 1, f"got seq={x_t.size(1)} in {self}"
        conv_state, ssm_state, kv_cache, attn_conv_state, t_ptr = states
        y_t, conv_state, ssm_state = self.ssm.step(x_t, conv_state, ssm_state)
        x_t = x_t + self.drop1(y_t)
        if self.use_mhsa:
            attn_out, (kv_cache, attn_conv_state, t_ptr) = self.mha.step(
                self.norm_attn(x_t), (kv_cache, attn_conv_state, t_ptr)
            )
            x_t = x_t + self.drop_attn(attn_out)
        x_t = x_t + self.drop2(self.ffn(self.norm2(x_t)))
        return x_t, (conv_state, ssm_state, kv_cache, attn_conv_state, t_ptr)
