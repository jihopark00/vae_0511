from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .rope import rope_apply, rope_rotate_half


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        device=None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias, device=device)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias, device=device)
        self.proj_drop = nn.Dropout(proj_drop)

    def apply_rope(
        self, q: Tensor, k: Tensor, rope: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        q_dtype, k_dtype = q.dtype, k.dtype
        sin, cos = rope
        rope_dtype = sin.dtype
        q = q.to(dtype=rope_dtype)
        k = k.to(dtype=rope_dtype)
        N = q.shape[-2]
        prefix = N - sin.shape[-2]
        assert prefix >= 0
        q_prefix, q_patch = q[:, :, :prefix, :], q[:, :, prefix:, :]
        k_prefix, k_patch = k[:, :, :prefix, :], k[:, :, prefix:, :]
        q_patch = rope_apply(q_patch, sin, cos)
        k_patch = rope_apply(k_patch, sin, cos)
        q = torch.cat((q_prefix, q_patch), dim=-2)
        k = torch.cat((k_prefix, k_patch), dim=-2)
        return q.to(q_dtype), k.to(k_dtype)

    def forward(self, x: Tensor, rope: Tuple[Tensor, Tensor] | None = None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = torch.unbind(qkv, dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if rope is not None:
            q, k = self.apply_rope(q, k, rope)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
