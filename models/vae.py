from functools import partial
from typing import Callable, Tuple, Union
import numpy as np

import torch
import torch.nn as nn
from torch import Tensor

from layers.block import SelfAttentionBlock
from layers.rms_norm import RMSNorm
from layers.rope import RopePositionEmbedding

import lejepa



# class DiagonalGaussianDistribution(object):
#     '''https://github.com/CompVis/latent-diffusion/blob/main/ldm/models/autoencoder.py
#     '''
#     def __init__(self, parameters, deterministic=False):
#         self.parameters = parameters
#         self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
#         self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
#         self.deterministic = deterministic
#         self.std = torch.exp(0.5 * self.logvar)
#         self.var = torch.exp(self.logvar)
#         if self.deterministic:
#             self.var = self.std = torch.zeros_like(self.mean).to(
#                 device=self.parameters.device
#             )

#     def sample(self):
#         x = self.mean + self.std * torch.randn(self.mean.shape, device=self.parameters.device)
#         return x

#     def kl(self, other=None):
#         if self.deterministic:
#             return torch.Tensor([0.0])
#         else:
#             if other is None:
#                 return 0.5 * torch.sum(
#                     torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
#                     dim=[1, 2, 3],
#                 )
#             else:
#                 return 0.5 * torch.sum(
#                     torch.pow(self.mean - other.mean, 2) / other.var
#                     + self.var / other.var
#                     - 1.0
#                     - self.logvar
#                     + other.logvar,
#                     dim=[1, 2, 3],
#                 )

#     def nll(self, sample, dims=[1, 2, 3]):
#         if self.deterministic:
#             return torch.Tensor([0.0])
#         logtwopi = np.log(2.0 * np.pi)
#         return 0.5 * torch.sum(
#             logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
#             dim=dims,
#         )

#     def mode(self):
#         return self.mean

norm_layer_dict = {
    "layernorm": partial(nn.LayerNorm, eps=1e-6),
    "layernormbf16": partial(nn.LayerNorm, eps=1e-5),
    "rmsnorm": RMSNorm,
}

class ViTVAE(nn.Module):
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_chans: int = 3,
        # Encoder
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        ffn_ratio: float = 4.0,
        # Latent
        latent_dim: int = 32,
        variational: bool = True,
        # Decoder
        decoder_embed_dim: int = 768,
        decoder_depth: int = 12,
        decoder_num_heads: int = 12,
        # Common
        qkv_bias: bool = True,
        norm_layer: str = "layernorm",
        init_values: float | None = None,
        # Image normalization
        img_mean: Tuple[float, ...] = (0.5, 0.5, 0.5),
        img_std: Tuple[float, ...] = (0.5, 0.5, 0.5),
        # img_mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        # img_std: Tuple[float, ...] = (0.229, 0.224, 0.225),
        # max_noise_before_decode: float = 0.0,
        noisy_latent = False,
        noise_tau = 0.0,
        # Register tokens
        n_encoder_registers: int = 0,
        n_decoder_registers: int = 0,
        # # loss
        # recon_loss_fn = 'l1',
        # loss_config = {},
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.variational = variational
        self.num_patches_h = img_size // patch_size
        self.num_patches_w = img_size // patch_size
        self.noisy_latent = noisy_latent
        self.noise_tau = noise_tau

        # if recon_loss_fn == "l1":
        #     self.recon_loss_fn = nn.functional.l1_loss
        # elif recon_loss_fn == "l2":
        #     self.recon_loss_fn = nn.functional.mse_loss
        # else:
        #     raise ValueError(f"Invalid recon_loss_fn: {recon_loss_fn}")

        # Resolve norm_layer from string key (dinov3 convention) or use as-is
        # if isinstance(norm_layer, str):
        norm_layer = norm_layer_dict[norm_layer]

        # self.loss_config = loss_config

        # Normalization buffers: [1, C, 1, 1] for broadcast
        self.register_buffer(
            "img_mean",
            torch.tensor(img_mean, dtype=torch.float32).reshape(1, -1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "img_std",
            torch.tensor(img_std, dtype=torch.float32).reshape(1, -1, 1, 1),
            persistent=False,
        )

        # ── Encoder ──────────────────────────────────────────────────────────
        self.patch_proj = nn.Linear(patch_size * patch_size * in_chans, embed_dim)
        self.encoder_rope = RopePositionEmbedding(embed_dim, num_heads=num_heads)
        self.encoder_blocks = nn.ModuleList([
            SelfAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio,
                qkv_bias=qkv_bias,
                init_values=init_values,
                act_layer=nn.GELU,
                norm_layer=norm_layer,
            )
            for _ in range(depth)
        ])
        self.encoder_norm = norm_layer(embed_dim)
        latent_out_dim = latent_dim * 2 if variational else latent_dim
        self.latent_proj = nn.Linear(embed_dim, latent_out_dim)

        self.n_encoder_registers = n_encoder_registers
        if n_encoder_registers > 0:
            self.encoder_registers = nn.Parameter(
                torch.zeros(1, n_encoder_registers, embed_dim)
            )
            nn.init.normal_(self.encoder_registers, std=0.02)

        # ── Decoder ──────────────────────────────────────────────────────────
        self.latent_unproj = nn.Linear(latent_dim, decoder_embed_dim)
        self.decoder_rope = RopePositionEmbedding(decoder_embed_dim, num_heads=decoder_num_heads)
        self.decoder_blocks = nn.ModuleList([
            SelfAttentionBlock(
                dim=decoder_embed_dim,
                num_heads=decoder_num_heads,
                ffn_ratio=ffn_ratio,
                qkv_bias=qkv_bias,
                init_values=init_values,
                act_layer=nn.GELU,
                norm_layer=norm_layer,
            )
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.pixel_head = nn.Linear(decoder_embed_dim, patch_size * patch_size * in_chans)

        self.n_decoder_registers = n_decoder_registers
        if n_decoder_registers > 0:
            self.decoder_registers = nn.Parameter(
                torch.zeros(1, n_decoder_registers, decoder_embed_dim)
            )
            nn.init.normal_(self.decoder_registers, std=0.02)


        # if self.loss_config.get('sigreg_loss', {"use": False})["use"]:
        #     _cfg = self.loss_config['sigreg_loss']
        #     univariate_test = lejepa.univariate.EppsPulley(n_points=_cfg.get('n_points'))
        #     sigreg_fn = lejepa.multivariate.SlicingUnivariateTest(
        #         univariate_test=univariate_test, 
        #         num_slices=_cfg.get('num_slices')
        #     )
        #     self.sigreg_fn = sigreg_fn

        # univariate_test = lejepa.univariate.EppsPulley(n_points=_cfg.n_points)
        # sigreg_fn = lejepa.multivariate.SlicingUnivariateTest(
        #     univariate_test=univariate_test, 
        #     num_slices=_cfg.num_slices
        # )
        # # univariate_test.to(device)
        # sigreg_fn.to(device)
        # sigreg_fn.global_step.add_(train_steps)

    # ── Standalone patch helpers (shape-only) ─────────────────────────────────

    def patchify(self, x: Tensor) -> Tensor:
        """[B, C, H, W] → [B, N, p*p*C]  (pure reshape, no learned weights)."""
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()  # [B, H/p, W/p, p, p, C]
        x = x.reshape(B, (H // p) * (W // p), p * p * C)
        return x

    def unpatchify(self, x: Tensor) -> Tensor:
        """[B, N, p*p*C] → [B, C, H, W]  (pure reshape, no learned weights)."""
        B = x.shape[0]
        p = self.patch_size
        H_p, W_p = self.num_patches_h, self.num_patches_w
        C = self.in_chans
        x = x.reshape(B, H_p, W_p, p, p, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()  # [B, C, H/p, p, W/p, p]
        x = x.reshape(B, C, H_p * p, W_p * p)
        return x

    # ── Core encode / decode ──────────────────────────────────────────────────

    # ── Latent shape helpers ──────────────────────────────────────────────────

    def z_unpatchify(self, z: Tensor) -> Tensor:
        """[B, N, D] → [B, D, H_p, W_p]  (token form → image form)."""
        B, _, D = z.shape
        H_p, W_p = self.num_patches_h, self.num_patches_w
        return z.reshape(B, H_p, W_p, D).permute(0, 3, 1, 2).contiguous()

    def z_patchify(self, z: Tensor) -> Tensor:
        """[B, D, H_p, W_p] → [B, N, D]  (image form → token form)."""
        B, D, H_p, W_p = z.shape
        return z.permute(0, 2, 3, 1).reshape(B, H_p * W_p, D).contiguous()

    # ── Core encode / decode ──────────────────────────────────────────────────

    def encode(self, x: Tensor, normalize: bool = True, unpatchify_latent: bool = False):
        """Encode image → latent.

        Image patchify is always applied. Latent is always projected through
        latent_proj. When unpatchify_latent=True, all latent outputs are
        reshaped from token form [B, N, latent_dim] to image form
        [B, latent_dim, H_p, W_p].

        Returns:
            variational=False : z
            variational=True  : (z, mu, logvar)
        """
        if normalize:
            x = (x - self.img_mean) / self.img_std
        x = self.patchify(x)            # [B, N, p*p*C]
        x = self.patch_proj(x)          # [B, N, embed_dim]

        if self.n_encoder_registers > 0:
            regs = self.encoder_registers.expand(x.shape[0], -1, -1)
            x = torch.cat([regs, x], dim=1)   # [B, R+N, embed_dim]

        rope = self.encoder_rope(H=self.num_patches_h, W=self.num_patches_w)
        for block in self.encoder_blocks:
            x = block(x, rope=rope)     # RoPE auto-skips R prefix tokens
        x = self.encoder_norm(x)

        if self.n_encoder_registers > 0:
            x = x[:, self.n_encoder_registers:]   # drop registers → [B, N, embed_dim]

        z = self.latent_proj(x)         # [B, N, latent_dim] or [B, N, 2*latent_dim]

        if self.variational:
            mu, logvar = z.chunk(2, dim=-1)
            logvar = torch.clamp(logvar, -30.0, 20.0)
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
            if unpatchify_latent:
                z = self.z_unpatchify(z)
                mu = self.z_unpatchify(mu)
                logvar = self.z_unpatchify(logvar)
            return z, mu, logvar

        if unpatchify_latent:
            z = self.z_unpatchify(z)
        return z

    def decode(self, z: Tensor, denormalize: bool = True, patchify_latent: bool = False) -> Tensor:
        """Decode latent → image. Image unpatchify is always applied.

        When patchify_latent=True, z is expected in image form
        [B, latent_dim, H_p, W_p] and is reshaped to token form
        [B, N, latent_dim] before latent_unproj. When patchify_latent=False
        (default), z is expected in token form [B, N, latent_dim].
        """
        if patchify_latent:
            z = self.z_patchify(z)

        x = self.latent_unproj(z)       # [B, N, decoder_embed_dim]

        if self.n_decoder_registers > 0:
            regs = self.decoder_registers.expand(x.shape[0], -1, -1)
            x = torch.cat([regs, x], dim=1)   # [B, R+N, decoder_embed_dim]

        rope = self.decoder_rope(H=self.num_patches_h, W=self.num_patches_w)
        for block in self.decoder_blocks:
            x = block(x, rope=rope)
        x = self.decoder_norm(x)

        if self.n_decoder_registers > 0:
            x = x[:, self.n_decoder_registers:]   # drop registers → [B, N, decoder_embed_dim]

        x = self.pixel_head(x)          # [B, N, p*p*C]
        x = self.unpatchify(x)          # [B, C, H, W]
        if denormalize:
            x = x * self.img_std + self.img_mean
        return x

    # ── End-to-end forward ────────────────────────────────────────────────────

    def forward(self, x: Tensor, return_dict = False) -> dict:
        """Full encode→decode pass on raw images.

        Returns dict with key 'recon' (and 'mu', 'logvar' if variational=True).
        """
        
        x_normalized = (x - self.img_mean) / self.img_std
        z = self.encode(x_normalized, normalize=False, unpatchify_latent=False)
        if self.variational:
            z, mu, logvar = z
        else:
            mu = logvar = None

        if self.noisy_latent and self.training:
            noise_sigma = self.noise_tau * torch.rand((z.size(0),) + (1,) * (len(z.shape) - 1), device=z.device)
            noise = noise_sigma * torch.randn_like(z)
            z_noisy = z + noise
            _z = z_noisy
        else:
            _z = z
            
        x_normalized_recon = self.decode(_z, denormalize=False, patchify_latent=False)
        x_recon = x_normalized_recon * self.img_std + self.img_mean
        if return_dict:
            return {
                'x_recon': x_recon,
                'x_normalized_recon': x_normalized_recon,
                'z': z,
                'mu': mu,
                'logvar': logvar,
            }
        else:
            return x_recon

        # out = vae(x, return_dict=True)
        # loss = 0.0
        # loss_log = {}

        # recon_loss = self.recon_loss_fn(x_normalized_pred, x_normalized)
        # loss += recon_loss * 1.0
        # loss_log["recon_loss"] = recon_loss.item()

        # # kld loss
        # if self.loss_config.get("kld_loss", {"use": False})["use"]:
        #     _loss_weight = self.loss_config["kld_loss"]["weight"]
        #     assert self.variational, "KLD loss is only applicable for variational VAE (variational=True)"
        #     _mu_gt, _std_gt = self.loss_config['kld_loss'].get('target_mu', 0.0), self.loss_config['kld_loss'].get('target_std', 1.0)
        #     # # KL( N(mu, sigma^2) || N(0, I) ) = -0.5 * (1 + logvar - mu^2 - exp(logvar))
        #     _var_gt, _logvar_gt = _std_gt ** 2, 2 * np.log(_std_gt)
        #     mu, logvar = mu.flatten(1), logvar.flatten(1)  # [B, latent_dim]
        #     logvar = torch.clamp(logvar, -30.0, 20.0)
        #     kld_loss = 0.5 * torch.sum(
        #         torch.pow(mu - _mu_gt, 2) / _var_gt
        #         + torch.exp(logvar) / _var_gt
        #         - 1.0
        #         - logvar
        #         + _logvar_gt,
        #         dim=[1],
        #     )
        #     kld_loss = kld_loss.mean()  # mean over batch
        #     # clamped_logvar = logvar.clamp(-30.0, 20.0)
        #     # kld_loss = -0.5 * torch.mean(
        #     #     1 + clamped_logvar - mu.pow(2) - clamped_logvar.exp()
        #     # )
        #     loss += kld_loss * _loss_weight
        #     loss_log["kld_loss"] = kld_loss.item()
        
        # if self.loss_config.get("sigreg_loss", {"use": False})["use"]:
        #     _loss_weight = self.loss_config["sigreg_loss"]["weight"]
        #     sigreg_loss = self.sigreg_fn(z.flatten(1))
        #     loss += sigreg_loss * _loss_weight
        #     loss_log["sigreg_loss"] = sigreg_loss.item()

        # return loss, loss_log, others
