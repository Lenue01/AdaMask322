"""Transformer model and timestep conditioning for AdaMask."""

import math
import torch
import torch.nn as nn


def timestamp_embedding(t, dim):
    # The transformer receives a timestep embedding to know how corrupted the input is.
    # This uses the same sinusoidal style embeddings as classic transformer position encodings.
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class AdaLNBlock(nn.Module):
    """Transformer block with adaptive layer-norm conditioning from timestep embeddings."""

    def __init__(self, config):
        super().__init__()
        H = config.hidden_size

        # Layer norm used before attention and before the feed-forward network.
        self.ln_1 = nn.LayerNorm(H, elementwise_affine=False)
        self.ln_2 = nn.LayerNorm(H, elementwise_affine=False)

        # Multi-head self-attention block.
        self.attn = nn.MultiheadAttention(H, config.heads, batch_first=True, dropout=0.0)

        # Feed-forward network in the transformer block.
        self.mlp = nn.Sequential(
            nn.Linear(H, H * 4),
            nn.GELU(),
            nn.Linear(H * 4, H),
            nn.Dropout(0.1),
        )

        # Project timestep embeddings into scale/shift values for adaptive LN.
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(H, 4 * H),
        )

    def forward(self, x, t_emb, key_padding_mask=None):
        """Process one transformer block with timestep-modulated normalization."""
        mod = self.adaLN_modulation(t_emb)
        shift_msa, scale_msa, shift_mlp, scale_mlp = mod.chunk(4, dim=-1)

        # Apply adaptive layer norm before attention.
        x_mod = self.ln_1(x) * (1 + scale_msa) + shift_msa
        attn_out, _ = self.attn(x_mod, x_mod, x_mod, key_padding_mask=key_padding_mask)
        x = x + attn_out

        # Apply adaptive layer norm before the MLP.
        x_mod = self.ln_2(x) * (1 + scale_mlp) + shift_mlp
        x = x + self.mlp(x_mod)
        return x


class MaskedDiffusionTransformer(nn.Module):
    """Transformer denoiser that predicts masked tokens conditioned on timestep."""

    def __init__(self, config):
        super().__init__()
        H = config.hidden_size

        # Token embedding layer maps token IDs to vectors.
        self.token_emb = nn.Embedding(config.vocab_size, H)

        # Learnable positional embeddings for sequence positions.
        self.pos_emb = nn.Parameter(torch.randn(1, config.context_length, H) * 0.02)

        # Timestep MLP converts timestep embeddings into a model-space vector.
        self.time_mlp = nn.Sequential(nn.Linear(H, H), nn.SiLU(), nn.Linear(H, H))

        # Sequence of transformer blocks.
        self.blocks = nn.ModuleList([AdaLNBlock(config) for _ in range(config.layers)])

        # Final normalization layer.
        self.ln_f = nn.LayerNorm(H)

        # Output head projects hidden states back to vocabulary logits.
        self.head = nn.Linear(H, config.vocab_size, bias=False)
        self._init_weights()

        # Weight tying: share embedding and unembedding matrices.
        self.head.weight = self.token_emb.weight

    def _init_weights(self):
        # Initialize weights for embeddings and linear layers.
        nn.init.normal_(self.token_emb.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Zero-init each block's adaLN output layer so scale=0 and shift=0 at
        # step zero: x*(1+0)+0 == x, so every block starts as an identity-ish
        # residual instead of an arbitrary random perturbation. Standard DiT
        # initialization -- without it, the generic std=0.02 init above gives
        # every one of the `layers` stacked blocks its own random nonzero
        # modulation from step one, compounding into noisy early training.
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)

    def forward(self, x_t, t, key_padding_mask=None):
        """Predict token logits for a corrupted sequence at timestep t."""
        # Embed tokens and add positional information.
        x = self.token_emb(x_t) + self.pos_emb[:, : x_t.size(1)]

        # Build a timestep embedding and project it into model dimension.
        t_emb = timestamp_embedding(t, x.size(-1))
        t_emb = self.time_mlp(t_emb).unsqueeze(1)

        # Pass through the transformer layers.
        for block in self.blocks:
            x = block(x, t_emb, key_padding_mask=key_padding_mask)

        # Final normalization and vocabulary projection.
        hidden = self.ln_f(x)
        logits = self.head(hidden)
        return logits
