"""Masked diffusion and adaptive difficulty loss-weighting logic."""

import math
import torch
import torch.nn.functional as F


class MaskedDiffusion:
    """Manages a progressive masking schedule for diffusion-style training."""

    def __init__(self, steps, masked_token_id, pad_token_id, device):
        self.steps = steps  # number of diffusion steps
        self.masked_token_id = masked_token_id  # token used for masked positions
        self.pad_token_id = pad_token_id  # token used for padding
        self.device = device

        # Build a cosine schedule from t=0 to t=steps.
        # alpha[t] controls how much of the sequence is kept unmasked.
        t = torch.arange(steps + 1, dtype=torch.float32, device=device)
        alpha = (torch.cos(math.pi / 2 * t / self.steps) ** 2)
        self.alpha = alpha

    def mask_rate(self, t):
        """Return the fraction of tokens to mask at timestep t."""
        return 1.0 - self.alpha[t]

    def corrupt(self, tokens, t, generator=None):
        """Corrupt input tokens by replacing some positions with the mask token.

        Pass a seeded `generator` to get a reproducible corruption pattern (used
        for validation, so the same inputs are compared across epochs).
        """
        rate = self.mask_rate(t).view(-1, 1)
        noise = torch.rand(tokens.shape, device=tokens.device, generator=generator)
        is_masked = (noise < rate) & (tokens != self.pad_token_id)
        x_t = tokens.clone()
        x_t[is_masked] = self.masked_token_id
        return x_t, is_masked


class TokenDifficulty(MaskedDiffusion):
    """Tracks per-token difficulty so the training loss can be weighted toward harder tokens."""

    def __init__(self, vocab_size, masked_token_id, pad_token_id, steps, device, decay=0.999):
        super().__init__(steps, masked_token_id, pad_token_id, device)
        # Track how many times each token has been seen in masked positions.
        self.total = torch.zeros(vocab_size, device=device)
        # Track the running sum of cross-entropy loss for each token when masked.
        self.loss_sum = torch.zeros(vocab_size, device=device)
        # Decay applied to both accumulators before each update, turning the
        # lifetime average into a recency-weighted one (effective window is
        # roughly 1 / (1 - decay) updates) -- otherwise loss from early training,
        # when the whole model is bad, permanently biases which tokens look hard.
        self.decay = decay

    def update(self, logits, tokens, is_masked):
        """Update difficulty statistics from the model's masked-token predictions."""
        masked_logits = logits[is_masked]
        target = tokens[is_masked]
        if target.numel() == 0:
            return

        # Per-token cross-entropy: higher loss means the token is harder.
        per_token_loss = F.cross_entropy(masked_logits, target, reduction="none")

        # Decay existing stats, then accumulate this batch's counts/loss on top.
        # Dividing two accumulators decayed by the same factor each step is
        # exactly the recursive form of an EMA, so loss_sum / total is still a
        # correct (now recency-weighted) mean.
        self.total *= self.decay
        self.loss_sum *= self.decay
        self.total.scatter_add_(0, target, torch.ones_like(target, dtype=torch.float32))
        self.loss_sum.scatter_add_(0, target, per_token_loss)

    def get_difficulty(self, tokens):
        """Return the mean cross-entropy loss for each token (higher = harder)."""
        return self.loss_sum[tokens] / (self.total[tokens] + 1e-8)
