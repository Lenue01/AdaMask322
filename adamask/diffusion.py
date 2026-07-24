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

    def corrupt(self, tokens, t):
        """Corrupt input tokens by replacing some positions with the mask token."""
        rate = self.mask_rate(t).view(-1, 1)
        noise = torch.rand(tokens.shape, device=tokens.device)
        is_masked = (noise < rate) & (tokens != self.pad_token_id)
        x_t = tokens.clone()
        x_t[is_masked] = self.masked_token_id
        return x_t, is_masked


class TokenDifficulty(MaskedDiffusion):
    """Tracks per-token difficulty so the training loss can be weighted toward harder tokens."""

    def __init__(self, vocab_size, masked_token_id, pad_token_id, steps, device):
        super().__init__(steps, masked_token_id, pad_token_id, device)
        # Track how many times each token has been seen in masked positions.
        self.total = torch.ones(vocab_size, device=device)
        # Track how often the model predicted each token correctly.
        self.correct = torch.full((vocab_size,), 0.5, device=device)

    def update(self, logits, tokens, is_masked):
        """Update difficulty statistics from the model's masked-token predictions."""
        masked_logits = logits[is_masked]
        target = tokens[is_masked]
        if target.numel() == 0:
            return

        # Predicted token IDs for masked positions.
        predictions = masked_logits.argmax(dim=-1)
        correct = (predictions == target).to(dtype=torch.float32)

        # Increment counters for seen tokens and correct predictions.
        self.total.scatter_add_(0, target, torch.ones_like(target, dtype=torch.float32))
        self.correct.scatter_add_(0, target, correct)

    def get_difficulty(self, tokens):
        """Return a difficulty score for each token in the input."""
        return 1.0 - (self.correct[tokens] / (self.total[tokens] + 1e-8))
