"""Project configuration for AdaMask, including tokenizer and device setup."""

import math
from dataclasses import dataclass, field
from typing import Optional
import torch
from transformers import AutoTokenizer

# Reference point the auto-scaling heuristics below are anchored to: the
# hidden_size/lr pair that main.py's full-scale config was tuned around.
_REFERENCE_HIDDEN_SIZE = 1024
_REFERENCE_LR = 1e-4


@dataclass
class Config:
    """Contains settings used across training, model, and data modules."""
    # Model and tokenizer selection
    model_name: str = "roberta-base"  # Pretrained tokenizer/model name
    dataset_name: str = "HuggingFaceFW/fineweb-edu"  # HF dataset name
    dataset_config: str = "sample-10BT"  # HF dataset configuration
    split: str = "train"  # dataset split to use

    # Sequence length and model dimensions
    context_length: int = 128  # number of tokens per example
    hidden_size: int = 1024  # model hidden dimension size
    heads: int = 16  # number of attention heads
    layers: int = 16  # number of transformer layers
    steps: int = 64  # diffusion timesteps / masking levels

    # Training hyperparameters
    batch_size: int = 128
    lr: Optional[float] = None  # peak LR; None auto-scales from hidden_size (see __post_init__)
    weight_decay: float = 0.01
    accum_steps: int = 1  # gradient accumulation steps
    num_epochs: int = 50
    save_every_epochs: int = 2
    warmup_steps: Optional[int] = None  # None defaults to 5% of total_steps
    steps_per_epoch: int = 8000
    total_steps: int = field(init=False)
    max_workers: int = 4

    # Device and tokenizer fields initialized after construction.
    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer: AutoTokenizer = field(init=False)
    vocab_size: int = field(init=False)
    mask_token_id: int = field(init=False)
    pad_token_id: int = field(init=False)

    def __post_init__(self):
        """Load tokenizer and fill in dependent configuration values."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.vocab_size = len(self.tokenizer)
        self.mask_token_id = self.tokenizer.mask_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        # Keep the LR schedule's horizon consistent with how long training actually runs.
        self.total_steps = self.num_epochs * self.steps_per_epoch

        # A fixed warmup step count stops matching the schedule once num_epochs or
        # steps_per_epoch change (e.g. a short sanity-check run vs. a full-scale one),
        # so default it to a fraction of total_steps instead of an absolute constant.
        if self.warmup_steps is None:
            self.warmup_steps = max(1, round(0.05 * self.total_steps))

        # Wider models need a smaller peak LR to stay stable; this heuristic scales
        # LR ~ 1/sqrt(hidden_size), anchored at the tuned 1024-wide/1e-4 default, so
        # changing --hidden-size doesn't silently leave an unsuitable LR in place.
        if self.lr is None:
            self.lr = _REFERENCE_LR * math.sqrt(_REFERENCE_HIDDEN_SIZE / self.hidden_size)
