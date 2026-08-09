"""Fast end-to-end sanity check for the AdaMask design.

Trains a small model on the TinyStories corpus for a short run, then
samples from it and prints decoded text. This is meant to answer one
question quickly (minutes, not hours, on a Colab GPU): does the
architecture + masking + sampling pipeline actually learn to produce
plausible English, before committing to the full-scale config in main.py?

TinyStories (roneneldan/TinyStories) is used here instead of wikitext-2:
wikitext's raw text keeps wiki markup artifacts (`= = = Heading = = =`
section headers, `@-@`-joined punctuation), which is noise a sanity check
doesn't need. TinyStories is plain, grammatical, simple-vocabulary prose
generated to be readable by a small model, so learning progress (or its
absence) is easy to see in the decoded samples. It's also a convenient
size for this purpose: ~2.1M train rows / ~22k validation rows, small
enough to stream through quickly without the hours-long download of the
full-scale run's fineweb-edu sample.

With this little training time, expect simple, sometimes repetitive
sentences rather than fully coherent stories. That is fine - the goal
here is a pipeline sanity check, not a quality benchmark.
"""

import argparse
import torch

from adamask.config import Config
from adamask.data import get_dataloader
from adamask.diffusion import TokenDifficulty
from adamask.model import MaskedDiffusionTransformer
from adamask.sample import sample
from adamask.train import train


def parse_args():
    parser = argparse.ArgumentParser(description="AdaMask sanity check (small-scale run)")
    parser.add_argument("--dataset-name", default="roneneldan/TinyStories")
    parser.add_argument("--dataset-config", default="default")
    parser.add_argument("--split", default="train")
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=1500)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--save-every-epochs", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--difficulty-loss-scale", type=float, default=0.3, help="Loss weight = 1 + scale * difficulty for hard tokens")
    parser.add_argument("--difficulty-decay", type=float, default=0.999, help="EMA decay for token difficulty stats; higher = longer memory")
    parser.add_argument("--val-split", default="validation")
    parser.add_argument("--val-batches", type=int, default=16)
    parser.add_argument("--val-seed", type=int, default=1234)
    parser.add_argument("--lr", type=float, default=None, help="Default: auto-scaled from --hidden-size")
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None, help="Path to a checkpoint to resume training from")
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        context_length=args.context_length,
        hidden_size=args.hidden_size,
        heads=args.heads,
        layers=args.layers,
        steps=args.steps,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        steps_per_epoch=args.steps_per_epoch,
        warmup_steps=args.warmup_steps,
        save_every_epochs=args.save_every_epochs,
        max_workers=args.max_workers,
        difficulty_loss_scale=args.difficulty_loss_scale,
        difficulty_decay=args.difficulty_decay,
        val_split=args.val_split,
        val_batches=args.val_batches,
        val_seed=args.val_seed,
        lr=args.lr,
    )
    if args.device:
        config.device = torch.device(args.device)

    dataloader = get_dataloader(config)
    model = MaskedDiffusionTransformer(config).to(config.device)
    diffusion = TokenDifficulty(
        config.vocab_size, config.mask_token_id, config.pad_token_id, config.steps, config.device,
        decay=config.difficulty_decay,
    )

    train(model, diffusion, dataloader, config, resume_path=args.resume)

    print("\n=== Sanity check samples ===")
    tokens = sample(model, diffusion, config, num_samples=4, temperature=0.9)
    for i, row in enumerate(tokens.tolist()):
        text = config.tokenizer.decode(row, skip_special_tokens=False)
        print(f"--- sample {i} ---")
        print(text)
        print()


if __name__ == "__main__":
    main()
