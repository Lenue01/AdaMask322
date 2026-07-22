"""Main entrypoint for running AdaMask training."""

import argparse
import torch
from adamask.config import Config
from adamask.data import get_dataloader
from adamask.diffusion import TokenDifficulty
from adamask.model import MaskedDiffusionTransformer
from adamask.train import train


def parse_args():
    parser = argparse.ArgumentParser(description="AdaMask masked diffusion training")
    parser.add_argument("--model-name", default="roberta-base")
    parser.add_argument("--dataset-name", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset-config", default="sample-10BT")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--steps-per-epoch", type=int, default=8000)
    parser.add_argument("--warmup-steps", type=int, default=None, help="Default: 5%% of total steps")
    parser.add_argument("--save-every-epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=None, help="Default: auto-scaled from --hidden-size")
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None, help="Path to a checkpoint to resume training from")
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        batch_size=args.batch_size,
        context_length=args.context_length,
        hidden_size=args.hidden_size,
        heads=args.heads,
        layers=args.layers,
        steps=args.steps,
        num_epochs=args.num_epochs,
        steps_per_epoch=args.steps_per_epoch,
        warmup_steps=args.warmup_steps,
        save_every_epochs=args.save_every_epochs,
        lr=args.lr,
    )
    if args.device:
        config.device = torch.device(args.device)

    # Load data and initialize model + diffusion helper
    dataloader = get_dataloader(config)
    model = MaskedDiffusionTransformer(config).to(config.device)
    diffusion = TokenDifficulty(
        config.vocab_size,
        config.mask_token_id,
        config.pad_token_id,
        config.steps,
        config.device,
    )
    train(model, diffusion, dataloader, config, resume_path=args.resume)


if __name__ == "__main__":
    main()
