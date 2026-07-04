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
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        batch_size=args.batch_size,
        context_length=args.context_length,
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
    train(model, diffusion, dataloader, config)


if __name__ == "__main__":
    main()
