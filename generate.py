"""Generate and decode text samples from a trained AdaMask checkpoint.

The Config passed here must describe the same architecture (hidden_size,
heads, layers, context_length, steps, model_name) used to produce the
checkpoint, or state_dict loading will fail with a shape mismatch.
"""

import argparse
import torch

from adamask.config import Config
from adamask.model import MaskedDiffusionTransformer
from adamask.sample import sample


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text from a trained AdaMask checkpoint")
    parser.add_argument("checkpoint", help="Path to a .pt state_dict saved during training")
    parser.add_argument("--model-name", default="roberta-base")
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config(
        model_name=args.model_name,
        context_length=args.context_length,
        hidden_size=args.hidden_size,
        heads=args.heads,
        layers=args.layers,
        steps=args.steps,
    )
    if args.device:
        config.device = torch.device(args.device)

    model = MaskedDiffusionTransformer(config).to(config.device)
    state_dict = torch.load(args.checkpoint, map_location=config.device)
    model.load_state_dict(state_dict)

    tokens = sample(model, config, num_samples=args.num_samples, temperature=args.temperature)

    for i, row in enumerate(tokens.tolist()):
        text = config.tokenizer.decode(row, skip_special_tokens=False)
        print(f"--- sample {i} ---")
        print(text)
        print()


if __name__ == "__main__":
    main()
