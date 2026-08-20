"""Generate and decode text samples from a trained AdaMask checkpoint.

Checkpoints saved after config tracking was added carry their own
architecture (hidden_size, heads, layers, context_length, steps, model_name)
and that's used automatically. Any of the flags below can still override it
explicitly; for a checkpoint saved before config tracking (no "config" key)
those flags are the only way to specify the right architecture, and getting
them wrong will fail with a shape mismatch (or, for --steps specifically,
load fine but silently sample against the wrong schedule).
"""

import argparse
import torch

from adamask.config import Config
from adamask.diffusion import MaskedDiffusion
from adamask.model import MaskedDiffusionTransformer
from adamask.sample import sample

_ARCH_FIELDS = ["model_name", "context_length", "hidden_size", "heads", "layers", "steps"]
# Fallback only for checkpoints saved before config tracking was added.
_LEGACY_DEFAULTS = {
    "model_name": "roberta-base", "context_length": 128,
    "hidden_size": 1024, "heads": 16, "layers": 16, "steps": 64,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text from a trained AdaMask checkpoint")
    parser.add_argument("checkpoint", help="Path to a .pt training checkpoint (as saved by save_checkpoint)")
    parser.add_argument("--model-name", default=None, help="Default: from checkpoint, else roberta-base")
    parser.add_argument("--context-length", type=int, default=None, help="Default: from checkpoint, else 128")
    parser.add_argument("--hidden-size", type=int, default=None, help="Default: from checkpoint, else 1024")
    parser.add_argument("--heads", type=int, default=None, help="Default: from checkpoint, else 16")
    parser.add_argument("--layers", type=int, default=None, help="Default: from checkpoint, else 16")
    parser.add_argument("--steps", type=int, default=None, help="Default: from checkpoint, else 64")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--remask-threshold", type=float, default=0.2, help="Remask an already-revealed position if the model's confidence in its current token drops below this")
    parser.add_argument("--max-remask-frac", type=float, default=0.1, help="Max fraction of currently-revealed positions eligible for remasking per step")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    ckpt_config = checkpoint.get("config")
    if ckpt_config is None:
        print(f"NOTE: {args.checkpoint} predates config tracking; using CLI flags / legacy defaults for architecture.")

    resolved = {}
    for field in _ARCH_FIELDS:
        cli_val = getattr(args, field)
        if cli_val is not None:
            resolved[field] = cli_val
        elif ckpt_config is not None and field in ckpt_config:
            resolved[field] = ckpt_config[field]
        else:
            resolved[field] = _LEGACY_DEFAULTS[field]

    config = Config(**resolved)
    config.device = device

    model = MaskedDiffusionTransformer(config).to(config.device)
    model.load_state_dict(checkpoint["model"])

    # Only the mask-rate schedule is needed here (no difficulty tracking at
    # inference time), so a plain MaskedDiffusion is enough.
    diffusion = MaskedDiffusion(config.steps, config.mask_token_id, config.pad_token_id, config.device)
    tokens = sample(
        model, diffusion, config, num_samples=args.num_samples, temperature=args.temperature,
        remask_threshold=args.remask_threshold, max_remask_frac=args.max_remask_frac,
    )

    for i, row in enumerate(tokens.tolist()):
        text = config.tokenizer.decode(row, skip_special_tokens=False)
        print(f"--- sample {i} ---")
        print(text)
        print()


if __name__ == "__main__":
    main()
