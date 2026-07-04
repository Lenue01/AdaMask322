"""Fast end-to-end sanity check for the AdaMask design.

Trains a small model on the small wikitext-2 corpus for a short run, then
samples from it and prints decoded text. This is meant to answer one
question quickly (minutes, not hours, on a Colab GPU): does the
architecture + masking + sampling pipeline actually learn to produce
plausible English, before committing to the full-scale config in main.py?

With this little data and training time, expect memorized/repeated
fragments from wikitext-2 rather than novel fluent prose. That is fine -
the goal here is a pipeline sanity check, not a quality benchmark.
"""

from adamask.config import Config
from adamask.data import get_dataloader
from adamask.diffusion import TokenDifficulty
from adamask.model import MaskedDiffusionTransformer
from adamask.sample import sample
from adamask.train import train


def main():
    config = Config(
        dataset_name="Salesforce/wikitext",
        dataset_config="wikitext-2-raw-v1",
        split="train",
        context_length=64,
        hidden_size=256,
        heads=4,
        layers=4,
        steps=16,
        batch_size=32,
        num_epochs=1,
        steps_per_epoch=1500,
        warmup_steps=100,
        save_every_epochs=1,
        max_workers=2,
    )

    dataloader = get_dataloader(config)
    model = MaskedDiffusionTransformer(config).to(config.device)
    diffusion = TokenDifficulty(
        config.vocab_size, config.mask_token_id, config.pad_token_id, config.steps, config.device
    )

    train(model, diffusion, dataloader, config)

    print("\n=== Sanity check samples ===")
    tokens = sample(model, config, num_samples=4, temperature=0.9)
    for i, row in enumerate(tokens.tolist()):
        text = config.tokenizer.decode(row, skip_special_tokens=False)
        print(f"--- sample {i} ---")
        print(text)
        print()


if __name__ == "__main__":
    main()
