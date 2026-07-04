"""Data loading and tokenization for AdaMask."""

from datasets import load_dataset
from torch.utils.data import DataLoader


def _tokenize_batch(examples, tokenizer, config):
    """Convert a batch of raw text examples into fixed-length token blocks.

    Must run with batched=True: each input document can expand into zero,
    one, or several chunks, so the output row count differs from the input
    row count. With batched=False, datasets.map keeps a 1:1 row mapping and
    would nest each document's chunks inside a single ragged row instead of
    flattening them into independent fixed-length training examples.
    """
    all_ids = tokenizer(examples["text"], add_special_tokens=False)["input_ids"]

    chunks = []
    masks = []

    for ids in all_ids:
        # Break the token list into non-overlapping windows of context_length.
        for i in range(0, len(ids), config.context_length):
            chunk = ids[i:i + config.context_length]

            # Ignore chunks that are too short, to avoid unstable very small examples.
            if len(chunk) < 16:
                continue

            # Pad the chunk to the full context length using the tokenizer pad token.
            pad_len = config.context_length - len(chunk)
            chunks.append(chunk + [tokenizer.pad_token_id] * pad_len)

            # Create an attention mask: 1 for real tokens, 0 for padding.
            masks.append([1] * len(chunk) + [0] * pad_len)

    return {"input_ids": chunks, "attention_mask": masks}


def get_dataloader(config):
    """Build a PyTorch DataLoader from the dataset configuration."""
    dataset = load_dataset(config.dataset_name, name=config.dataset_config, split=config.split)

    # Shuffle the raw dataset before tokenization for randomness.
    dataset = dataset.shuffle(seed=42)

    # Tokenize in batches so each document's chunks become independent rows.
    dataset = dataset.map(
        lambda batch: _tokenize_batch(batch, config.tokenizer, config),
        batched=True,
        remove_columns=dataset.column_names,
    )

    # Convert dataset examples to PyTorch tensors.
    dataset = dataset.with_format("torch")

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.max_workers,
        pin_memory=True,
        shuffle=True,
    )
