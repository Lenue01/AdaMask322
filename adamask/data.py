"""Data loading and tokenization for AdaMask."""

from datasets import load_dataset
from torch.utils.data import DataLoader


def _tokenize_example(example, tokenizer, config):
    """Convert a raw text example into fixed-length token blocks."""
    # Tokenize the raw text string into token IDs without adding special tokens.
    ids = tokenizer(example["text"], add_special_tokens=False)["input_ids"]

    chunks = []
    masks = []

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

    # Tokenize each example one by one and remove the original text columns.
    dataset = dataset.map(
        lambda example: _tokenize_example(example, config.tokenizer, config),
        batched=False,
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
